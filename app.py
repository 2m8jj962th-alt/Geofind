import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI

load_dotenv()

app = FastAPI(title="GeoFind AI")
templates = Jinja2Templates(directory=".")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "8"))

def data_url(data: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{base64.b64encode(data).decode()}"

def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not m:
            raise ValueError("AI did not return JSON")
        return json.loads(m.group(1))

async def google_text_search(query: str):
    if not GOOGLE_MAPS_API_KEY:
        return []
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "places.id,places.displayName,places.formattedAddress,"
            "places.location,places.photos,places.googleMapsUri"
        ),
        "Content-Type": "application/json",
    }
    payload = {"textQuery": query, "pageSize": 5}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json().get("places", [])

async def streetview_metadata(lat: float, lng: float):
    if not GOOGLE_MAPS_API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": GOOGLE_MAPS_API_KEY}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def streetview_image(lat: float, lng: float, heading: int, fov: int = 80):
    if not GOOGLE_MAPS_API_KEY:
        return None
    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "heading": heading,
        "pitch": 0,
        "fov": fov,
        "key": GOOGLE_MAPS_API_KEY,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.content

def analyze_image(image_b64_url: str):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = """
You are the visual reconnaissance engine of a worldwide geolocation system.
Analyze this single street/building image without assuming any candidate city.

Extract:
- visible text/signs (exact transcription if possible)
- architecture styles and distinctive building features
- facade materials, colors, window/door patterns
- roof/gable/tower shapes
- road geometry, slope and lane markings
- parking/signage/traffic conventions
- street furniture, lamps, utility infrastructure
- vegetation, weather, season
- vehicle/license-plate clues
- likely continent/cultural region only when evidence supports it

Then propose 12-20 distinctive search phrases that could locate the exact building or street.
Return ONLY valid JSON:
{
  "observations": [...],
  "ocr": [...],
  "search_queries": [...],
  "regional_hypotheses": [{"region": "...", "reason": "...", "weight": 0-1}]
}
"""
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_b64_url, "detail": "high"},
            ],
        }],
    )
    return extract_json(resp.output_text)

def worldwide_candidates(analysis: dict):
    client = OpenAI(api_key=OPENAI_API_KEY)
    compact = json.dumps(analysis, ensure_ascii=False)
    prompt = f"""
You are the candidate-generation engine for a world-wide visual geolocation app.

Given these observations:
{compact}

Search the public web broadly. Do NOT start from a fixed list of cities.
Use the distinctive architectural clues, OCR, building descriptions and combinations
of neighboring-building features. Consider every continent. Search for actual named
buildings/streets, not generic architecture pages.

Return ONLY valid JSON:
{{
  "candidates": [
    {{
      "name": "building or street",
      "city": "city",
      "country": "country",
      "address": "best known address, if available",
      "why": "specific visual match",
      "confidence": 0-1,
      "source_urls": ["https://..."]
    }}
  ]
}}

Return up to {MAX_CANDIDATES} strongest candidates. Do not invent addresses.
"""
    resp = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
        input=prompt,
    )
    return extract_json(resp.output_text)

def compare_images(source_url: str, candidate_images: list[bytes], candidate_meta: dict):
    if not candidate_images:
        return {"score": 0, "reason": "No Street View imagery available."}
    client = OpenAI(api_key=OPENAI_API_KEY)
    content = [
        {"type": "input_text", "text": f"""
Compare the user's source image to these Street View views of this candidate:
{json.dumps(candidate_meta, ensure_ascii=False)}

Score how likely they depict the SAME physical street/building scene.
Focus on fixed geometry: facade shape, window placement, roofline, neighboring
buildings, street width/slope, trees and other permanent features. Ignore weather,
cars and temporary objects.

Return ONLY JSON:
{{"score": 0-100, "reason": "specific evidence", "matched_features": ["..."]}}
"""},
        {"type": "input_image", "image_url": source_url, "detail": "high"},
    ]
    for img in candidate_images:
        content.append({"type": "input_image", "image_url": data_url(img, "image/jpeg"), "detail": "high"})
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "user", "content": content}],
    )
    return extract_json(resp.output_text)

async def run_search(image_bytes: bytes, content_type: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY puuttuu.")
    source = data_url(image_bytes, content_type)
    analysis = analyze_image(source)
    candidates_raw = worldwide_candidates(analysis)
    candidates = candidates_raw.get("candidates", [])[:MAX_CANDIDATES]

    verified = []
    for c in candidates:
        places = await google_text_search(
            f"{c.get('name','')} {c.get('address','')} {c.get('city','')} {c.get('country','')}"
        )
        if not places:
            c["verification"] = {"score": 0, "reason": "Google Places did not resolve a location."}
            verified.append(c)
            continue

        best = places[0]
        loc = best.get("location", {})
        lat, lng = loc.get("latitude"), loc.get("longitude")
        if lat is None or lng is None:
            c["verification"] = {"score": 0, "reason": "No coordinates."}
            verified.append(c)
            continue

        meta = await streetview_metadata(lat, lng)
        imgs = []
        headings = [0, 90, 180, 270]
        if meta and meta.get("status") == "OK":
            for h in headings:
                try:
                    img = await streetview_image(lat, lng, h)
                    if img:
                        imgs.append(img)
                except Exception:
                    pass

        verification = compare_images(
            source,
            imgs,
            {
                "candidate": c,
                "resolved_address": best.get("formattedAddress"),
                "google_maps_uri": best.get("googleMapsUri"),
                "coordinates": {"lat": lat, "lng": lng},
                "streetview_date": (meta or {}).get("date"),
            },
        ) if imgs else {"score": 0, "reason": "No Street View panorama at resolved point."}

        c["resolved"] = {
            "address": best.get("formattedAddress"),
            "lat": lat,
            "lng": lng,
            "google_maps_uri": best.get("googleMapsUri"),
            "streetview_date": (meta or {}).get("date"),
        }
        c["verification"] = verification
        verified.append(c)

    verified.sort(
        key=lambda x: (
            x.get("verification", {}).get("score", 0),
            x.get("confidence", 0),
        ),
        reverse=True,
    )
    return {
        "analysis": analysis,
        "results": verified,
        "note": "Tulokset ovat todennäköisyyksiä; sovellus ei saa väittää varmaa sijaintia ilman fyysisten piirteiden täsmäosumaa."
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "Tyhjä tiedosto."}, status_code=400)
    try:
        result = await run_search(data, file.content_type or "image/jpeg")
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
