import os
import base64
import httpx
import time
import asyncio
from google import genai
from google.genai import types

GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
POLLINATIONS_KEY = os.environ.get("POLLINATIONS_KEY", "")

client = genai.Client(api_key=GEMINI_API_KEY)

STYLE_PREFIX = (
    "soft watercolor illustration, warm nostalgic palette, "
    "children's book artistry, golden hour lighting, loose brushwork. "
    "Medium-wide composition with comfortable headroom above all subjects. "
    "Every person in the scene must have a complete head, fully visible expressive face, neck, and torso comfortably inside the frame. "
    "Never crop or cut off heads or faces. Complete human figures, anatomically correct. No text or words in the image. "
)

_last_request_time = 0

async def generate_illustration(vignette: dict) -> str:
    global _last_request_time
    import urllib.parse
    desc = vignette.get("illustration_prompt") or vignette.get("scene_description") or "A warm nostalgic memory scene"
    people = vignette.get("key_people")
    people_clause = ""
    if isinstance(people, list) and people:
        people_clause = f"Featuring {', '.join(people)}, each with complete heads and faces visible. "
    prompt = f"{STYLE_PREFIX}{people_clause}{desc}"

    elapsed = time.time() - _last_request_time
    if elapsed < 35:
        wait_time = 35 - elapsed
        print(f"Rate limiting Pollinations API... waiting {wait_time:.1f} seconds")
        await asyncio.sleep(wait_time)

    encoded_prompt = urllib.parse.quote(prompt[:900])

    if POLLINATIONS_KEY:
        try:
            url = f"https://gen.pollinations.ai/image/{encoded_prompt}"
            params = {
                "model":   "flux",
                "width":   "800",
                "height":  "600",
                "enhance": "false",
                "key":     POLLINATIONS_KEY,
            }
            _last_request_time = time.time()
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
                response = await http.get(url, params=params)
                if response.status_code == 200 and "image" in response.headers.get("content-type",""):
                    b64  = base64.b64encode(response.content).decode()
                    mime = response.headers.get("content-type","image/jpeg").split(";")[0]
                    print(f"✅ Pollinations (key) image generated")
                    return f"data:{mime};base64,{b64}"
                else:
                    print(f"Pollinations key error: {response.status_code} {response.text[:100]}")
        except Exception as e:
            print(f"Pollinations key error: {e}")

    # Try Pollinations free (works in browser fetch, blocked in Cloud Shell)
    try:
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=800&height=600&nologo=true&seed=42"
        _last_request_time = time.time()
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as http:
            response = await http.get(url)
            if response.status_code == 200 and "image" in response.headers.get("content-type",""):
                b64  = base64.b64encode(response.content).decode()
                mime = response.headers.get("content-type","image/jpeg").split(";")[0]
                return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Pollinations free error: {e}")

    return _fallback_placeholder(vignette.get("emotion", "nostalgia"))


def _fallback_placeholder(emotion: str) -> str:
    palettes = {
        "joy":       ("FFF9C4", "F9A825"),
        "nostalgia": ("FCE4EC", "C2185B"),
        "love":      ("F3E5F5", "7B1FA2"),
        "wonder":    ("E3F2FD", "1565C0"),
        "pride":     ("E8F5E9", "2E7D32"),
    }
    bg, fg = palettes.get(emotion, ("F5F0E8", "8B7355"))
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='400' height='300'>
      <rect width='400' height='300' fill='#{bg}'/>
      <text x='200' y='145' font-family='Georgia' font-size='14' fill='#{fg}' text-anchor='middle'>✦ Illustration loading ✦</text>
      <text x='200' y='170' font-family='Georgia' font-size='11' fill='#{fg}88' text-anchor='middle'>{emotion}</text>
    </svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()