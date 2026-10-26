import os
import json
from google import genai
from google.genai import types

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        _client = genai.Client(api_key=api_key)
    return _client

# ---------------------------------------------------------------------------
# Stage 1: Memory Completeness Assessment
# ---------------------------------------------------------------------------
COMPLETENESS_PROMPT = """
You are a memory-completeness evaluator for a personal memoir book system.
Analyze the conversation below between an interviewer and a storyteller.
Determine whether a COMPLETE, vivid memory has been shared.

A memory is COMPLETE when it contains ALL FOUR of these elements:
1. SETTING — a place, location, or environment (e.g. "on the veranda", "in the kitchen")
2. PEOPLE or OBJECTS — at least one person, animal, or meaningful object
3. EMOTION — a feeling or emotional tone (can be implicit)
4. SENSORY / NARRATIVE DETAIL — a specific action, dialogue, sensation, or moment

A memory is INCOMPLETE when:
- The user has only shared a vague or general statement ("I had a happy childhood")
- The user is still mid-story and adding new details in a continuing flow
- Key elements (setting, people, emotion, or sensory detail) are missing
- The user seems to be warming up and has not yet gotten to the vivid core

IMPORTANT:
- Be patient. Wait for the story to develop fully before marking it complete.
- If the user shared 3+ substantial messages building one story, it is likely complete.
- A single short sentence is almost never a complete memory.
- If in doubt, mark as INCOMPLETE — it is better to ask one more question than to capture too early.

Return ONLY valid JSON:
{
  "is_complete": true/false,
  "has_setting": true/false,
  "has_people": true/false,
  "has_emotion": true/false,
  "has_detail": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why complete or not"
}

Conversation:
"""

async def assess_memory_completeness(conversation_messages: list[dict]) -> dict:
    """
    Evaluate whether the current conversation contains a complete memory.
    conversation_messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns: {"is_complete": bool, "confidence": float, ...}
    """
    # Need at least 2 user messages before even considering completeness
    user_msgs = [m for m in conversation_messages if m["role"] == "user"]
    if len(user_msgs) < 2:
        return {
            "is_complete": False,
            "has_setting": False,
            "has_people": False,
            "has_emotion": False,
            "has_detail": False,
            "confidence": 0.0,
            "reasoning": "Too few user messages to form a complete memory."
        }

    # Format conversation for the evaluator
    convo_text = ""
    for m in conversation_messages:
        role_label = "Storyteller" if m["role"] == "user" else "Interviewer"
        convo_text += f"{role_label}: {m['content']}\n"

    try:
        response = await get_client().aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=COMPLETENESS_PROMPT + convo_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,
            )
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        # Only consider complete if confidence is above threshold
        if result.get("confidence", 0) < 0.7:
            result["is_complete"] = False
        return result
    except Exception as e:
        print(f"[COMPLETENESS] Error: {e}")
        return {
            "is_complete": False,
            "confidence": 0.0,
            "reasoning": f"Evaluation failed: {e}"
        }


# ---------------------------------------------------------------------------
# Stage 2: Literary Vignette Extraction
# ---------------------------------------------------------------------------
VIGNETTE_PROMPT = """
You are a gifted literary memoirist and visual storyteller.
Your task is to transform a raw conversational memory into a beautiful, polished page
for an illustrated personal memoir book.

CRITICAL RULES FOR NARRATION:
- Do NOT copy the user's words verbatim. REWRITE the memory completely.
- Write in FIRST PERSON, past tense, as the storyteller.
- Use rich, evocative, literary memoir prose — the kind found in award-winning memoirs.
- Compress the entire memory into a vivid 2–3 sentence VIGNETTE.
- Include sensory details: smells, textures, sounds, light, warmth.
- The narration should feel like a polished excerpt from a published memoir, not a transcript.

EXAMPLE of the transformation you must perform:

USER INPUT: "I remember sitting on the veranda during the rain while my grandmother made bhajias."

YOUR NARRATION: "On monsoon afternoons our veranda became a small island of warmth. While rain hammered the tin roof, my grandmother fried hot bhajias and the whole house smelled of spices and comfort."

Return ONLY valid JSON with these exact fields:
{
  "narration": "2-3 sentence literary memoir narration. First person, past tense. Evocative, beautiful prose. NOT a copy of the user's words.",
  "scene_description": "Detailed visual description for an illustrator: setting, time of day, objects, all people present, atmosphere, era-appropriate details. 2-3 sentences.",
  "illustration_prompt": "Prompt for an image generation model. Style: soft watercolor illustration, medium-wide shot with clear headroom above all subjects. Every person mentioned MUST have a complete head, fully visible expressive face, neck, and body comfortably within the frame. Never crop or cut off heads. Under 100 words.",
  "era": "Approximate decade (e.g. '1950s', '1970s', 'early 2000s')",
  "emotion": "Primary emotion (e.g. 'joy', 'nostalgia', 'love', 'wonder', 'pride', 'warmth')",
  "key_people": ["list", "of", "people", "mentioned"],
  "chapter": "Life chapter (childhood, family, work, love, milestone, everyday)"
}

Here is the raw conversation to transform. Focus only on the STORYTELLER's words:
"""

async def extract_vignette(conversation_messages: list[dict]) -> dict:
    """
    Transform raw conversation messages into a polished literary vignette.
    conversation_messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns: full vignette dict with literary narration, scene description, etc.
    """
    # Build story context from conversation
    convo_text = ""
    for m in conversation_messages:
        role_label = "Storyteller" if m["role"] == "user" else "Interviewer"
        convo_text += f"{role_label}: {m['content']}\n"

    try:
        response = await get_client().aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=VIGNETTE_PROMPT + convo_text,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.8,
            )
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[VIGNETTE] Error: {e}")
        # Fallback: use raw user text
        user_text = " ".join(
            m["content"] for m in conversation_messages if m["role"] == "user"
        )
        return {
            "narration": user_text[:300],
            "scene_description": user_text[:200],
            "illustration_prompt": f"Soft watercolor illustration of a nostalgic memory: {user_text[:100]}",
            "era": "unknown",
            "emotion": "nostalgia",
            "key_people": [],
            "chapter": "memory"
        }


# ---------------------------------------------------------------------------
# Combined Pipeline: Assess + Extract (used by both text and voice)
# ---------------------------------------------------------------------------
async def capture_memory_pipeline(conversation_messages: list[dict]) -> dict | None:
    """
    Unified memory capture pipeline for both text and voice inputs.

    1. Assess if the memory is complete
    2. If complete, extract a literary vignette
    3. Return the vignette dict or None if memory isn't ready

    conversation_messages: list of {"role": "user"|"assistant", "content": "..."}
    """
    print(f"[PIPELINE] Assessing memory completeness ({len(conversation_messages)} messages)...")

    completeness = await assess_memory_completeness(conversation_messages)
    print(f"[PIPELINE] Completeness result: complete={completeness.get('is_complete')}, "
          f"confidence={completeness.get('confidence', 0):.2f}, "
          f"reason={completeness.get('reasoning', '')[:100]}")

    if not completeness.get("is_complete"):
        return None

    print(f"[PIPELINE] ✅ Memory is complete! Extracting literary vignette...")
    vignette = await extract_vignette(conversation_messages)
    print(f"[PIPELINE] Vignette narration: {vignette.get('narration', '')[:120]}...")

    return vignette