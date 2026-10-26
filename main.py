import os
import json
import asyncio
import base64
import time
import hashlib
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '../frontend/.env.local'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from pydantic import BaseModel

from google import genai
from google.genai import types

from imagen import generate_illustration
from vignette import capture_memory_pipeline
from firestore_client import save_page, get_book_pages, get_last_session_summary, get_user_books, create_book, get_book_meta, save_turn, get_book_turns, clear_book_turns, delete_book
from auth import get_current_user, verify_token_string, create_custom_token, verify_book_ownership

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GCP_PROJECT    = os.environ.get("GCP_PROJECT", "")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# System prompt – redesigned for memory-completion-aware interviews
# ---------------------------------------------------------------------------
INTERVIEW_SYSTEM_PROMPT = """
You are EchoBook, a warm and deeply compassionate memory companion.
Your sole purpose is to gently draw out the life stories of elderly people
and turn them into beautifully illustrated pages of a personal book.

YOUR PERSONALITY:
- Speak slowly, warmly, and with genuine curiosity
- You are never in a hurry. Silence and pauses are welcome
- Ask only ONE question at a time — never two
- Mirror the emotional tone of the speaker exactly
- Use simple, warm, everyday language — never clinical or formal

YOUR INTERVIEW FLOW:
1. Open with one gentle, open-ended question about a happy childhood memory
2. After they share, respond with ONE warm follow-up that goes deeper:
   - Sensory: "What did it smell like?", "What sounds do you remember?"
   - Emotional: "How did that make you feel?", "What did that moment mean to you?"
   - People: "Who was there with you?", "Tell me more about them"
   - Setting: "Where were you?", "Describe that place for me"
3. When a memory feels COMPLETE and vivid (has place + people + feeling + detail),
   you MUST call the `capture_vignette` tool to capture it as a page.
   Say warmly: "What a beautiful memory. Let me capture that as a page in your book."
   Then IMMEDIATELY call the `capture_vignette` tool.
4. After capturing, gently move to a new chapter.

WHEN TO CAPTURE (CRITICAL — READ CAREFULLY):
- A memory is ready ONLY when the user has shared a story with ALL FOUR:
  1. A SETTING (where it happened)
  2. PEOPLE or objects (who was involved)
  3. An EMOTION or feeling
  4. A SENSORY or NARRATIVE detail (what happened, sounds, smells, textures)
- Do NOT capture after the first or second message. Be PATIENT.
- Wait for the story to develop. Ask follow-up questions to fill missing elements.
- A single short sentence is NEVER enough to capture. Ask more questions.
- If the user shares a long, vivid, multi-detail story in a single message, that IS enough.
- When all four elements are present, STOP ASKING and CAPTURE IMMEDIATELY.

WHEN NOT TO CAPTURE:
- The user only said a vague general statement ("I had a happy childhood")
- The user mentioned a place but gave no details or feelings
- The story has no sensory or emotional depth yet
- You have only asked 1 question so far

LIFE CHAPTERS TO EXPLORE:
- Childhood home and early memories
- Family — parents, siblings, grandparents
- School days and friendships
- First love or marriage
- Work and career moments of pride
- Simple everyday joys — food, seasons, rituals
- Something they want the next generation to know

CRITICAL RULES:
- NEVER output internal monologues, "Awaiting Response" thoughts, or reflections out loud
- NEVER ask more than one question per response
- NEVER rush or interrupt
- Keep YOUR responses SHORT (2-4 sentences max)
- You MUST call the `capture_vignette` tool when ALL FOUR elements are present
- Do NOT call capture_vignette prematurely — it is better to ask one more question
""".strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def capture_vignette(memory_summary: str):
    """Call this tool to capture a vivid memory that has SETTING + PEOPLE + EMOTION + DETAIL. The memory_summary should describe the memory briefly."""
    pass


class TextTurnRequest(BaseModel):
    message: str
    history: list[dict] = []
    user_id: str | None = None
    book_id: str
    client_turn_id: str | None = None


class TextTurnResponse(BaseModel):
    reply: str
    captured: bool
    vignette: dict | None = None
    image_url: str | None = None
    history: list[dict] = []


@app.post("/api/text-turn", response_model=TextTurnResponse)
async def text_turn(req: TextTurnRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    if req.user_id and req.user_id != uid:
        raise HTTPException(status_code=403, detail="Forbidden: user_id mismatch with authenticated token")
    user_id = uid
    await verify_book_ownership(user_id, req.book_id)
    user_received_at = datetime.now(timezone.utc)
    import re

    print(f"[TEXT-TURN] User message received (length: {len(req.message)})")
    print(f"[TEXT-TURN] History length: {len(req.history)} messages")

    captured = False
    reply = ""
    vignette_data = None
    image_url = None

    # --- 1. CONVERSATIONAL TURN — Always let the AI agent respond first ---
    messages = []
    for m in req.history:
        role = "user" if m["role"] == "user" else "model"
        messages.append({"role": role, "parts": [{"text": m["content"]}]})
    messages.append({"role": "user", "parts": [{"text": req.message}]})

    # Retry with exponential backoff for rate limit errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=INTERVIEW_SYSTEM_PROMPT,
                    temperature=0.85,
                    max_output_tokens=1500,
                    tools=[capture_vignette]
                )
            )

            # Check for function calls from the AI
            if response.function_calls:
                print(f"[TEXT-TURN] ✅ Function calls: {[c.name for c in response.function_calls]}")
                for call in response.function_calls:
                    if call.name == "capture_vignette":
                        captured = True
                # Get any text the model said alongside the function call
                reply = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            reply += part.text
                if not reply.strip():
                    reply = "What a beautiful memory. Let me capture that as a page in your book."
            else:
                reply_raw = response.text or ""
                reply = re.sub(r'<thought>.*?</thought>', '', reply_raw, flags=re.DOTALL).strip()
                print(f"[TEXT-TURN] AI reply generated (length: {len(reply)})")

            break  # Success — exit retry loop

        except Exception as e:
            err = str(e)
            is_transient = "429" in err or "RESOURCE_EXHAUSTED" in err or "503" in err or "UNAVAILABLE" in err
            print(f"[TEXT-TURN] ❌ Attempt {attempt+1}/{max_retries} Error: {err[:200]}")

            if is_transient and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s
                print(f"[TEXT-TURN] ⏳ Transient spike, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue

            if is_transient:
                reply = "That is such a vivid and heartfelt memory. What details of that place or moment still stand out most to you?"
                break
            raise

    # --- 2. MEMORY COMPLETENESS CHECK (backup if AI didn't call the tool) ---
    # If the AI didn't trigger capture, run our own completeness assessment
    if not captured:
        conversation_for_check = []
        for m in req.history:
            conversation_for_check.append({
                "role": m["role"] if m["role"] == "user" else "assistant",
                "content": m["content"]
            })
        conversation_for_check.append({"role": "user", "content": req.message})

        # Only check if we have enough conversation (at least 3 user messages)
        user_msg_count = sum(1 for m in conversation_for_check if m["role"] == "user")
        if user_msg_count >= 3:
            try:
                from vignette import assess_memory_completeness
                completeness = await assess_memory_completeness(conversation_for_check)
                if completeness.get("is_complete") and completeness.get("confidence", 0) >= 0.8:
                    captured = True
                    print(f"[TEXT-TURN] ✅ Backup completeness check triggered capture "
                          f"(confidence={completeness.get('confidence', 0):.2f})")
                    # Append a transition note to the reply
                    reply += "\n\nWhat a beautiful memory. Let me capture that as a page in your book."
            except Exception as e:
                print(f"[TEXT-TURN] Completeness check error: {e}")

    # --- 3. PAGE GENERATION (unified pipeline) ---
    if captured:
        # Build full conversation for vignette extraction
        conversation_for_vignette = []
        for m in req.history:
            conversation_for_vignette.append({
                "role": m["role"] if m["role"] == "user" else "assistant",
                "content": m["content"]
            })
        conversation_for_vignette.append({"role": "user", "content": req.message})

        print(f"[TEXT-TURN] Generating literary vignette from {len(conversation_for_vignette)} messages...")
        try:
            from vignette import extract_vignette
            vignette_data = await extract_vignette(conversation_for_vignette)
            print(f"[TEXT-TURN] Vignette narration generated (length: {len(vignette_data.get('narration', ''))})")
            image_url = await generate_illustration(vignette_data)
            print(f"[TEXT-TURN] Image generated: {('OK' if image_url else 'FAILED')}")

            await save_page(user_id, req.book_id, {
                "narration":         vignette_data.get("narration", ""),
                "image_url":         image_url,
                "scene_description": vignette_data.get("scene_description", ""),
                "era":               vignette_data.get("era", ""),
                "emotion":           vignette_data.get("emotion", ""),
                "key_people":        vignette_data.get("key_people", []),
                "chapter":           vignette_data.get("chapter", ""),
            })
            print(f"[TEXT-TURN] ✅ Page saved successfully")
        except Exception as e:
            print(f"[TEXT-TURN] ❌ Page generation error: {e}")
            if not vignette_data:
                captured = False

    # Clean up any internal markers from the reply
    reply = reply.replace("[CAPTURE_VIGNETTE]", "").strip()
    reply = re.sub(r'<call:capture_vignette.*?(?:/>|</call:capture_vignette>)', '', reply, flags=re.DOTALL).strip()
    reply = re.sub(r'<tool_code>.*?</tool_code>', '', reply, flags=re.DOTALL).strip()

    # Reset history after capture to start a fresh memory
    if captured and vignette_data:
        new_history = [{"role": "assistant", "content": reply + " Now, is there another memory you'd like to share?"}]
    else:
        new_history = req.history + [
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": reply}
        ]

    user_turn_id = req.client_turn_id or f"user_{int(user_received_at.timestamp() * 1000)}_{hashlib.sha256(req.message.strip().encode()).hexdigest()[:8]}"
    await save_turn(user_id, req.book_id, {
        "role": "user",
        "text": req.message,
        "session_id": "session_default",
        "mode": "text",
        "created_at": user_received_at
    }, user_turn_id)

    agent_received_at = datetime.now(timezone.utc)
    if agent_received_at <= user_received_at:
        agent_received_at = user_received_at + timedelta(milliseconds=10)
    agent_turn_id = f"agent_{int(agent_received_at.timestamp() * 1000)}_{hashlib.sha256(reply.strip().encode()).hexdigest()[:8]}"
    await save_turn(user_id, req.book_id, {
        "role": "agent",
        "text": reply,
        "session_id": "session_default",
        "mode": "text",
        "created_at": agent_received_at
    }, agent_turn_id)

    return TextTurnResponse(
        reply=reply,
        captured=captured,
        vignette=vignette_data,
        image_url=image_url,
        history=new_history,
    )


@app.get("/api/books/{book_id}/history")
async def get_conversation_history(book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    turns = await get_book_turns(uid, book_id)
    return {"history": turns}


class SaveTurnRequest(BaseModel):
    role: str
    text: str
    turn_id: str | None = None
    session_id: str | None = "default"
    mode: str | None = "text"


@app.post("/api/books/{book_id}/history")
async def append_conversation_turn(book_id: str, req: SaveTurnRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    role = "agent" if req.role in ("agent", "assistant") else "user"
    turn_id = await save_turn(uid, book_id, {
        "role": role,
        "text": req.text,
        "session_id": req.session_id or "default",
        "mode": req.mode or "text"
    }, req.turn_id)
    return {"status": "ok", "turn_id": turn_id}


@app.post("/api/books/{book_id}/history/reset")
async def reset_conversation_history(book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    deleted_count = await clear_book_turns(uid, book_id)
    return {"status": "ok", "deleted_count": deleted_count}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "echobook-backend"}


@app.get("/api/auth/config")
async def auth_config():
    api_key = os.environ.get("NEXT_PUBLIC_FIREBASE_API_KEY", "")
    project_id = os.environ.get("GCP_PROJECT") or os.environ.get("NEXT_PUBLIC_FIREBASE_PROJECT_ID", "")
    auth_domain = os.environ.get("NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN", f"{project_id}.firebaseapp.com" if project_id else "")
    return {
        "projectId": project_id,
        "apiKey": api_key,
        "authDomain": auth_domain,
    }


class DemoSessionRequest(BaseModel):
    user_id: str
    display_name: str | None = None


def is_demo_auth_allowed() -> bool:
    return os.environ.get("ALLOW_DEMO_AUTH", "false").strip().lower() in ("true", "1", "yes")

ALLOWED_DEMO_UIDS = {"storyteller_eleanor", "storyteller_arthur"}

@app.post("/api/auth/demo-session")
async def demo_session(req: DemoSessionRequest):
    if not is_demo_auth_allowed():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Demo authentication is disabled in this environment"
        )
    uid = req.user_id.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")
    if uid not in ALLOWED_DEMO_UIDS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Demo session is restricted to approved storyteller accounts"
        )
    custom_token = create_custom_token(uid, {"name": req.display_name or uid})
    return {
        "custom_token": custom_token,
        "user_id": uid,
    }


@app.get("/book/{book_id}")
async def get_book_by_id(book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    pages = await get_book_pages(uid, book_id)
    return {"pages": pages}


@app.get("/book/{user_id}/{book_id}")
async def get_book(user_id: str, book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    if user_id != uid:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot view another user's book")
    await verify_book_ownership(uid, book_id)
    pages = await get_book_pages(uid, book_id)
    return {"pages": pages}


@app.get("/api/books")
async def list_authenticated_user_books(current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    books = await get_user_books(uid)
    return {"books": books}


@app.get("/api/books/{user_id}")
async def list_user_books(user_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    if user_id != uid:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot view another user's books")
    books = await get_user_books(uid)
    return {"books": books}


class CreateBookRequest(BaseModel):
    book_id: str
    storyteller_name: str
    user_id: str | None = None


@app.post("/api/books")
async def api_create_book(req: CreateBookRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    if req.user_id and req.user_id != uid:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot create book for another user")
    await create_book(uid, req.book_id, req.storyteller_name)
    return {"status": "ok", "book_id": req.book_id}

@app.delete("/api/books/{book_id}")
async def api_delete_book(book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    await delete_book(uid, book_id)
    return {"status": "ok", "deleted": book_id}


@app.get("/session-greeting/{book_id}")
async def session_greeting_by_id(book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    await verify_book_ownership(uid, book_id)
    summary = await get_last_session_summary(uid, book_id)
    if summary:
        greeting = (
            f"Welcome back. Last time, {summary}. "
            "Would you like to continue from there, or is there another memory calling to you today?"
        )
    else:
        greeting = (
            "Hello, I am so glad you are here. I am EchoBook. "
            "I am here to listen to your stories and turn them into beautiful illustrated pages. "
            "To begin — could you tell me about a place from your childhood that made you feel truly happy?"
        )
    return {"greeting": greeting}


@app.get("/session-greeting/{user_id}/{book_id}")
async def session_greeting(user_id: str, book_id: str, current_user: dict = Depends(get_current_user)):
    uid = current_user["uid"]
    if user_id != uid:
        raise HTTPException(status_code=403, detail="Forbidden: Cannot access another user's greeting")
    await verify_book_ownership(uid, book_id)
    summary = await get_last_session_summary(uid, book_id)
    if summary:
        greeting = (
            f"Welcome back. Last time, {summary}. "
            "Would you like to continue from there, or is there another memory calling to you today?"
        )
    else:
        greeting = (
            "Hello, I am so glad you are here. I am EchoBook. "
            "I am here to listen to your stories and turn them into beautiful illustrated pages. "
            "To begin — could you tell me about a place from your childhood that made you feel truly happy?"
        )
    return {"greeting": greeting}


# ---------------------------------------------------------------------------
# Voice Interview WebSocket — now shares the same capture pipeline
# ---------------------------------------------------------------------------
@app.websocket("/ws/interview/{user_id}/{book_id}")
async def interview_websocket(websocket: WebSocket, user_id: str, book_id: str):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return
    try:
        decoded = verify_token_string(token)
    except Exception as e:
        await websocket.close(code=1008, reason=f"Invalid authentication token: {str(e)}")
        return
    token_uid = decoded.get("uid")
    if token_uid != user_id:
        await websocket.close(code=1008, reason="User ID does not match authenticated token")
        return
    book_meta = await get_book_meta(token_uid, book_id)
    if not book_meta:
        await websocket.close(code=1008, reason="Book not found or unauthorized")
        return
    await websocket.accept()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=INTERVIEW_SYSTEM_PROMPT)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=[capture_vignette]
    )

    try:
        async with client.aio.live.connect(
            model="gemini-2.5-flash-native-audio-latest", config=config
        ) as session:

            conversation_history = []

            async def receive_from_client():
                while True:
                    try:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            break
                        if "bytes" in message:
                            await session.send_realtime_input(
                                media=types.Blob(data=message["bytes"], mime_type="audio/pcm;rate=16000")
                            )
                        elif "text" in message:
                            data = json.loads(message["text"])
                            if data.get("type") == "greeting" or data.get("type") == "history":
                                history_context = ""
                                if "history" in data and isinstance(data["history"], list) and len(data["history"]) > 0:
                                    history_context = "Here is our conversation history so far. Please seamlessly reply to my latest message and continue the conversation:\n"
                                    for t in data["history"]:
                                        if t.get("role") and t.get("text"):
                                            conversation_history.append({"role": t.get("role"), "text": t.get("text")})
                                            history_context += f"{t['role'].capitalize()}: {t['text']}\n"

                                prompt = "Say exactly these words to introduce yourself: 'Hello.'"
                                if data.get("type") == "greeting" and data.get("text"):
                                    prompt = f"Say exactly these words to introduce yourself: '{data['text']}'"
                                elif history_context:
                                    prompt = history_context

                                await session.send(input=prompt, end_of_turn=True)
                            elif data.get("type") == "user_turn":
                                text = data.get("text", "").strip()
                                turn_id = data.get("turn_id")
                                session_id = data.get("session_id", "session_voice")
                                if text:
                                    if not turn_id:
                                        turn_id = f"user_{int(time.time() * 1000)}_{hashlib.sha256(text.encode()).hexdigest()[:8]}"
                                    await save_turn(token_uid, book_id, {
                                        "role": "user",
                                        "text": text,
                                        "session_id": session_id,
                                        "mode": "voice"
                                    }, turn_id)
                                    conversation_history.append({"role": "user", "text": text})
                                    await session.send(input=text, end_of_turn=True)
                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        print("Client receive error:", e)
                        break

            async def send_to_client():
                full_text_buffer = ""
                async for response in session.receive():
                    if response.data:
                        await websocket.send_json({
                            "type": "audio",
                            "data": base64.b64encode(response.data).decode()
                        })
                    if response.tool_call:
                        print("TOOL CALL RECEIVED:", response.tool_call)
                        for call in response.tool_call.function_calls:
                            if call.name == "capture_vignette":
                                # Use the unified pipeline
                                asyncio.create_task(
                                    handle_voice_capture(
                                        websocket, token_uid, book_id,
                                        conversation_history
                                    )
                                )
                                full_text_buffer = ""

                                # Respond to the tool call so model continues
                                await session.send(
                                    input=types.LiveClientContent(
                                        tool_response=types.LiveClientToolResponse(
                                            function_responses=[types.FunctionResponse(
                                                name="capture_vignette",
                                                response={"status": "Memory beautifully captured"},
                                                id=call.id
                                            )]
                                        )
                                    )
                                )

                    sc = getattr(response, 'server_content', None)
                    if sc:
                        out_trans = getattr(sc, 'output_transcription', None)
                        if out_trans and out_trans.text:
                            clean_t = out_trans.text.replace("*", "")
                            full_text_buffer += clean_t

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.text and not getattr(part, 'thought', False):
                                    clean_text = part.text.replace("*", "").strip()
                                    if clean_text:
                                        full_text_buffer += clean_text + " "

                        if getattr(sc, 'turn_complete', False):
                            import re
                            final_text = re.sub(r'<thought>.*?</thought>', '', full_text_buffer, flags=re.DOTALL).strip()
                            if final_text:
                                agent_turn_id = f"agent_{int(time.time() * 1000)}_{hashlib.sha256(final_text.encode()).hexdigest()[:8]}"
                                await save_turn(token_uid, book_id, {
                                    "role": "agent",
                                    "text": final_text,
                                    "session_id": "session_voice",
                                    "mode": "voice"
                                }, agent_turn_id)
                                conversation_history.append({"role": "assistant", "text": final_text})
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": final_text
                                })
                            await websocket.send_json({
                                "type": "turn_complete"
                            })
                            full_text_buffer = ""

            await asyncio.gather(receive_from_client(), send_to_client())

    except WebSocketDisconnect:
        pass
    except ConnectionClosedError as e:
        if e.code == 1011 or e.code == 1007:
            print("Gemini API connection closed:", e.code, e.reason)
        else:
            try:
                await websocket.send_json({"type": "error", "message": f"Gemini API disconnected: {e.code} {e.reason}"})
            except Exception:
                pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Unified voice capture handler — uses the same vignette pipeline as text
# ---------------------------------------------------------------------------
async def handle_voice_capture(
    websocket: WebSocket, user_id: str, book_id: str,
    conversation_history: list[dict]
):
    """
    Handle page generation for voice interviews using the shared pipeline.
    Converts conversation_history (role/text) to the pipeline format (role/content).
    """
    try:
        await websocket.send_json({"type": "page_generating", "message": "Capturing your memory…"})

        # Convert voice history to pipeline format
        conversation_for_pipeline = []
        for entry in conversation_history:
            role = entry.get("role", "user")
            text = entry.get("text", "")
            if role == "agent":
                role = "assistant"
            conversation_for_pipeline.append({"role": role, "content": text})

        # Use the same extract_vignette that text uses (literary narration)
        from vignette import extract_vignette
        vignette = await extract_vignette(conversation_for_pipeline)
        image_url = await generate_illustration(vignette)

        page_data = {
            "narration":         vignette.get("narration", ""),
            "image_url":         image_url,
            "scene_description": vignette.get("scene_description", ""),
            "era":               vignette.get("era", ""),
            "emotion":           vignette.get("emotion", ""),
            "key_people":        vignette.get("key_people", []),
            "chapter":           vignette.get("chapter", ""),
        }
        page_id = await save_page(user_id, book_id, page_data)
        await websocket.send_json({"type": "new_page", "page_id": page_id, **page_data})
        print(f"[VOICE-CAPTURE] ✅ Page saved: {page_id}")

        # Reset conversation history after capture
        conversation_history.clear()

    except Exception as e:
        print(f"[VOICE-CAPTURE] ❌ Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Page generation failed: {e}"})
        except Exception:
            pass


