import os
from datetime import datetime, timezone
from google.cloud import firestore

_db = None

def get_db():
    global _db
    if _db is None:
        _db = firestore.AsyncClient(project=os.environ.get("GCP_PROJECT", ""))
    return _db


async def save_page(user_id: str, book_id: str, page_data: dict) -> str:
    db = get_db()
    page_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("pages")
        .document()
    )
    data = {
        **page_data,
        "created_at": datetime.now(timezone.utc),
        "page_number": await _get_next_page_number(user_id, book_id),
    }
    await page_ref.set(data)
    await _update_session_summary(user_id, book_id, page_data)
    return page_ref.id


async def get_book_pages(user_id: str, book_id: str) -> list:
    db = get_db()
    pages_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("pages")
        .order_by("page_number")
    )
    docs = pages_ref.stream()
    pages = []
    async for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
        pages.append(d)
    return pages


async def get_last_session_summary(user_id: str, book_id: str) -> str | None:
    db = get_db()
    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    doc = await book_ref.get()
    if doc.exists:
        data = doc.to_dict()
        return data.get("last_session_summary")
    return None


async def get_user_books(user_id: str) -> list:
    db = get_db()
    books_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .order_by("last_updated", direction=firestore.Query.DESCENDING)
    )
    docs = books_ref.stream()
    books = []
    async for dict_doc in docs:
        d = dict_doc.to_dict()
        d["id"] = dict_doc.id
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
        if "last_updated" in d and hasattr(d["last_updated"], "isoformat"):
            d["last_updated"] = d["last_updated"].isoformat()
        books.append(d)
    return books


async def create_book(user_id: str, book_id: str, storyteller_name: str) -> str:
    db = get_db()
    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    now = datetime.now(timezone.utc)
    await book_ref.set(
        {
            "storyteller_name": storyteller_name,
            "created_at": now,
            "last_updated": now,
            "last_session_summary": "Just started",
        },
        merge=True
    )
    return book_id


async def _get_next_page_number(user_id: str, book_id: str) -> int:
    db = get_db()
    pages_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("pages")
    )
    docs = pages_ref.stream()
    count = 0
    async for _ in docs:
        count += 1
    return count + 1


async def _update_session_summary(user_id: str, book_id: str, page_data: dict):
    db = get_db()
    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    chapter = page_data.get("chapter", "a memory")
    emotion = page_data.get("emotion", "")
    summary = f"you shared a memory about {chapter}"
    if emotion:
        summary += f" filled with {emotion}"
    await book_ref.set(
        {
            "last_session_summary": summary,
            "last_updated": datetime.now(timezone.utc),
        },
        merge=True,
    )
async def get_book_meta(user_id: str, book_id: str) -> dict | None:
    db = get_db()
    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    doc = await book_ref.get()
    if doc.exists:
        data = doc.to_dict()
        data["id"] = doc.id
        return data
    return None

async def save_turn(user_id: str, book_id: str, turn_data: dict, turn_id: str | None = None) -> str:
    db = get_db()
    turns_coll = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("turns")
    )
    if turn_id:
        doc_ref = turns_coll.document(turn_id)
    else:
        doc_ref = turns_coll.document()

    now = datetime.now(timezone.utc)
    created_at = turn_data.get("created_at")
    if not created_at:
        existing = await doc_ref.get()
        if existing.exists and "created_at" in (existing.to_dict() or {}):
            created_at = existing.to_dict()["created_at"]
        else:
            created_at = now

    data = {
        "role": turn_data.get("role", "user"),
        "text": turn_data.get("text", ""),
        "session_id": turn_data.get("session_id", "default"),
        "mode": turn_data.get("mode", "text"),
        "created_at": created_at,
    }
    await doc_ref.set(data, merge=True)

    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    await book_ref.set({"last_updated": now}, merge=True)
    return doc_ref.id


async def get_book_turns(user_id: str, book_id: str) -> list[dict]:
    db = get_db()
    turns_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("turns")
        .order_by("created_at")
    )
    docs = turns_ref.stream()
    turns = []
    async for doc in docs:
        d = doc.to_dict()
        d["id"] = doc.id
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
        turns.append(d)
    return turns


async def clear_book_turns(user_id: str, book_id: str) -> int:
    db = get_db()
    turns_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
        .collection("turns")
    )
    docs = turns_ref.stream()
    count = 0
    async for doc in docs:
        await doc.reference.delete()
        count += 1
    return count

async def delete_book(user_id: str, book_id: str) -> bool:
    db = get_db()
    book_ref = (
        db.collection("users")
        .document(user_id)
        .collection("books")
        .document(book_id)
    )
    pages = book_ref.collection("pages").stream()
    async for p in pages:
        await p.reference.delete()
    turns = book_ref.collection("turns").stream()
    async for t in turns:
        await t.reference.delete()
    await book_ref.delete()
    return True
