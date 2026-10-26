import os
from fastapi import Header, HTTPException, status
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from firestore_client import get_book_meta

def init_firebase():
    if not firebase_admin._apps:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.environ.get("GCP_PROJECT")
        if cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        else:
            firebase_admin.initialize_app(options={"projectId": project_id})

init_firebase()

async def get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing"
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme. Expected 'Bearer <token>'"
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token missing"
        )
    try:
        decoded_token = firebase_auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}"
        )

def verify_token_string(token: str) -> dict:
    if not token:
        raise ValueError("Token missing")
    return firebase_auth.verify_id_token(token)

def create_custom_token(uid: str, claims: dict | None = None) -> str:
    token_bytes = firebase_auth.create_custom_token(uid, claims)
    return token_bytes.decode("utf-8") if isinstance(token_bytes, bytes) else token_bytes

async def verify_book_ownership(user_id: str, book_id: str) -> dict:
    book = await get_book_meta(user_id, book_id)
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Book not found or does not belong to the authenticated user"
        )
    return book
