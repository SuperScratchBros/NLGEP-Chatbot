import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
BASE_DIR = Path(__file__).parent


def env_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1].strip()
    return value


# Feature switches: DISABLED_FEATURES=cloudchat (comma list)
OFF = {f.strip().lower() for f in os.getenv("DISABLED_FEATURES", "").split(",") if f.strip()}
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SECRET_KEY = env_secret("SUPABASE_SECRET_KEY") or env_secret("SUPABASE_SERVICE_ROLE_KEY")

RATE_LIMIT_IP, RATE_LIMIT_GLOBAL = 20, 120  # write requests per minute
SESSION_COOKIE = "infinet_session"
# Sessions are browser-session cookies (no max_age), so closing the browser logs you out.
# This is only the server-side upper bound for a single session.
SESSION_MAX_AGE = 60 * 60 * 12

app = FastAPI()
ip_hits: dict[str, deque] = defaultdict(deque)
global_hits: deque = deque()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def scrub(text) -> str:
    """Hide the server key if it ever ends up inside an error message."""
    text = str(text)
    if len(SUPABASE_SECRET_KEY) > 4:
        text = text.replace(SUPABASE_SECRET_KEY, "***")
    return text


def client_ip(request: Request) -> str:
    if not TRUST_PROXY_HEADERS:
        return request.client.host if request.client else "unknown"
    for header in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip"):
        value = request.headers.get(header, "").strip()
        if not value:
            continue
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if parts:
            return parts[0]
    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> None:
    ip = client_ip(request)
    now = time.time()
    while global_hits and now - global_hits[0] > 60:
        global_hits.popleft()
    for k in [k for k, q in ip_hits.items() if not q or now - q[-1] > 60]:
        del ip_hits[k]
    q = ip_hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_IP or len(global_hits) >= RATE_LIMIT_GLOBAL:
        raise HTTPException(429, "Too many requests. Try again in a minute.")
    q.append(now)
    global_hits.append(now)


# ---------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored) -> bool:
    try:
        salt_hex, digest_hex = (stored or "").split("$")
        salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(digest_hex)
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=len(expected))
    return hmac.compare_digest(digest, expected)


# A throwaway hash so login takes about the same time whether or not the username exists.
DUMMY_HASH = hash_password("not-a-real-password")


# ---------------------------------------------------------------- Supabase

USERNAME_PATTERN = r"^[A-Za-z0-9_ .-]+$"


class CredentialsRequest(BaseModel):
    username: str = Field(min_length=2, max_length=24, pattern=USERNAME_PATTERN)
    password: str = Field(min_length=6, max_length=128)


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=2000)


def supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SECRET_KEY and "cloudchat" not in OFF)


def supabase_headers():
    if not SUPABASE_SECRET_KEY:
        raise RuntimeError("Supabase server key is not configured.")
    headers = {"apikey": SUPABASE_SECRET_KEY, "Content-Type": "application/json"}
    # Legacy service_role keys are JWTs; new sb_secret_* keys must be sent as apikey only.
    if SUPABASE_SECRET_KEY.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {SUPABASE_SECRET_KEY}"
    return headers


def supabase_rows(table: str, params=None, method="GET", payload=None, prefer=None):
    if not supabase_configured():
        raise HTTPException(503, "Supabase chat is not configured. Add SUPABASE_URL and SUPABASE_SECRET_KEY in Render.")
    headers = supabase_headers()
    if prefer:
        headers["Prefer"] = prefer
    r = httpx.request(method, f"{SUPABASE_URL}/rest/v1/{table}", headers=headers,
                      params=params or {}, json=payload, timeout=10)
    if r.status_code not in (200, 201, 204):
        raise HTTPException(502, f"Supabase request failed ({r.status_code}): {scrub(r.text[:300])}")
    return r.json() if r.text else []


def session_user(request: Request):
    session_id = request.cookies.get(SESSION_COOKIE, "")
    if not session_id or not supabase_configured():
        return None
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/chat_sessions",
        headers=supabase_headers(),
        params={"select": "session_id,user_id,username,last_seen", "session_id": f"eq.{session_id}",
                "expires_at": f"gt.{utc_now()}", "limit": "1"},
        timeout=10,
    )
    if r.status_code != 200:
        raise HTTPException(502, f"Session lookup failed: {scrub(r.text[:300])}")
    rows = r.json()
    if not rows:
        return None
    # Polling hits this every few seconds; only write presence periodically.
    should_touch = True
    try:
        last_seen = rows[0].get("last_seen", "")
        if last_seen:
            last_seen_dt = datetime.fromisoformat(last_seen.replace("Z", "+00:00"))
            should_touch = (datetime.now(timezone.utc) - last_seen_dt).total_seconds() >= 15
    except Exception:
        should_touch = True
    if should_touch:
        try:
            supabase_rows("chat_sessions", {"session_id": f"eq.{session_id}"}, method="PATCH",
                          payload={"last_seen": utc_now()}, prefer="return=minimal")
        except HTTPException:
            pass
    return rows[0]


def delete_user_quietly(user):
    if user is None:
        return
    try:
        supabase_rows("chat_users", {"user_id": f"eq.{user['user_id']}"}, method="DELETE")
    except Exception:
        pass


def clean_username(raw: str) -> str:
    username = " ".join(raw.strip().split())
    if len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters.")
    return username


def find_user(username: str):
    """Case-insensitive exact lookup (ilike can over-match '_', so compare again in Python)."""
    rows = supabase_rows("chat_users", {"select": "user_id,username,password_hash,is_guest",
                                        "username": f"ilike.{username}", "limit": "50"})
    for row in rows:
        if row["username"].lower() == username.lower():
            return row
    return None


def start_session(user, request: Request, response: Response) -> None:
    """Create a server-side session and set a non-persistent cookie (cleared when the browser closes)."""
    session_id = os.urandom(24).hex()
    expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + SESSION_MAX_AGE))
    supabase_rows("chat_sessions", method="POST",
                  payload={"session_id": session_id, "user_id": user["user_id"], "username": user["username"],
                           "expires_at": expires_at, "last_seen": utc_now()},
                  prefer="return=minimal")
    response.set_cookie(SESSION_COOKIE, session_id, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")


# ---------------------------------------------------------------- API

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"off": sorted(OFF), "cloud_chat": supabase_configured()}


@app.post("/api/chat-register")
def register(body: CredentialsRequest, request: Request, response: Response):
    check_rate_limit(request)
    username = clean_username(body.username)
    if username.lower().startswith("guest"):
        raise HTTPException(400, "Usernames starting with \"Guest\" are reserved.")
    user = None
    try:
        if find_user(username):
            raise HTTPException(409, "That username is already taken. Choose another one.")
        user = supabase_rows("chat_users", {"select": "user_id,username"}, method="POST",
                             payload={"username": username, "password_hash": hash_password(body.password),
                                      "is_guest": False},
                             prefer="return=representation")[0]
        start_session(user, request, response)
    except HTTPException as e:
        delete_user_quietly(user)
        if e.status_code == 502 and "duplicate key" in str(e.detail).lower():
            raise HTTPException(409, "That username is already taken. Choose another one.")
        raise
    except Exception as e:
        delete_user_quietly(user)
        raise HTTPException(502, f"Could not create account: {scrub(e)}")
    return {"username": user["username"], "guest": False}


@app.post("/api/chat-login")
def login(body: CredentialsRequest, request: Request, response: Response):
    check_rate_limit(request)
    username = clean_username(body.username)
    user = find_user(username)
    stored = user.get("password_hash") if user else None
    ok = verify_password(body.password, stored or DUMMY_HASH)
    if not user or not stored or not ok:
        raise HTTPException(401, "Incorrect username or password.")
    try:
        start_session(user, request, response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Could not sign in: {scrub(e)}")
    return {"username": user["username"], "guest": False}


@app.post("/api/chat-guest")
def guest_login(request: Request, response: Response):
    check_rate_limit(request)
    # Housekeeping: guests are throwaway, so drop any that are older than a session could be.
    try:
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - SESSION_MAX_AGE))
        supabase_rows("chat_users", {"is_guest": "eq.true", "created_at": f"lt.{cutoff}"}, method="DELETE")
    except Exception:
        pass
    user = None
    for _ in range(8):
        name = f"Guest-{secrets.randbelow(10000):04d}"
        try:
            user = supabase_rows("chat_users", {"select": "user_id,username"}, method="POST",
                                 payload={"username": name, "password_hash": None, "is_guest": True},
                                 prefer="return=representation")[0]
            break
        except HTTPException as e:
            if e.status_code == 502 and "duplicate key" in str(e.detail).lower():
                continue
            raise
    if user is None:
        raise HTTPException(503, "Could not find a free guest name. Try again.")
    try:
        start_session(user, request, response)
    except HTTPException:
        delete_user_quietly(user)
        raise
    except Exception as e:
        delete_user_quietly(user)
        raise HTTPException(502, f"Could not start guest session: {scrub(e)}")
    return {"username": user["username"], "guest": True}


@app.delete("/api/chat-session")
def delete_chat_session(request: Request, response: Response):
    session_id = request.cookies.get(SESSION_COOKIE, "")
    response.delete_cookie(SESSION_COOKIE)
    if session_id and supabase_configured():
        try:
            rows = supabase_rows("chat_sessions", {"select": "user_id", "session_id": f"eq.{session_id}", "limit": "1"})
            supabase_rows("chat_sessions", {"session_id": f"eq.{session_id}"}, method="DELETE")
            if rows:
                # Guests vanish when they leave (their messages go with them via ON DELETE CASCADE).
                users = supabase_rows("chat_users", {"select": "user_id,is_guest", "user_id": f"eq.{rows[0]['user_id']}"})
                if users and users[0].get("is_guest"):
                    delete_user_quietly(users[0])
        except HTTPException:
            pass
    return {"ok": True}


@app.get("/api/chat-me")
def chat_me(request: Request):
    return {"user": session_user(request)}


@app.get("/api/chat-users")
def chat_users(request: Request):
    if not session_user(request):
        raise HTTPException(401, "Please sign in first.")
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
    rows = supabase_rows("chat_sessions", {
        "select": "user_id,username,last_seen",
        "last_seen": f"gt.{cutoff}",
        "expires_at": f"gt.{utc_now()}",
        "order": "username.asc",
        "limit": "100",
    })
    seen, users = set(), []
    for row in rows:
        if row["user_id"] in seen:
            continue
        seen.add(row["user_id"])
        users.append({"user_id": row["user_id"], "username": row["username"]})
    return {"users": users, "count": len(users)}


@app.get("/api/chat-messages")
def chat_messages(request: Request, after: str = ""):
    me = session_user(request)
    if not me:
        raise HTTPException(401, "Please sign in first.")
    params = {
        "select": "message_id,user_id,username,content,created_at",
        "order": "message_id.asc",
        "limit": "100",
    }
    if after and after.isdigit():
        params["message_id"] = f"gt.{after}"
    return {"messages": supabase_rows("chat_messages", params), "me": me["username"]}


@app.post("/api/chat-messages")
def send_chat_message(body: ChatMessageRequest, request: Request):
    check_rate_limit(request)
    me = session_user(request)
    if not me:
        raise HTTPException(401, "Please sign in first.")
    content = body.content.strip()
    if not content:
        raise HTTPException(400, "Message cannot be empty.")
    rows = supabase_rows("chat_messages", method="POST",
                         payload={"user_id": me["user_id"], "username": me["username"],
                                  "content": content, "created_at": utc_now()},
                         prefer="return=representation")
    return {"message": rows[0]}


# ---------------------------------------------------------------- pages

@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    """Make browsers re-check pages/CSS on every load so UI updates show up right away."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/chat") or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/chat")
def chatting():
    return FileResponse(BASE_DIR / "static" / "chatting.html")
