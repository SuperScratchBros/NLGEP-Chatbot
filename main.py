import os
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
SESSION_MAX_AGE = 60 * 60 * 24 * 30

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


# ---------------------------------------------------------------- Supabase

class UsernameRequest(BaseModel):
    username: str = Field(min_length=2, max_length=24, pattern=r"^[A-Za-z0-9_ .-]+$")


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


# ---------------------------------------------------------------- API

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"off": sorted(OFF), "cloud_chat": supabase_configured()}


@app.post("/api/chat-session")
def create_chat_session(body: UsernameRequest, request: Request, response: Response):
    check_rate_limit(request)
    if not supabase_configured():
        raise HTTPException(503, "Supabase chat is not configured. Add SUPABASE_URL and SUPABASE_SECRET_KEY in Render.")
    username = " ".join(body.username.strip().split())
    if not username:
        raise HTTPException(400, "Username cannot be empty.")
    # A random session ID is the browser's login token. No password is stored.
    session_id = os.urandom(24).hex()
    user = None
    try:
        existing = supabase_rows("chat_users", {"select": "user_id,username", "username": f"ilike.{username}", "limit": "1"})
        if existing:
            raise HTTPException(409, "That username is already taken. Choose another one.")
        user = supabase_rows("chat_users", {"select": "user_id,username"}, method="POST",
                             payload={"username": username}, prefer="return=representation")[0]
        expires_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + SESSION_MAX_AGE))
        supabase_rows("chat_sessions", method="POST",
                      payload={"session_id": session_id, "user_id": user["user_id"], "username": user["username"],
                               "expires_at": expires_at, "last_seen": utc_now()},
                      prefer="return=minimal")
    except HTTPException as e:
        delete_user_quietly(user)
        if e.status_code == 502 and "duplicate key" in str(e.detail).lower():
            raise HTTPException(409, "That username is already taken. Choose another one.")
        raise
    except Exception as e:
        delete_user_quietly(user)
        raise HTTPException(502, f"Could not create chat session: {scrub(e)}")
    response.set_cookie(SESSION_COOKIE, session_id, max_age=SESSION_MAX_AGE, httponly=True,
                        samesite="lax", secure=request.url.scheme == "https")
    return {"username": user["username"]}


@app.delete("/api/chat-session")
def delete_chat_session(request: Request, response: Response):
    session_id = request.cookies.get(SESSION_COOKIE, "")
    response.delete_cookie(SESSION_COOKIE)
    if session_id and supabase_configured():
        try:
            supabase_rows("chat_sessions", {"session_id": f"eq.{session_id}"}, method="DELETE")
        except HTTPException:
            pass
    return {"ok": True}


@app.get("/api/chat-me")
def chat_me(request: Request):
    return {"user": session_user(request)}


@app.get("/api/chat-users")
def chat_users(request: Request):
    if not session_user(request):
        raise HTTPException(401, "Choose a username first.")
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
        raise HTTPException(401, "Choose a username first.")
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
        raise HTTPException(401, "Choose a username first.")
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
