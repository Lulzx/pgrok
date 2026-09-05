import asyncio
import hashlib
import html
import json
import os
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web


LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 12341
PUBLIC_ORIGIN = "https://inference.lulzx.space"
PUBLIC_API_ROOT = f"{PUBLIC_ORIGIN}/api/v1"
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "").rstrip("/")
UPSTREAM_API_TOKEN = os.environ.get("UPSTREAM_API_TOKEN", "")
UPSTREAM_MODEL = os.environ.get("UPSTREAM_MODEL", "")
UPSTREAM_LABEL = os.environ.get("UPSTREAM_LABEL", "")
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", "/var/lib/inference-gateway/state.sqlite3"))
STATIC_PATH = Path(os.environ.get("STATIC_PATH", "/opt/inference-gateway/static"))

TOKEN_LIFETIME_DAYS = 30
TOKEN_ISSUES_PER_HOUR = 3
REQUESTS_PER_MINUTE = 10
MAX_CONCURRENT_REQUESTS = 2
MAX_REQUEST_BYTES = 20 * 1024 * 1024

API_PATHS = {
    "/api/v1/chat/completions": "/chat/completions",
    "/api/v1/completions": "/completions",
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Authorization, Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Max-Age": "86400",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def open_database() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS access_tokens (
            token_hash TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            issued_ip TEXT NOT NULL,
            last_used_at TEXT,
            request_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS access_tokens_issued_ip_created_at
            ON access_tokens (issued_ip, created_at);
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            started_at TEXT NOT NULL,
            requests INTEGER NOT NULL DEFAULT 0,
            successful_requests INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            issued_tokens INTEGER NOT NULL DEFAULT 0,
            last_request_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO metrics (id, started_at) VALUES (1, ?)",
        (utc_now().isoformat(),),
    )
    legacy_hash = os.environ.get("LEGACY_TOKEN_HASH", "").strip()
    if legacy_hash:
        now = utc_now()
        connection.execute(
            """
            INSERT OR IGNORE INTO access_tokens
                (token_hash, created_at, expires_at, issued_ip)
            VALUES (?, ?, ?, ?)
            """,
            (legacy_hash, now.isoformat(), (now + timedelta(days=3650)).isoformat(), "legacy"),
        )
    connection.commit()
    return connection


database = open_database()
database_lock = asyncio.Lock()
capacity_lock = asyncio.Lock()
request_times: deque[float] = deque()
token_request_times: dict[str, deque[float]] = defaultdict(deque)
concurrency = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


def client_ip(request: web.Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()
    return request.remote or "unknown"


async def issue_token(request: web.Request) -> web.Response:
    ip = client_ip(request)
    now = utc_now()
    cutoff = (now - timedelta(hours=1)).isoformat()
    async with database_lock:
        recent = database.execute(
            "SELECT COUNT(*) FROM access_tokens WHERE issued_ip = ? AND created_at >= ?",
            (ip, cutoff),
        ).fetchone()[0]
        if recent >= TOKEN_ISSUES_PER_HOUR:
            return web.json_response(
                {"error": {"message": "Token creation limit reached. Try again later.", "type": "rate_limit_error"}},
                status=429,
                headers={**CORS_HEADERS, "Retry-After": "3600", "Cache-Control": "no-store"},
            )
        token = f"inf_{secrets.token_urlsafe(32)}"
        expires = now + timedelta(days=TOKEN_LIFETIME_DAYS)
        database.execute(
            "INSERT INTO access_tokens (token_hash, created_at, expires_at, issued_ip) VALUES (?, ?, ?, ?)",
            (token_hash(token), now.isoformat(), expires.isoformat(), ip),
        )
        database.execute("UPDATE metrics SET issued_tokens = issued_tokens + 1 WHERE id = 1")
        database.commit()
    return web.json_response(
        {
            "token": token,
            "token_type": "bearer",
            "expires_at": expires.isoformat(),
        },
        status=201,
        headers={**CORS_HEADERS, "Cache-Control": "no-store"},
    )


async def authenticate(request: web.Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return None
    digest = token_hash(authorization[7:].strip())
    now = utc_now().isoformat()
    async with database_lock:
        row = database.execute(
            "SELECT token_hash FROM access_tokens WHERE token_hash = ? AND expires_at > ?",
            (digest, now),
        ).fetchone()
    return digest if row else None


async def at_capacity(digest: str) -> bool:
    now = time.monotonic()
    async with capacity_lock:
        while request_times and request_times[0] <= now - 60:
            request_times.popleft()
        per_token = token_request_times[digest]
        while per_token and per_token[0] <= now - 60:
            per_token.popleft()
        if len(request_times) >= REQUESTS_PER_MINUTE or len(per_token) >= REQUESTS_PER_MINUTE:
            return True
        request_times.append(now)
        per_token.append(now)
        return False


async def record_request(digest: str, status: int, usage: dict | None = None) -> None:
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    now = utc_now().isoformat()
    async with database_lock:
        database.execute(
            """
            UPDATE metrics SET
                requests = requests + 1,
                successful_requests = successful_requests + ?,
                input_tokens = input_tokens + ?,
                output_tokens = output_tokens + ?,
                total_tokens = total_tokens + ?,
                last_request_at = ?
            WHERE id = 1
            """,
            (1 if 200 <= status < 400 else 0, input_tokens, output_tokens, total_tokens, now),
        )
        database.execute(
            "UPDATE access_tokens SET last_used_at = ?, request_count = request_count + 1 WHERE token_hash = ?",
            (now, digest),
        )
        database.commit()


def extract_usage(content_type: str, body: bytes) -> dict:
    try:
        text = body.decode("utf-8")
        if "text/event-stream" in content_type:
            found = {}
            for line in text.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        candidate = json.loads(data).get("usage")
                        if candidate:
                            found = candidate
            return found
        return json.loads(text).get("usage") or {}
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {}


def sanitize(body: bytes) -> bytes:
    upstream_host = urlsplit(UPSTREAM_URL).netloc
    replacements = {
        UPSTREAM_MODEL.encode(): b"default",
        UPSTREAM_URL.encode(): PUBLIC_API_ROOT.encode(),
        upstream_host.encode(): urlsplit(PUBLIC_ORIGIN).netloc.encode(),
        UPSTREAM_LABEL.encode(): b"Inference service",
        UPSTREAM_LABEL.lower().encode(): b"inference service",
    }
    for source, target in replacements.items():
        if source:
            body = body.replace(source, target)
    return body


async def options(_: web.Request) -> web.Response:
    return web.Response(status=204, headers=CORS_HEADERS)


async def models(request: web.Request) -> web.Response:
    digest = await authenticate(request)
    if digest is None:
        return unauthorized()
    await record_request(digest, 200)
    return web.json_response(
        {"object": "list", "data": [{"id": "default", "object": "model", "owned_by": "inference"}]},
        headers={**CORS_HEADERS, "Cache-Control": "no-store"},
    )


def unauthorized() -> web.Response:
    return web.json_response(
        {"error": {"message": "Invalid or expired API token", "type": "authentication_error"}},
        status=401,
        headers={**CORS_HEADERS, "WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
    )


async def proxy(request: web.Request) -> web.StreamResponse:
    digest = await authenticate(request)
    if digest is None:
        return unauthorized()
    if await at_capacity(digest):
        return web.json_response(
            {"error": {"message": "Request limit reached. Try again shortly.", "type": "rate_limit_error"}},
            status=429,
            headers={**CORS_HEADERS, "Retry-After": "60", "Cache-Control": "no-store"},
        )
    if not (UPSTREAM_URL and UPSTREAM_API_TOKEN and UPSTREAM_MODEL):
        await record_request(digest, 503)
        return web.json_response(
            {"error": {"message": "Inference service is temporarily unavailable", "type": "service_error"}},
            status=503,
            headers={**CORS_HEADERS, "Cache-Control": "no-store"},
        )
    try:
        payload = json.loads(await request.read())
    except (json.JSONDecodeError, UnicodeDecodeError):
        await record_request(digest, 400)
        return web.json_response(
            {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            status=400,
            headers={**CORS_HEADERS, "Cache-Control": "no-store"},
        )
    payload["model"] = UPSTREAM_MODEL
    if payload.get("stream") is True:
        payload.setdefault("stream_options", {})["include_usage"] = True
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    upstream_path = API_PATHS[request.path]
    session: ClientSession = request.app["session"]
    request_headers = {
        "Authorization": f"Bearer {UPSTREAM_API_TOKEN}",
        "Content-Type": "application/json",
        "Accept": request.headers.get("Accept", "application/json"),
        "User-Agent": "Inference-API/1.0",
    }
    async with concurrency:
        try:
            async with session.post(
                f"{UPSTREAM_URL}{upstream_path}",
                data=body,
                headers=request_headers,
            ) as upstream:
                content_type = upstream.headers.get("Content-Type", "application/json")
                safe_headers = {
                    **CORS_HEADERS,
                    "Content-Type": content_type,
                    "Cache-Control": "no-store",
                }
                if "text/event-stream" not in content_type:
                    upstream_body = await upstream.read()
                    usage = extract_usage(content_type, upstream_body)
                    await record_request(digest, upstream.status, usage)
                    public_status = upstream.status
                    if upstream.status in {401, 403}:
                        public_status = 503
                        upstream_body = b'{"error":{"message":"Inference service is temporarily unavailable","type":"service_error"}}'
                    return web.Response(
                        body=sanitize(upstream_body),
                        status=public_status,
                        headers=safe_headers,
                    )

                response = web.StreamResponse(status=upstream.status, headers=safe_headers)
                await response.prepare(request)
                observed = bytearray()
                pending = bytearray()
                async for chunk in upstream.content.iter_any():
                    if len(observed) < 2 * 1024 * 1024:
                        observed.extend(chunk[: (2 * 1024 * 1024) - len(observed)])
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        await response.write(sanitize(line + b"\n"))
                if pending:
                    await response.write(sanitize(bytes(pending)))
                await response.write_eof()
                await record_request(digest, upstream.status, extract_usage(content_type, bytes(observed)))
                return response
        except ClientError:
            await record_request(digest, 502)
            return web.json_response(
                {"error": {"message": "Inference service is temporarily unavailable", "type": "service_error"}},
                status=502,
                headers={**CORS_HEADERS, "Cache-Control": "no-store"},
            )


async def metrics_snapshot() -> dict:
    async with database_lock:
        return dict(database.execute("SELECT * FROM metrics WHERE id = 1").fetchone())


async def index(_: web.Request) -> web.Response:
    metrics = await metrics_snapshot()
    started = html.escape(metrics["started_at"])
    last_request = html.escape(metrics["last_request_at"] or "none yet")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="OpenAI-compatible text and vision inference API.">
<title>Inference API</title>
<style>
@font-face{{font-family:'IBM Plex Sans';src:url('/assets/IBMPlexSans-Regular.woff2') format('woff2');font-weight:400;font-style:normal;font-display:swap}}
@font-face{{font-family:'IBM Plex Sans';src:url('/assets/IBMPlexSans-SemiBold.woff2') format('woff2');font-weight:600;font-style:normal;font-display:swap}}
@font-face{{font-family:'IBM Plex Mono';src:url('/assets/IBMPlexMono-Regular.woff2') format('woff2');font-weight:400;font-style:normal;font-display:swap}}
html{{font-family:'IBM Plex Sans',sans-serif;color:#111;background:#fff;line-height:1.5}}
body{{max-width:850px;margin:32px auto;padding:0 18px}}
h1,h2{{font-weight:600;line-height:1.2}} h1{{font-size:2rem}} h2{{margin-top:2rem;font-size:1.25rem}}
code,pre{{font-family:'IBM Plex Mono',monospace}} pre{{padding:14px;border:1px solid #aaa;background:#f7f7f7;overflow:auto;white-space:pre}}
a{{color:#0645ad}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #aaa;padding:7px 9px;text-align:left;vertical-align:top}}
th{{font-weight:600;background:#f2f2f2}} dt{{font-weight:600;margin-top:1rem}} dd{{margin-left:0}} hr{{border:0;border-top:1px solid #888;margin:2rem 0}}
</style>
</head>
<body>
<h1>Inference API</h1>
<p>A small OpenAI-compatible API for text, chat, streaming, and image understanding.</p>
<p><strong>Base URL:</strong> <code>{PUBLIC_API_ROOT}</code></p>
<p>There is one default model. Omit <code>model</code>, or use <code>default</code> when a client requires a value.</p>

<h2>1. Create an API token</h2>
<p>No account is required. A token is displayed once and expires after {TOKEN_LIFETIME_DAYS} days.</p>
<pre>curl -X POST {PUBLIC_ORIGIN}/api/token</pre>
<p>Save the returned <code>token</code> value and send it with each request:</p>
<pre>Authorization: Bearer YOUR_API_TOKEN</pre>

<h2>2. Chat completion</h2>
<pre>curl {PUBLIC_API_ROOT}/chat/completions \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{{
    "messages": [
      {{"role": "system", "content": "You are a helpful assistant."}},
      {{"role": "user", "content": "Explain entropy in two sentences."}}
    ],
    "max_tokens": 256
  }}'</pre>

<h2>Python</h2>
<pre>from openai import OpenAI

client = OpenAI(
    base_url="{PUBLIC_API_ROOT}",
    api_key="YOUR_API_TOKEN",
)

response = client.chat.completions.create(
    model="default",
    messages=[{{"role": "user", "content": "Hello"}}],
)

print(response.choices[0].message.content)</pre>

<h2>Streaming</h2>
<p>Set <code>stream</code> to <code>true</code>. The response uses server-sent events.</p>
<pre>{{
  "messages": [{{"role": "user", "content": "Write a short poem."}}],
  "stream": true
}}</pre>

<h2>Image input</h2>
<p>Use an image URL in a chat message alongside a text instruction.</p>
<pre>{{
  "messages": [
    {{
      "role": "user",
      "content": [
        {{"type": "text", "text": "Describe this image."}},
        {{"type": "image_url", "image_url": {{"url": "https://example.com/image.jpg"}}}}
      ]
    }}
  ]
}}</pre>

<h2>Endpoints</h2>
<table>
<thead><tr><th>Method</th><th>Path</th><th>Description</th></tr></thead>
<tbody>
<tr><td>POST</td><td><code>/api/token</code></td><td>Create an API token without an account.</td></tr>
<tr><td>GET</td><td><code>/api/v1/models</code></td><td>Return the neutral model identifier <code>default</code>.</td></tr>
<tr><td>POST</td><td><code>/api/v1/chat/completions</code></td><td>Chat, vision, and streaming completions.</td></tr>
<tr><td>POST</td><td><code>/api/v1/completions</code></td><td>Legacy text completions.</td></tr>
</tbody>
</table>

<h2>Limits and errors</h2>
<dl>
<dt><code>400</code></dt><dd>The request body is invalid.</dd>
<dt><code>401</code></dt><dd>The API token is missing, invalid, or expired.</dd>
<dt><code>429</code></dt><dd>Capacity or token-creation limit reached. Retry after the indicated interval.</dd>
<dt><code>5xx</code></dt><dd>The inference service is temporarily unavailable.</dd>
</dl>
<p>At most {REQUESTS_PER_MINUTE} inference requests are admitted per rolling minute. Token creation is limited to {TOKEN_ISSUES_PER_HOUR} per address per hour.</p>

<h2>Usage statistics</h2>
<table>
<tbody>
<tr><th>API tokens issued</th><td>{metrics['issued_tokens']:,}</td></tr>
<tr><th>Inference requests</th><td>{metrics['requests']:,}</td></tr>
<tr><th>Successful requests</th><td>{metrics['successful_requests']:,}</td></tr>
<tr><th>Input tokens processed</th><td>{metrics['input_tokens']:,}</td></tr>
<tr><th>Output tokens served</th><td>{metrics['output_tokens']:,}</td></tr>
<tr><th>Total tokens processed</th><td>{metrics['total_tokens']:,}</td></tr>
<tr><th>Counting since</th><td>{started}</td></tr>
<tr><th>Last request</th><td>{last_request}</td></tr>
</tbody>
</table>

<h2>Data handling</h2>
<p>Prompt and response contents are not stored. The service retains token hashes, issuance and request timestamps, aggregate token counts, and request totals for access control and capacity management.</p>
<hr>
<p>Experimental service. Availability and limits may change.</p>
</body>
</html>"""
    return web.Response(
        text=document,
        content_type="text/html",
        charset="utf-8",
        headers={"Cache-Control": "no-store"},
    )


async def health(_: web.Request) -> web.Response:
    configured = bool(UPSTREAM_URL and UPSTREAM_API_TOKEN and UPSTREAM_MODEL)
    return web.json_response({"status": "ok" if configured else "needs_configuration"})


async def session_context(app: web.Application):
    timeout = ClientTimeout(total=180, connect=15, sock_read=120)
    async with ClientSession(timeout=timeout, connector=TCPConnector(limit=MAX_CONCURRENT_REQUESTS)) as session:
        app["session"] = session
        yield


app = web.Application(client_max_size=MAX_REQUEST_BYTES)
app.cleanup_ctx.append(session_context)
app.router.add_get("/", index)
app.router.add_get("/health", health)
app.router.add_post("/api/token", issue_token)
app.router.add_options("/api/token", options)
app.router.add_get("/api/v1/models", models)
app.router.add_options("/api/v1/{tail:.*}", options)
for public_path in API_PATHS:
    app.router.add_post(public_path, proxy)
app.router.add_static("/assets", STATIC_PATH, show_index=False, append_version=True)


if __name__ == "__main__":
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, access_log=None)
