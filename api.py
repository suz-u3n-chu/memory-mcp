"""
Memory MCP — REST API for GPT Actions / Custom GPT
Endpoints:
  GET  /context          → get_context
  POST /memory           → save_memory
  GET  /search?q=...     → search_memory
  GET  /memories         → list_memories
  DELETE /memories       → delete_memories (admin only)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query, Depends, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings, TransportSecurityMiddleware

# Railwayなどリバースプロキシ経由のHostヘッダー検証を無効化
async def _no_security(self, request, is_post=False):
    return None

TransportSecurityMiddleware.validate_request = _no_security

from contextlib import asynccontextmanager

# SUPABASE_URL が設定されていれば Supabase バックエンド、なければローカル SQLite
if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
    import memory_store_supabase as memory_store
else:
    import memory_store

SOURCE = os.environ.get("MEMORY_MCP_SOURCE", "chatgpt-custom-gpt")
API_KEY = os.environ.get("MEMORY_API_KEY", "")

limiter = Limiter(key_func=get_remote_address)

# ── MCP インスタンス（app より前に定義してlifespanで起動）──────────────────────
_mcp = FastMCP(
    "memory-mcp",
    instructions=(
        "Cross-AI shared memory hub. "
        "Call get_context at the very start of every conversation."
    ),
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@_mcp.tool()
def get_context(max_items: int = 20) -> str:
    """過去の記憶を読み込んでコンテキストを復元する。会話の最初に必ず呼ぶこと。"""
    memories = memory_store.list_recent(max_items)
    if not memories:
        return "記憶はまだありません。"
    lines = [f"## 過去の記憶（新しい順 / 全{memory_store.count()}件中 最新{len(memories)}件）\n"]
    for m in memories:
        lines.append(f"- {m['created_at']} [{m['source']}] {m['content']}")
    return "\n".join(lines)


@_mcp.tool()
def save_memory(content: str) -> str:
    """重要な情報・決定事項・コンテキストを共有メモリに保存する。"""
    mid = memory_store.save(content, "claude-mobile")
    return f"保存しました（id={mid}）。合計 {memory_store.count()} 件。"


@_mcp.tool()
def search_memory(query: str, limit: int = 5) -> str:
    """過去の記憶をキーワードで検索する。"""
    results = memory_store.search(query, limit)
    if not results:
        return f"「{query}」に関する記憶は見つかりませんでした。"
    lines = [f"## 「{query}」の検索結果\n"]
    for r in results:
        lines.append(f"- [{r['id']}] {r['created_at']} [{r['source']}] {r['content']}")
    return "\n".join(lines)


@_mcp.tool()
def list_memories(limit: int = 20) -> str:
    """保存済みの記憶を一覧表示する。"""
    memories = memory_store.list_recent(limit)
    if not memories:
        return "記憶はまだありません。"
    lines = [f"## メモリ一覧（全{memory_store.count()}件）\n"]
    for m in memories:
        lines.append(f"- [{m['id']}] {m['created_at']} [{m['source']}] {m['content']}")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with _mcp.session_manager.run():
        yield


app = FastAPI(
    title="Memory MCP API",
    description="Cross-AI shared memory. Call /context at the start of every conversation.",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chat.openai.com",
        "https://chatgpt.com",
        "https://claude.ai",
        "https://www.claude.ai",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

MAX_BODY_BYTES = 64 * 1024  # 64KB


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
    return await call_next(request)


def verify_key(key: str = Security(api_key_header)):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured on server")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


# ── Models ────────────────────────────────────────────────────────────────

class SaveRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    source: str = Field(default="", max_length=100)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/context", summary="会話開始時に必ず呼ぶ。過去の記憶を返す。")
@limiter.limit("60/minute")
def get_context(request: Request, max_items: int = Query(20, ge=1, le=100), _=Depends(verify_key)):
    memories = memory_store.list_recent(max_items)
    total = memory_store.count()
    return {
        "total": total,
        "returned": len(memories),
        "memories": memories,
    }


@app.post("/memory", summary="重要な情報・決定事項をメモリに保存する。")
@limiter.limit("30/minute")
def save_memory(request: Request, body: SaveRequest, _=Depends(verify_key)):
    source = body.source or SOURCE
    mid = memory_store.save(body.content, source)
    return {"id": mid, "total": memory_store.count()}


@app.get("/search", summary="過去の記憶をキーワード検索する。")
@limiter.limit("60/minute")
def search_memory(
    request: Request,
    q: str = Query(..., min_length=1, max_length=200, description="検索キーワード"),
    limit: int = Query(5, ge=1, le=20),
    _=Depends(verify_key),
):
    results = memory_store.search(q, limit)
    return {"query": q, "results": results}


@app.get("/memories", summary="保存済みメモリの一覧を返す。")
@limiter.limit("30/minute")
def list_memories(request: Request, limit: int = Query(20, ge=1, le=100), _=Depends(verify_key)):
    memories = memory_store.list_recent(limit)
    return {"total": memory_store.count(), "memories": memories}


@app.delete("/memories", summary="全メモリを削除する（慎重に）。")
@limiter.limit("5/minute")
def delete_memories(request: Request, _=Depends(verify_key)):
    n = memory_store.delete_all()
    return {"deleted": n}


@app.get("/health")
def health():
    return {"status": "ok", "total_memories": memory_store.count()}


# streamable_http_app() でセッションマネージャを初期化してから
# lifespan なし Starlette アプリとしてマウント
# （子の lifespan を除外することで session_manager.run() の二重起動を防ぐ）
from starlette.applications import Starlette as _Starlette
from starlette.routing import Route as _Route

_mcp_inner = _mcp.streamable_http_app()  # セッションマネージャ生成
_mcp_handler = _mcp_inner.routes[0].endpoint  # StreamableHTTPASGIApp を取り出す

# /mcp と /mcp/ 両方をリダイレクトなしで処理するASGIラッパー
# app.mount() はスラッシュなしを307リダイレクトするため、ミドルウェアで対応
_fastapi_app = app

_CORS_HEADERS = [
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, POST, DELETE, OPTIONS"),
    (b"access-control-allow-headers", b"*"),
]


class _AppWithMCP:
    """FastAPI の前段で /mcp/* をインターセプトして MCP ハンドラに渡す"""
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            # OPTIONSプリフライトに即応 (CORSミドルウェアをバイパスするため手動対応)
            if scope.get("method") == "OPTIONS":
                await send({"type": "http.response.start", "status": 204, "headers": _CORS_HEADERS})
                await send({"type": "http.response.body", "body": b""})
                return

            async def send_with_cors(event):
                if event["type"] == "http.response.start":
                    event = {**event, "headers": list(event.get("headers", [])) + _CORS_HEADERS}
                await send(event)

            scope = {**scope, "path": "/"}
            await _mcp_handler(scope, receive, send_with_cors)
        else:
            await _fastapi_app(scope, receive, send)

app = _AppWithMCP()
