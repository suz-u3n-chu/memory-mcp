"""
memory_store — Supabase PostgreSQL バックエンド

環境変数:
  SUPABASE_URL              https://xxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY service_role JWT

memories テーブルのスキーマ (抜粋):
  id          uuid PK
  summary     text   (= content)
  source_app  text   (= source)
  occurred_at timestamptz
  importance  int
  deleted_at  timestamptz (soft delete)
"""

import os
import unicodedata
import re
from datetime import datetime, timezone
from pathlib import Path

# ── .env 読み込み ──────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k, _v)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

MAX_MEMORIES = 500
COMPRESS_TAKE = 100

# ── Supabase クライアント ───────────────────────────────────────────────────────
_sb = None

def _get_sb():
    global _sb
    if _sb is None:
        from supabase import create_client
        _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _sb


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: dict) -> dict:
    """Supabase の memories 行を memory_store 共通形式に変換"""
    return {
        "id":         row.get("id", ""),
        "content":    row.get("summary", ""),
        "source":     row.get("source_app", "unknown"),
        "created_at": (row.get("occurred_at") or row.get("created_at") or "")[:19].replace("T", " "),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

def save(content: str, source: str = "unknown") -> str:
    sb = _get_sb()
    import uuid
    new_id = str(uuid.uuid4())
    uid = os.environ.get("SUPABASE_USER_ID") or None

    # 1. raw_logs に INSERT（memories の外部キー親）
    raw_row = {
        "id":      new_id,
        "source":  source,
        "role":    "system",
        "content": content,
    }
    if uid:
        raw_row["user_id"] = uid
    sb.table("raw_logs").insert(raw_row).execute()

    # 2. memories に INSERT
    mem_row = {
        "id":          new_id,
        "raw_log_id":  new_id,
        "summary":     content,
        "source_app":  source,
        "occurred_at": _now_iso(),
        "importance":  3,
    }
    if uid:
        mem_row["user_id"] = uid
    sb.table("memories").insert(mem_row).execute()

    if count() > MAX_MEMORIES:
        _compress_old_memories()

    return new_id


def list_recent(limit: int = 20) -> list[dict]:
    sb = _get_sb()
    res = sb.table("memories") \
        .select("id, summary, source_app, occurred_at, created_at") \
        .is_("deleted_at", "null") \
        .order("occurred_at", desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_dict(r) for r in (res.data or [])]


def search(query: str, limit: int = 5) -> list[dict]:
    sb = _get_sb()
    # Supabase の全文検索 RPC があれば使う、なければ ilike フォールバック
    try:
        res = sb.rpc("search_memories", {"query_text": query, "match_limit": limit}).execute()
        if res.data:
            return [_row_to_dict(r) for r in res.data]
    except Exception:
        pass

    # フォールバック: ilike
    res = sb.table("memories") \
        .select("id, summary, source_app, occurred_at, created_at") \
        .is_("deleted_at", "null") \
        .ilike("summary", f"%{query}%") \
        .order("occurred_at", desc=True) \
        .limit(limit) \
        .execute()
    return [_row_to_dict(r) for r in (res.data or [])]


def count() -> int:
    sb = _get_sb()
    res = sb.table("memories") \
        .select("id", count="exact") \
        .is_("deleted_at", "null") \
        .execute()
    return res.count or 0


def delete_all() -> int:
    sb = _get_sb()
    # soft delete
    res = sb.table("memories") \
        .update({"deleted_at": _now_iso()}) \
        .is_("deleted_at", "null") \
        .execute()
    return len(res.data or [])


# ── 自動圧縮 ──────────────────────────────────────────────────────────────────

def _compress_old_memories():
    sb = _get_sb()
    rows = sb.table("memories") \
        .select("id, summary") \
        .is_("deleted_at", "null") \
        .order("occurred_at", desc=False) \
        .limit(COMPRESS_TAKE) \
        .execute().data or []

    if not rows:
        return

    texts = [r["summary"] for r in rows]
    ids   = [r["id"]      for r in rows]
    compressed = _compress_with_llm(texts)

    # 古い件を soft delete
    sb.table("memories").update({"deleted_at": _now_iso()}) \
        .in_("id", ids).execute()

    # 圧縮版を保存
    import uuid
    new_id = str(uuid.uuid4())
    sb.table("memories").insert({
        "id":          new_id,
        "raw_log_id":  new_id,
        "summary":     compressed,
        "source_app":  "auto-compress",
        "occurred_at": _now_iso(),
        "importance":  3,
    }).execute()


def _compress_with_llm(texts: list[str]) -> str:
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
            msg = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=800,
                messages=[{"role": "user", "content": (
                    "以下は過去の会話メモリです。重要な情報・決定事項・ユーザーの好みを"
                    "漏らさず、300字以内の日本語で圧縮してください。\n\n" + numbered
                )}]
            )
            return f"【圧縮メモリ】{msg.content[0].text.strip()}"
        except Exception:
            pass
    joined = " / ".join(texts)
    return f"【圧縮メモリ】{joined[:500]}{'…' if len(joined) > 500 else ''}"
