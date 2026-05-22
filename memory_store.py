import sqlite3
import unicodedata
from datetime import datetime
from pathlib import Path

import os as _os
_db_env = _os.environ.get("MEMORY_DB_PATH")
DB_PATH = Path(_db_env) if _db_env else (Path.home() / ".local" / "share" / "memory-mcp" / "memories.db")

MAX_MEMORIES  = 500   # この件数を超えたら古いメモリを圧縮する
COMPRESS_TAKE = 100   # 一度に圧縮する古いメモリの件数
COMPRESS_KEEP = 400   # 圧縮後に残す件数（= MAX - COMPRESS_TAKE + 1 圧縮分）

_conn: sqlite3.Connection | None = None


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _init_db(_conn)
    return _conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content    TEXT    NOT NULL,
            source     TEXT    NOT NULL DEFAULT 'unknown',
            created_at TEXT    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, source, content=memories, content_rowid=id);

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, source)
            VALUES (new.id, new.content, new.source);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, source)
            VALUES ('delete', old.id, old.content, old.source);
        END;
    """)
    conn.commit()


def _normalize(text: str) -> str:
    """NFKC正規化（全角→半角、ひらがな/カタカナ統一など）して小文字化"""
    return unicodedata.normalize("NFKC", text).lower()


def _compress_with_llm(texts: list[str]) -> str:
    """
    古いメモリ群を LLM（Claude haiku）で要約・圧縮する。
    ANTHROPIC_API_KEY が未設定の場合は単純結合フォールバック。
    """
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
                messages=[{
                    "role": "user",
                    "content": (
                        "以下は過去の会話メモリです。重要な情報・決定事項・ユーザーの好みを"
                        "漏らさず、300字以内の日本語で圧縮してください。\n\n"
                        f"{numbered}"
                    )
                }]
            )
            return f"【圧縮メモリ】{msg.content[0].text.strip()}"
        except Exception as e:
            pass  # フォールバックへ

    # フォールバック: 先頭から結合して500字に切る
    joined = " / ".join(texts)
    summary = joined[:500] + ("…" if len(joined) > 500 else "")
    return f"【圧縮メモリ】{summary}"


def _compress_old_memories() -> None:
    """
    最古の COMPRESS_TAKE 件を圧縮して1件に置き換える。
    save() 内から非同期的に呼ばれる（次回保存時に実行）。
    """
    conn = get_conn()

    # 最古 COMPRESS_TAKE 件を取得
    rows = conn.execute(
        "SELECT id, content FROM memories ORDER BY id ASC LIMIT ?",
        (COMPRESS_TAKE,)
    ).fetchall()

    if not rows:
        return

    ids    = [r["id"] for r in rows]
    texts  = [r["content"] for r in rows]
    oldest = ids[0]
    newest = ids[-1]

    # LLM で圧縮
    compressed = _compress_with_llm(texts)

    # 古い件を削除 → 圧縮版を保存
    conn.execute(
        f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})",
        ids
    )
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO memories (content, source, created_at) VALUES (?, ?, ?)",
        (compressed, "auto-compress", now)
    )
    conn.commit()


def save(content: str, source: str = "unknown") -> int:
    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO memories (content, source, created_at) VALUES (?, ?, ?)",
        (content, source, now),
    )
    conn.commit()
    new_id = cur.lastrowid

    # ── 上限チェック → 古いメモリを圧縮 ────────────────────────
    if count() > MAX_MEMORIES:
        _compress_old_memories()

    return new_id


def _tokenize(text: str) -> list[str]:
    """
    ハイブリッドトークン分割:
    1. スペース・句読点で単語分割
    2. 日本語の連続文字列から 2〜4文字の n-gram を抽出
    → 「プロジェクト管理」を「プロジェ」「ロジェク」「ジェクト」「管理」等に分解
    """
    import re
    norm = _normalize(text)

    # 通常の単語分割
    word_tokens = re.split(r"[\s　、。・,.\-/()（）「」【】\[\]]+", norm)
    tokens = set(t for t in word_tokens if len(t) >= 2)

    # 日本語文字の連続部分を抽出してn-gramも追加
    jp_runs = re.findall(r"[ぁ-んァ-ヶー一-龯々]+", norm)
    for run in jp_runs:
        for n in (2, 3, 4):
            for i in range(len(run) - n + 1):
                tokens.add(run[i:i+n])

    return list(tokens)


def search(query: str, limit: int = 5) -> list[dict]:
    """
    ハイブリッド検索:
      1. FTS5 全文検索（英語・ASCII向け）
      2. トークン分割 + Python側部分一致（日本語対応）
    両結果をマージして重複排除し、最大 limit 件を返す。
    """
    conn = get_conn()
    seen_ids: set[int] = set()
    results: list[dict] = []

    # ── 1. FTS5 ────────────────────────────────────────────────
    try:
        rows = conn.execute(
            """SELECT m.id, m.content, m.source, m.created_at
               FROM memories_fts f
               JOIN memories m ON m.id = f.rowid
               WHERE memories_fts MATCH ?
               ORDER BY rank
               LIMIT ?""",
            (query, limit),
        ).fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                results.append(dict(r))
    except Exception:
        pass

    # ── 2. トークン分割 + 部分一致（日本語・FTS5が空振りした時） ──
    if len(results) < limit:
        tokens = _tokenize(query)
        if not tokens:
            tokens = [_normalize(query)]

        candidates = conn.execute(
            "SELECT id, content, source, created_at FROM memories ORDER BY id DESC LIMIT 300"
        ).fetchall()

        scored: list[tuple[int, dict]] = []
        for r in candidates:
            if r["id"] in seen_ids:
                continue
            norm_content = _normalize(r["content"])
            score = sum(1 for t in tokens if t in norm_content)
            if score > 0:
                scored.append((score, dict(r)))

        scored.sort(key=lambda x: -x[0])
        for score, row in scored:
            seen_ids.add(row["id"])
            results.append(row)
            if len(results) >= limit:
                break

    return results[:limit]


def list_recent(limit: int = 20) -> list[dict]:
    conn = get_conn()
    cur = conn.execute(
        "SELECT id, content, source, created_at FROM memories ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in cur.fetchall()]


def delete_all() -> int:
    conn = get_conn()
    cur = conn.execute("DELETE FROM memories")
    conn.commit()
    return cur.rowcount


def count() -> int:
    conn = get_conn()
    (n,) = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
    return n
