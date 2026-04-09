"""
Driftwatch CLI — run as: python -m cli <command>

Commands:
  generate-key
  golden-set add --prompt "..." --description "..."
  golden-set list
  golden-set clear
"""

import argparse
import asyncio
import hashlib
import os
import secrets
import sys
from datetime import datetime


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        print(f"Error: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


# ── generate-key ──────────────────────────────────────────────────────────────

def cmd_generate_key() -> None:
    from app.schemas.cli import KeygenResult

    raw = secrets.token_hex(32)
    api_key = f"dwk_{raw}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    result = KeygenResult(api_key=api_key, key_hash=key_hash)
    print(f"Your API key: {result.api_key}")
    print(f"Add to .env:  DRIFTWATCH_KEY_HASH={result.key_hash}")


# ── golden-set add ─────────────────────────────────────────────────────────────

async def _golden_set_add(prompt: str, description: str) -> None:
    import httpx
    import torch
    from sentence_transformers import SentenceTransformer

    from db.session import AsyncSessionLocal
    from db.vector_store import get_vector_store

    # OPENAI_API_KEY is read only here in the CLI — never stored, never logged
    api_key = _require_env("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{openai_base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()

    data = response.json()
    response_text: str = data["choices"][0]["message"]["content"]

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding: list[float] = st_model.encode(response_text, device=device).tolist()

    vector_store = get_vector_store()
    await vector_store.insert_golden(prompt=response_text, embedding=embedding, description=description)
    count = await vector_store.count_golden()
    print(f"Added. Golden set size: {count}")


def cmd_golden_set_add(prompt: str, description: str) -> None:
    asyncio.run(_golden_set_add(prompt, description))


# ── golden-set list ────────────────────────────────────────────────────────────

async def _golden_set_list() -> None:
    from sqlalchemy import select

    from app.schemas.cli import GoldenSetEntry
    from db.models import GoldenSet
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(GoldenSet).order_by(GoldenSet.created_at))
        rows = result.scalars().all()

    if not rows:
        print("Golden set is empty.")
        return

    entries = [GoldenSetEntry(id=r.id, description=r.description, created_at=r.created_at) for r in rows]
    print(f"{'ID':<38}  {'Description':<30}  Created At")
    print("-" * 90)
    for e in entries:
        print(f"{str(e.id):<38}  {e.description:<30}  {e.created_at.isoformat()}")


def cmd_golden_set_list() -> None:
    asyncio.run(_golden_set_list())


# ── golden-set clear ───────────────────────────────────────────────────────────

async def _golden_set_clear() -> None:
    from sqlalchemy import delete, func, select

    from db.models import GoldenSet
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        count_result = await session.execute(select(func.count()).select_from(GoldenSet))
        count = count_result.scalar_one()

    confirm = input(f"This will delete all {count} golden set entries. Type 'yes' to confirm: ")
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    async with AsyncSessionLocal() as session:
        await session.execute(delete(GoldenSet))
        await session.commit()

    print("Cleared.")


def cmd_golden_set_clear() -> None:
    asyncio.run(_golden_set_clear())


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m cli", description="Driftwatch CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate-key", help="Generate a new Driftwatch API key")

    gs_parser = subparsers.add_parser("golden-set", help="Manage the golden set")
    gs_sub = gs_parser.add_subparsers(dest="subcommand", required=True)

    add_parser = gs_sub.add_parser("add", help="Add an entry to the golden set")
    add_parser.add_argument("--prompt", required=True, help="Prompt to send to OpenAI")
    add_parser.add_argument("--description", required=True, help="Human-readable label for this entry")

    gs_sub.add_parser("list", help="List all golden set entries")
    gs_sub.add_parser("clear", help="Clear all golden set entries")

    args = parser.parse_args()

    if args.command == "generate-key":
        cmd_generate_key()
    elif args.command == "golden-set":
        if args.subcommand == "add":
            cmd_golden_set_add(args.prompt, args.description)
        elif args.subcommand == "list":
            cmd_golden_set_list()
        elif args.subcommand == "clear":
            cmd_golden_set_clear()


if __name__ == "__main__":
    main()
