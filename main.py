import os
import json
import hashlib
from datetime import datetime
from email.message import EmailMessage
import smtplib
import difflib
import random

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# =========================
# Config
# =========================
DB_DIR = "chroma_db"                  # folder created by ingest.py
COLLECTION_NAME = "manu_shah_book"    # must match ingest.py
EMBED_MODEL = "text-embedding-3-small"
POEM_MODEL = os.getenv("POEM_MODEL", "gpt-4.1-mini")

HISTORY_PATH = "poem_history.json"
MAX_RETRIES = 6

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")      # your gmail address
SMTP_PASS = os.getenv("SMTP_PASS")      # Gmail App Password (16 chars)
EMAIL_TO = os.getenv("EMAIL_TO", "")    # comma-separated
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX", "Daily Memoir Poem")
SIGNATURE_NAME = os.getenv("SIGNATURE_NAME", "Milan")

THEMES = [
    "early life and identity",
    "immigration and starting over",
    "grit, risk, and perseverance",
    "family, love, and duty",
    "integrity and doing things right",
    "leadership through humility",
    "building something lasting",
    "gratitude and generosity",
    "reflection, meaning, and faith",
    "legacy and what we leave behind",
]

client = OpenAI()  # uses OPENAI_API_KEY env var


# =========================
# Helpers
# =========================
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_history() -> dict:
    if not os.path.exists(HISTORY_PATH):
        return {"poems": []}
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"poems": []}

def save_history(hist: dict) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

def pick_theme(date_key: str) -> str:
    seed = int(hashlib.md5(date_key.encode("utf-8")).hexdigest(), 16)
    return THEMES[seed % len(THEMES)]

def too_similar(a: str, b: str, threshold: float = 0.92) -> bool:
    if not a or not b:
        return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold

def embed_one(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# =========================
# Retrieval (Chroma)
# =========================
def get_book_context(query: str, k: int = 7) -> str:
    chroma = chromadb.PersistentClient(
        path=DB_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    col = chroma.get_collection(COLLECTION_NAME)

    q_emb = embed_one(query)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents"]
    )

    docs = res["documents"][0] if res and res.get("documents") else []
    snippets = []
    for d in docs:
        if d and d.strip():
            snippets.append(f"[Book context] {d.strip()}")
    return "\n\n".join(snippets)


# =========================
# Poem generation
# =========================
def generate_poem(today_human: str, theme: str, context: str, avoid_lines: list[str]) -> str:
    avoid_block = ""
    if avoid_lines:
        avoid_block = "Avoid reusing or echoing these lines/phrases:\n" + "\n".join(
            f"- {x}" for x in avoid_lines[:14]
        )

    # Add a small “randomizer” so poems vary even if run twice the same day
    salt = sha256_text(f"{today_human}|{random.random()}")[:10]

    prompt = f"""
You are writing an ORIGINAL poem inspired by a memoir titled "The Life and Times of Manu Shah."
The author owns the rights.

Hard rules:
- Do NOT quote the book verbatim.
- Do NOT closely paraphrase any sentence from the book.
- Use the provided book context ONLY to ground themes and details, not as text to copy.
- 14–22 lines, free verse (no rhyming requirement).
- Vivid, clear language. Not overly flowery.
- Include: (1) one concrete image (object/place/sound), (2) one emotional turn (a pivot).
- End with one memorable closing line.
- No headings, no bullet points.

Date: {today_human}
Theme focus: {theme}
Uniqueness salt: {salt}

{avoid_block}

Book context (inspiration only):
{context}

Now write the poem.
""".strip()

    r = client.responses.create(
        model=POEM_MODEL,
        input=prompt,
        temperature=0.95,
        max_output_tokens=500,
    )

    out = []
    for item in r.output:
        for c in item.content:
            if c.type == "output_text":
                out.append(c.text)

    poem = "\n".join(out).strip()
    return poem


# =========================
# Email
# =========================
def send_email(poem: str, today_human: str) -> None:
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("Missing SMTP_USER/SMTP_PASS env vars. Gmail requires an App Password.")
    if not EMAIL_TO.strip():
        raise RuntimeError("Missing EMAIL_TO env var (comma-separated recipient list).")

    recipients = [e.strip() for e in EMAIL_TO.split(",") if e.strip()]

    subject = f"{SUBJECT_PREFIX} — {today_human}"
    body = f"Hi,\n\nHere’s your poem for {today_human}:\n\n{poem}\n\n– {SIGNATURE_NAME}\n"

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


# =========================
# Main
# =========================
def main():
    # Required: OPENAI_API_KEY must be set in this terminal session
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in this terminal session.")

    today_key = datetime.now().strftime("%Y-%m-%d")
    today_human = datetime.now().strftime("%B %d, %Y")
    theme = pick_theme(today_key)

    hist = load_history()
    poems = hist.get("poems", [])

    # Build hashes + recent text list for non-repetition
    existing_hashes = {p.get("hash") for p in poems if p.get("hash")}
    recent_poems = poems[-25:] if len(poems) > 25 else poems

    # Avoid lines: first+last lines from recent poems
    avoid_lines = []
    for p in recent_poems:
        lines = [ln.strip() for ln in p.get("text", "").splitlines() if ln.strip()]
        if lines:
            avoid_lines.append(lines[0])
            avoid_lines.append(lines[-1])

    # Query for context grounded in theme/date
    retrieval_query = f"{theme}; key moments; values; lessons; turning points; {today_human}"
    context = get_book_context(retrieval_query, k=7)

    best_poem = None
    best_hash = None

    for _ in range(MAX_RETRIES):
        poem = generate_poem(today_human, theme, context, avoid_lines)

        # Basic validation
        if not poem or len(poem.splitlines()) < 10:
            continue

        h = sha256_text(poem)
        if h in existing_hashes:
            continue

        # Similarity check vs recent poems (cheap text compare)
        if any(too_similar(poem, p.get("text", "")) for p in recent_poems):
            continue

        best_poem = poem
        best_hash = h
        break

    if not best_poem:
        raise RuntimeError("Could not generate a sufficiently unique poem after retries.")

    # Save history
    poems.append({"date": today_key, "theme": theme, "hash": best_hash, "text": best_poem})
    hist["poems"] = poems
    save_history(hist)

    # Send email
    send_email(best_poem, today_human)
    print("✅ Sent daily poem and updated poem_history.json")


if __name__ == "__main__":
    main()

