"""
Leaders Academia — RAG Chatbot
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> Gradio chat UI.
"""

import asyncio
import os
import time

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types
from groq import Groq
import gradio as gr
import requests
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3"

TEAM_HEAD_NUMBER = "0311-1534344"

# Courses to always mention first, in this order, when giving any kind of
# course list or recommendation — matches the website's own priority order.
PRIORITY_COURSES = [
    "AI Automation",
    "Web Development",
    "Video Editing",
    "Graphic Design",
]

HUMAN_HANDOFF_KEYWORDS = [
    "human", "real person", "agent", "representative",
    "insan se baat", "banda se baat", "customer support",
]

# ---------- Load API key from Railway's environment variables (never hardcode it) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it under this project's "
        "Variables tab in the Railway dashboard."
    )
# 20 s hard timeout so a slow/hung Gemini call can never stall a reply for a minute
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=20000),  # milliseconds
)

# ---------- Groq: free backup, used only when Gemini is busy/rate-limited ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = (
    Groq(api_key=GROQ_API_KEY, timeout=20.0, max_retries=1) if GROQ_API_KEY else None
)

# ---------- WhatsApp Cloud API config (from Meta's "Try it out" page) ----------
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
# A password you make up yourself — must match exactly what you type into
# Meta's webhook "Verify token" field. Not secret from Meta, just needs to match.
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "leaders_academia_verify")


# ---------- Build the knowledge base once, when the Space starts ----------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=1400, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


print("Loading PDF and building the knowledge base...")
full_text = load_pdf_text(PDF_PATH)
chunks = chunk_text(full_text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embed_model.encode(chunks, convert_to_numpy=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print(f"Knowledge base ready: {len(chunks)} chunks indexed.")


# ---------- Complete, guaranteed course name list ----------
# Semantic search (top-k chunks) can miss some courses when the user asks for
# a full list — this pulls every course name directly from the PDF's
# "Courses & Curriculum Catalog" section, so listing is never incomplete.
def extract_course_names(text):
    start = text.find("Courses & Curriculum Catalog")
    end = text.find("Instructors & Mentors Directory")
    section = text[start:end] if start != -1 and end != -1 else text
    lines = [line.strip() for line in section.split("\n")]
    names = []
    for i in range(len(lines) - 1):
        if lines[i] and lines[i + 1].startswith("Category:"):
            names.append(lines[i])
    return names


def order_by_priority(all_names, priority_keywords):
    """Priority courses first (matched loosely — ignoring punctuation/spacing
    differences like 'AI Automation' vs 'AI & Automation'), then everything
    else in its original order."""
    import re

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    ordered = []
    used = set()
    for keyword in priority_keywords:
        keyword_norm = normalize(keyword)
        for name in all_names:
            if name not in used and keyword_norm in normalize(name):
                ordered.append(name)
                used.add(name)
                break
    for name in all_names:
        if name not in used:
            ordered.append(name)
    return ordered


ALL_COURSE_NAMES = order_by_priority(extract_course_names(full_text), PRIORITY_COURSES)
print(f"Extracted {len(ALL_COURSE_NAMES)} course names: {ALL_COURSE_NAMES}")
COURSE_LIST_TEXT = "\n".join(f"- {name}" for name in ALL_COURSE_NAMES)


# ---------- RAG logic ----------
def retrieve_chunks(query, top_k=6):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


# ---------- Per-user conversation memory ----------
# Keyed by WhatsApp number. Keeps the last few exchanges so follow-up
# questions like "is course ka instructor kon hai" work correctly — without
# this, every message was treated as a brand-new, context-free question.
conversation_history = {}
MAX_HISTORY_TURNS = 5  # user+assistant pairs kept per person


def get_history(user_id):
    return conversation_history.get(user_id, [])


def add_to_history(user_id, role, text):
    history = conversation_history.setdefault(user_id, [])
    history.append((role, text))
    # Keep only the most recent turns so the prompt doesn't grow forever
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        conversation_history[user_id] = history[-max_messages:]


def format_history(history):
    if not history:
        return "(no earlier messages — this is the start of the conversation)"
    lines = []
    for role, text in history:
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_retrieval_query(user_id, user_question):
    """Pull in the last user message too, so a short follow-up like 'iska
    instructor kon hai' still retrieves the RIGHT course's chunks, not a
    random one."""
    history = get_history(user_id)
    recent_user_lines = [text for role, text in history[-4:] if role == "user"]
    return " ".join(recent_user_lines + [user_question])


def wants_human(text):
    text = text.lower()
    return any(keyword in text for keyword in HUMAN_HANDOFF_KEYWORDS)


def is_transient_error(e):
    """Errors that mean 'Gemini is busy / out of quota / too slow right now'."""
    text = str(e).lower()
    return any(
        marker in text
        for marker in (
            "429", "503", "500", "502", "504", "quota", "unavailable",
            "resource_exhausted", "overloaded", "timeout", "timed out", "deadline",
        )
    )


# Circuit breaker: when Gemini fails, skip it for a few minutes and go straight to
# the backup, instead of making every single message wait for Gemini to fail first.
GEMINI_COOLDOWN_SECONDS = 180
GEMINI_TTS_COOLDOWN_SECONDS = 600
_gemini_down_until = 0.0
_gemini_tts_down_until = 0.0


def gemini_is_up():
    return time.time() >= _gemini_down_until


def mark_gemini_down():
    global _gemini_down_until
    _gemini_down_until = time.time() + GEMINI_COOLDOWN_SECONDS
    print(f"Gemini marked unavailable for {GEMINI_COOLDOWN_SECONDS}s, using Groq meanwhile.")


def gemini_tts_is_up():
    return time.time() >= _gemini_tts_down_until


def mark_gemini_tts_down():
    global _gemini_tts_down_until
    _gemini_tts_down_until = time.time() + GEMINI_TTS_COOLDOWN_SECONDS
    print(f"Gemini TTS marked unavailable for {GEMINI_TTS_COOLDOWN_SECONDS}s, using edge-tts meanwhile.")


def generate_text(prompt):
    """Try Gemini first; if it's busy, out of quota or slow, fall back to Groq
    right away — same prompt goes to both, so the tone and style of the reply
    stays the same no matter which one actually answers."""
    if gemini_is_up():
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
            if response.text:
                return response.text
            print("Gemini returned an empty reply, falling back to Groq.")
        except Exception as e:
            print(f"Gemini failed, falling back to Groq: {e}")
            if is_transient_error(e):
                mark_gemini_down()

    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Groq also failed: {e}")
    else:
        print("GROQ_API_KEY is not set — no backup available, only Gemini was tried.")

    return (
        "Is waqt thora zyada rush hai is liye jawab dene mein masla ho raha "
        f"hai. Chand minute mein dobara message kar dein, ya seedha hamari "
        f"team se baat kar lein: {TEAM_HEAD_NUMBER}"
    )


def rag_answer(user_question, user_id=None, voice=False):
    # Direct handoff — skip the model entirely for this case
    if wants_human(user_question):
        reply = (
            f"Zaroor! Aap hamari team head se seedha rabta kar sakte hain: "
            f"{TEAM_HEAD_NUMBER}"
        )
        if user_id:
            add_to_history(user_id, "user", user_question)
            add_to_history(user_id, "assistant", reply)
        return reply

    history = get_history(user_id) if user_id else []
    retrieval_query = build_retrieval_query(user_id, user_question) if user_id else user_question
    context_chunks = retrieve_chunks(retrieval_query, top_k=6)
    context = "\n\n---\n\n".join(context_chunks)

    voice_rule = ""
    if voice:
        voice_rule = """
VOICE RULE: This reply will be spoken aloud as a voice note by a FEMALE
voice — write it the way she would naturally say it out loud to a person
sitting in front of her, not like a written announcement. Keep it short
(2-3 sentences). No lists, no bullet points, no emojis, no symbols — nothing
that only makes sense in writing. Avoid stiff, formal, or "literary" Urdu
words (no heavy/classical vocabulary) — use the same everyday, casual Urdu
words a real person uses when talking, the way friends or colleagues
actually speak, not textbook Urdu. If the user spoke Urdu or Hindi, reply in
Urdu script (اردو) using this natural spoken style. If the user spoke
English, reply in natural spoken English the same way.
"""

    prompt = f"""You are Leaders Academia's official assistant, chatting with someone
on WhatsApp. Always speak as a representative of Leaders Academia — never
break character, never mention being an AI model. Your tone is that of a friendly, confident admissions officer
texting on WhatsApp: short, warm, natural, typed the way a real person types.
Never write essay-style paragraphs. No corporate phrases ("main aapko yakeen
dilati hoon", "humari credibility..."), no thanking the person for every
message, no repeating back what they just told you. Sound like a human
texting, not a brochure. Keep this tone consistent in every reply.

LENGTH RULE: Answer ONLY what was asked. Normal reply: 1-3 short lines
(under about 280 characters). Explaining a course: at most 4-5 short lines.
No bullet-point walls, no long intros. Share the website link only if they ask
for more details.

ENDING RULE: Never end a message with a filler question such as "Kya main aur
madad kar sakti hun?" or "Kuch aur janna chahenge?" or "Kya aap schedule jan-
na chahte hain?". Finish after giving the answer. The ONLY question you may
ask is the enrollment form request below.

GRAMMAR GENDER RULE: When writing in Urdu (script or Roman), always use
FEMININE verb forms when referring to yourself (e.g. "bata dungi", "kar
dungi", "madad karungi" — not the masculine "dunga"/"karunga"). Stay
consistent with this in every reply, in both text and voice messages.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

MEMORY RULE: You are in an ongoing conversation — CONVERSATION SO FAR below
has everything said until now. Use it to understand what the user means by
"this", "it", "the course", etc. Never ask the user to repeat something they
already told you earlier in this conversation. Never re-explain something
you already fully explained — if they're asking a focused follow-up (like
just the instructor's name), give a short, direct answer to exactly that,
not the whole course rundown again.

SCOPE RULE: You only help with Leaders Academia — its courses, instructors,
fees, schedules and platform. Greetings and basic pleasantries are fine —
answer those briefly and professionally, then guide the conversation back to
Leaders Academia. If someone asks for something unrelated (recipes, coding
help, general knowledge, news, homework, etc.), politely say you can only
help with Leaders Academia and ask what they'd like to know about the
courses or services. Do NOT provide the unrelated information, not even
briefly.

DATA RULE: Only use these categories of information from the CONTEXT below:
course details (name, description, modules, duration, pricing/payment),
instructor details, and core platform information. If the CONTEXT contains
anything outside these categories, ignore it completely — don't mention it,
don't repeat it, even in passing. If the CONTEXT doesn't clearly name an
instructor for the specific course being asked about, say you'll have the
team confirm — never guess or attach the wrong instructor to a course.

GUIDANCE RULE: This is mandatory — whenever you name or describe a specific
course, you MUST include a short line on the real practical benefit or
career value it offers (why it's worth taking), the way a good advisor
would, without being pushy or salesy. Never describe a course without this.
When it's natural (e.g. someone wanting more detail), point them to the
website for the full picture: https://leadersacademia.com/

PRICING RULE: This is a hard rule — follow it exactly. Never state a fee or
price unless the user's message explicitly asks about cost, price, fees, or
payment (words like "fee", "price", "kitna", "cost", "kharcha", "payment").
When someone asks about a course in general, describe what it teaches and
the real career/practical benefit it gives them — nothing about money at
all. If they then separately ask about price, give it briefly and factually,
without repeating the whole course description again.

LEAD CAPTURE RULE: When someone wants to enroll/join, or asks about fee or
payment, AND KNOWN STUDENT INFO below shows the name or city as unknown, send
this form ONCE (one short line before it, then exactly this layout, in the
user's language):
Enrollment ke liye ye details bhej dein:
Name:
City:
Course:
Never ask for the name or city in a normal sentence, never ask twice, and
never ask for anything already known in KNOWN STUDENT INFO. When they send
their details, confirm in one short line that registration is noted and the
team will contact them. Use their name only occasionally, naturally.

LEADERS ACADEMIA RULE: For questions about courses, instructors, pricing,
schedules or the platform, answer using the CONTEXT below as if it's simply
what you know — speak naturally and confidently, the way a helpful human
staff member would. Never mention "context", "the information provided",
"based on the available data" or anything that reveals you're reading from a
document.

WHEN YOU DON'T KNOW: If a Leaders Academia question isn't answered by the
CONTEXT, don't guess or invent details. Instead say naturally, in your own
words, that you don't have that detail on hand and say you'll have {TEAM_HEAD_NAME} (team head) help and give his number: {TEAM_HEAD_NUMBER}

COMPLETE COURSE LIST (authoritative — this is every course we offer, already
in the right order to mention them in): every time the user asks what
courses are offered, or wants a list of all courses, use exactly this list,
in exactly this order. Don't skip any, don't invent extra ones, don't
reorder them yourself:
{COURSE_LIST_TEXT}
{voice_rule}
{lead_info_text(user_id)}

CONVERSATION SO FAR:
{format_history(history)}

CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply like a real person texting: short, to the point, no closing question."""

    reply = generate_text(prompt)

    if user_id:
        add_to_history(user_id, "user", user_question)
        add_to_history(user_id, "assistant", reply)

    return reply


# ---------- Gradio frontend ----------
def chat_fn(message, history):
    return rag_answer(message, user_id="gradio-demo-user")


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Leaders Academia — AI Assistant",
    description="Courses, instructors aur platform ke baare mein kuch bhi puchein.",
)

# ---------- WhatsApp send helper ----------
def send_whatsapp_message(to_number, message_text):
    """Send a plain text reply back to a WhatsApp user via the Cloud API."""
    url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    if response.status_code != 200:
        print("WhatsApp send failed:", response.status_code, response.text)
    return response


# ---------- Voice message support ----------
def download_whatsapp_media(media_id):
    """WhatsApp gives us a media ID, not a direct file — this fetches the
    real download URL first, then downloads the actual audio bytes."""
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

    info_url = f"https://graph.facebook.com/v25.0/{media_id}"
    info = requests.get(info_url, headers=headers, timeout=20).json()
    media_url = info["url"]
    mime_type = info.get("mime_type", "audio/ogg")

    media_response = requests.get(media_url, headers=headers, timeout=30)
    return media_response.content, mime_type


def transcribe_audio(audio_bytes, mime_type):
    """Send the voice note to Gemini first; if it's busy, fall back to
    Groq's free Whisper transcription."""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    "Transcribe exactly what is said in this audio clip. If the "
                    "speech is Urdu or Hindi, write it in Urdu script (اردو), not "
                    "Devanagari. If it is English, write it in English. Reply with "
                    "only the transcription, nothing else.",
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )
            return response.text.strip()
        except Exception as e:
            if is_overloaded_error(e) and attempt == 0:
                time.sleep(8)
                continue
            print(f"Gemini transcription failed, falling back to Groq: {e}")
            break

    if groq_client:
        extension = "ogg" if "ogg" in mime_type else "mp3"
        transcription = groq_client.audio.transcriptions.create(
            file=(f"voice.{extension}", audio_bytes),
            model=GROQ_WHISPER_MODEL,
        )
        return transcription.text.strip()

    raise RuntimeError("Both Gemini and Groq failed to transcribe the audio.")


SINGLE_VOICE = "ur-PK-UzmaNeural"  # one consistent (female) voice, used regardless of language


def clean_text_for_speech(text):
    """Strip markdown symbols and emojis so the voice doesn't read them out."""
    import re

    text = re.sub(r"[*_#`>~|]", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Which voice engine to use. "gemini" = newer, more natural Gemini TTS (falls back
# to edge-tts automatically if it fails). "edge" = always use edge-tts.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "gemini").lower()
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")  # warm, natural female voice
GEMINI_TTS_STYLE = (
    "natural and conversational, like a real woman talking to a friend on a "
    "call — warm, relaxed, human. Never robotic, never stiff, never reading "
    "like an announcement."
)


def gemini_text_to_speech(text):
    """Ask Gemini TTS for speech. Returns raw 24 kHz mono 16-bit PCM bytes."""
    import base64

    interaction = client.interactions.create(
        model=GEMINI_TTS_MODEL,
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": GEMINI_TTS_STYLE,
                }],
            }],
        }],
        response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
        generation_config={"speech_config": [{"voice": GEMINI_TTS_VOICE}]},
    )
    return base64.b64decode(interaction.output_audio.data)


def encode_pcm_for_whatsapp(pcm_bytes, sample_rate=24000):
    """WhatsApp doesn't accept WAV/PCM, so convert to Ogg/Opus (shows up as a
    proper voice note) or MP3 as a backup. Returns (bytes, mime_type, filename)."""
    import io

    import numpy as np
    import soundfile as sf

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    try:
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="OGG", subtype="OPUS")
        return buffer.getvalue(), "audio/ogg", "reply.ogg"
    except Exception as e:
        print(f"Ogg/Opus encoding failed, trying MP3: {e}")

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="MP3")
    return buffer.getvalue(), "audio/mpeg", "reply.mp3"


async def edge_text_to_speech(text):
    """Backup voice: free Microsoft Edge neural voices (no API key needed).
    Voice is picked by script: Urdu script -> Urdu voice, otherwise English."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=SINGLE_VOICE, rate="+5%")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


async def synthesize_reply_audio(text):
    """Turn a text reply into a WhatsApp-ready voice note.
    Returns (audio_bytes, mime_type, filename)."""
    text = clean_text_for_speech(text)

    if TTS_ENGINE == "gemini":
        try:
            pcm = await asyncio.to_thread(gemini_text_to_speech, text)
            audio = await asyncio.to_thread(encode_pcm_for_whatsapp, pcm)
            print("Voice reply generated with Gemini TTS")
            return audio
        except Exception as e:
            print(f"Gemini TTS failed, falling back to edge-tts: {e}")

    audio_bytes = await edge_text_to_speech(text)
    print("Voice reply generated with edge-tts")
    return audio_bytes, "audio/mpeg", "reply.mp3"


def send_whatsapp_voice(to_number, audio_bytes, mime_type="audio/mpeg", filename="reply.mp3"):
    """Upload audio to WhatsApp's media store, then send it as a voice note.
    Returns True if it was sent, False otherwise."""
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

    upload_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    files = {"file": (filename, audio_bytes, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}
    upload_response = requests.post(
        upload_url, headers=headers, files=files, data=data, timeout=30
    ).json()
    media_id = upload_response.get("id")
    if not media_id:
        print("Voice upload failed:", upload_response)
        return False

    send_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "audio",
        "audio": {"id": media_id},
    }
    response = requests.post(
        send_url,
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    if response.status_code != 200:
        print("WhatsApp voice send failed:", response.status_code, response.text)
        return False
    return True


# ---------- FastAPI app: hosts the webhook AND the Gradio UI together ----------
app = FastAPI()

# Track WhatsApp message IDs we've already replied to, so Meta's automatic
# retries (when our server is slow to respond) don't trigger duplicate replies.
processed_message_ids = set()


# ---------- Storage + Dashboard (SQLite: survives restarts if DB_PATH is on a volume) ----------
import datetime
import hashlib
import hmac
import re
import sqlite3
import threading

from fastapi.responses import JSONResponse

TEAM_HEAD_NAME = os.environ.get("TEAM_HEAD_NAME", "Sir Ihtisham")
LONG_REPLY_CHARS = 380  # text answers longer than this are sent as a voice note instead
FOLLOWUP_AFTER_DAYS = 3
FORWARD_CHECK_AFTER_DAYS = 2
TEMPLATE_LANG = os.environ.get("TEMPLATE_LANG", "en")
FOLLOWUP_TEMPLATE = os.environ.get("FOLLOWUP_TEMPLATE", "")  # approved WhatsApp template names
FORWARD_CHECK_TEMPLATE = os.environ.get("FORWARD_CHECK_TEMPLATE", "")

DB_PATH = os.environ.get("DB_PATH", "agent.db")
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db_lock = threading.Lock()
_db.executescript("""
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, phone TEXT,
  direction TEXT, type TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS idx_msg_phone ON messages(phone);
CREATE TABLE IF NOT EXISTS leads(phone TEXT PRIMARY KEY, name TEXT DEFAULT '', city TEXT DEFAULT '',
  course TEXT DEFAULT '', status TEXT DEFAULT 'Follow-up Needed', last_user_ts REAL DEFAULT 0,
  last_agent_ts REAL DEFAULT 0, followup_sent INTEGER DEFAULT 0, forwarded_at REAL DEFAULT 0,
  forward_checked INTEGER DEFAULT 0);
""")


def db(sql, args=()):
    with _db_lock:
        cur = _db.execute(sql, args)
        _db.commit()
        return [dict(r) for r in cur.fetchall()]


def log_message(phone, direction, msg_type, text):
    """direction: 'in' (student) or 'out' (agent)."""
    now = time.time()
    db("INSERT INTO messages(time,phone,direction,type,text) VALUES(?,?,?,?,?)",
       (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), phone, direction, msg_type, text))
    db("INSERT OR IGNORE INTO leads(phone) VALUES(?)", (phone,))
    if direction == "in":
        db("UPDATE leads SET last_user_ts=?, followup_sent=0 WHERE phone=?", (now, phone))
    else:
        db("UPDATE leads SET last_agent_ts=? WHERE phone=?", (now, phone))


def record_forward_if_needed(phone, reply_text):
    if TEAM_HEAD_NUMBER in reply_text:
        db("UPDATE leads SET forwarded_at=? WHERE phone=? AND forwarded_at=0", (time.time(), phone))


# Restore short-term memory after a restart so the agent never "forgets" a student
for _r in db("SELECT DISTINCT phone FROM messages"):
    for _m in reversed(db("SELECT direction,text FROM messages WHERE phone=? ORDER BY id DESC LIMIT ?",
                          (_r["phone"], MAX_HISTORY_TURNS * 2))):
        conversation_history.setdefault(_r["phone"], []).append(
            ("user" if _m["direction"] == "in" else "assistant", _m["text"]))


def lead_info_text(phone):
    r = db("SELECT name,city FROM leads WHERE phone=?", (phone,)) if phone else []
    name = (r[0]["name"] if r else "") or "unknown"
    city = (r[0]["city"] if r else "") or "unknown"
    return f"KNOWN STUDENT INFO: name = {name}, city = {city}"


def voice_version(text):
    """Text turned into what should be spoken: no links, and never the phone number."""
    spoken = re.sub(r"https?://\S+", "", text)
    if TEAM_HEAD_NUMBER in spoken:
        spoken = spoken.replace(TEAM_HEAD_NUMBER, "")
        spoken += " Main number abhi text mein share kar rahi hun."
    return spoken


def contact_card():
    return f"{TEAM_HEAD_NAME} (Team Head)\n{TEAM_HEAD_NUMBER}"


async def refresh_lead(phone):
    """Read the conversation, save name/city/course/status. Runs only while info is
    missing, or every 3rd student message, to save API quota."""
    try:
        row = db("SELECT name,city FROM leads WHERE phone=?", (phone,))[0]
        n_in = db("SELECT COUNT(*) c FROM messages WHERE phone=? AND direction='in'", (phone,))[0]["c"]
        if row["name"] and row["city"] and n_in % 3 != 0:
            return
        lead = await asyncio.to_thread(extract_lead_info, phone, get_history(phone))
        clean = lambda v: "" if v in (None, "Not provided", "General inquiry") else str(v)
        db("UPDATE leads SET name=COALESCE(NULLIF(?,''),name), city=COALESCE(NULLIF(?,''),city),"
           " course=COALESCE(NULLIF(?,''),course), status=? WHERE phone=?",
           (clean(lead["name"]), clean(lead["address"]), clean(lead["course_interested"]),
            lead["status"], phone))
        await asyncio.to_thread(upsert_student, phone)
    except Exception as e:
        print(f"refresh_lead failed for {phone}: {e}")


# ---------- Dashboard auth: sign-in page + cookie ----------
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")


def _dash_token():
    return hmac.new(DASHBOARD_PASSWORD.encode(), b"wa-agent-dashboard", hashlib.sha256).hexdigest()


def is_authed(request: Request):
    return hmac.compare_digest(request.cookies.get("wa_dash", ""), _dash_token())


def check_dashboard_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(status_code=401, detail="Sign in required")
    return True


@app.post("/dashboard/login")
async def dashboard_login(request: Request):
    d = await request.json()
    ok = hmac.compare_digest(str(d.get("user", "")), DASHBOARD_USER) and \
        hmac.compare_digest(str(d.get("password", "")), DASHBOARD_PASSWORD)
    if not ok:
        return JSONResponse({"ok": False}, status_code=401)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("wa_dash", _dash_token(), httponly=True, samesite="lax", max_age=7 * 86400)
    return resp


@app.post("/dashboard/logout")
def dashboard_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("wa_dash")
    return resp


VIEW_SQL = {
    "live": "1=1",
    "interested": "status='Interested'",
    "not_interested": "status='Not Interested'",
    "history": "1=1",
    "followups": f"status='Interested' AND (followup_sent=1 OR (last_user_ts>0 AND ?-last_user_ts>{FOLLOWUP_AFTER_DAYS}*86400))",
    "forwarded": "forwarded_at>0",
}


@app.get("/dashboard/api/leads")
def dashboard_leads(view: str = "live", auth: bool = Depends(check_dashboard_auth)):
    cond = VIEW_SQL.get(view, "1=1")
    args = (time.time(),) if "?" in cond else ()
    rows = db(
        f"SELECT leads.*, (SELECT text FROM messages m WHERE m.phone=leads.phone ORDER BY id DESC LIMIT 1) AS last_text,"
        f" MAX(last_user_ts,last_agent_ts) AS last_ts FROM leads WHERE {cond} ORDER BY last_ts DESC LIMIT 300", args)
    now = time.time()
    for r in rows:
        r["live"] = now - r["last_ts"] < 300
        r["ago_min"] = int((now - r["last_ts"]) / 60) if r["last_ts"] else None
    return {"leads": rows}


@app.get("/dashboard/api/chat")
def dashboard_chat(phone: str, auth: bool = Depends(check_dashboard_auth)):
    msgs = db("SELECT time,direction,type,text FROM messages WHERE phone=? ORDER BY id DESC LIMIT 200", (phone,))
    lead = db("SELECT * FROM leads WHERE phone=?", (phone,))
    return {"messages": msgs[::-1], "lead": lead[0] if lead else None}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return HTMLResponse(DASHBOARD_HTML if is_authed(request) else LOGIN_HTML)


_STYLE = """
:root{--ink:#0B1F4B;--ink2:#14306E;--sky:#38BDF8;--skyl:#E0F2FE;--bg:#F3F8FD;--line:#D6E4F2}
*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:#10213f}
"""

LOGIN_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>WhatsApp Agent Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + _STYLE + """
body{display:grid;place-items:center;min-height:100vh;background:linear-gradient(135deg,var(--ink),var(--ink2) 60%,#1d5fa8)}
form{background:#fff;padding:32px;border-radius:14px;width:min(360px,92vw);display:grid;gap:12px;border-top:5px solid var(--sky)}
h1{margin:0 0 4px;font-size:22px;color:var(--ink)}input{padding:11px;border:1px solid var(--line);border-radius:8px;font-size:15px}
button{padding:12px;border:0;border-radius:8px;background:var(--ink);color:#fff;font-size:15px;cursor:pointer}
button:hover{background:var(--ink2)}#err{color:#b91c1c;font-size:13px;min-height:16px}</style></head><body>
<form onsubmit="go(event)"><h1>WhatsApp Agent Dashboard</h1><input id="u" placeholder="Username" autocomplete="username">
<input id="p" type="password" placeholder="Password" autocomplete="current-password"><div id="err"></div><button>Sign in</button></form>
<script>async function go(e){e.preventDefault();const r=await fetch('/dashboard/login',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({user:u.value,password:p.value})});if(r.ok)location.reload();else err.textContent='Wrong username or password'}</script></body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>WhatsApp Agent Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>""" + _STYLE + """
.app{display:grid;grid-template-columns:230px 330px 1fr;height:100vh}
nav{background:var(--ink);color:#fff;padding:18px 12px;display:flex;flex-direction:column;gap:6px}
nav h1{font-size:16px;margin:0 8px 14px;line-height:1.3}nav h1 span{color:var(--sky)}
nav button{background:none;border:0;color:#c7d8f2;text-align:left;padding:11px 12px;border-radius:8px;font-size:14px;cursor:pointer}
nav button:hover{background:var(--ink2)}nav button.on{background:var(--sky);color:var(--ink);font-weight:600}
nav .sp{flex:1}.list{background:#fff;border-right:1px solid var(--line);overflow-y:auto}
.list h2{margin:0;padding:16px;font-size:15px;color:var(--ink);border-bottom:1px solid var(--line);position:sticky;top:0;background:#fff}
.row{padding:12px 16px;border-bottom:1px solid #edf3fa;cursor:pointer}.row:hover{background:var(--skyl)}.row.on{background:var(--skyl);border-left:4px solid var(--sky)}
.row b{font-size:14px;color:var(--ink)}.row small{display:block;color:#5b6f8f;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-left:6px}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;background:var(--skyl);color:var(--ink2);margin-left:6px}
.chat{display:flex;flex-direction:column;min-width:0}.head{background:#fff;padding:14px 20px;border-bottom:1px solid var(--line)}
.head b{color:var(--ink);font-size:16px}.head div{font-size:12px;color:#5b6f8f;margin-top:3px}
.feed{flex:1;overflow-y:auto;padding:18px 22px}.msg{max-width:68%;margin:7px 0;padding:9px 13px;border-radius:12px;font-size:14px;line-height:1.45;white-space:pre-wrap;word-wrap:break-word}
.msg.in{background:#fff;border:1px solid var(--line)}.msg.out{background:var(--ink);color:#fff;margin-left:auto}
.msg i{display:block;font-size:10.5px;opacity:.65;font-style:normal;margin-bottom:3px}
.empty{color:#7a8ca8;text-align:center;margin-top:60px;font-size:14px}
@media(max-width:900px){.app{grid-template-columns:1fr;height:auto}.list{max-height:40vh}.chat{height:60vh}nav{flex-direction:row;flex-wrap:wrap}nav h1{width:100%}}
</style></head><body><div class="app"><nav id="nav"></nav><div class="list" id="list"></div>
<div class="chat"><div class="head" id="head"><b>Pick a chat</b></div><div class="feed" id="feed"><div class="empty">Select a number to read the conversation live.</div></div></div></div>
<script>
const VIEWS=[['live','Live Chats'],['interested','Interested Students'],['not_interested','Not Interested'],['history','History'],['followups','Follow-ups'],['forwarded','Forwarded to Team Head']];
let view='live',phone=null;
const esc=s=>{const d=document.createElement('div');d.textContent=s??'';return d.innerHTML};
function nav(){document.getElementById('nav').innerHTML='<h1>WhatsApp Agent <span>Dashboard</span></h1>'+
VIEWS.map(([k,l])=>`<button class="${k==view?'on':''}" onclick="setView('${k}')">${l}</button>`).join('')+'<div class="sp"></div><button onclick="logout()">Sign out</button>'}
async function api(u){const r=await fetch(u);if(r.status==401){location.reload();return null}return r.json()}
function setView(v){view=v;nav();refresh()}
function pick(p){phone=p;refresh()}
async function logout(){await fetch('/dashboard/logout',{method:'POST'});location.reload()}
async function refresh(){
 const d=await api('/dashboard/api/leads?view='+view);if(!d)return;
 const title=VIEWS.find(v=>v[0]==view)[1];
 document.getElementById('list').innerHTML=`<h2>${title} (${d.leads.length})</h2>`+(d.leads.map(l=>`<div class="row ${l.phone==phone?'on':''}" onclick="pick('${esc(l.phone)}')">
 <b>${esc(l.name||l.phone)}</b>${l.live?'<span class="dot"></span>':''}${l.name?`<span class="tag">${esc(l.phone)}</span>`:''}
 <small>${esc(l.last_text)}</small><small>${esc(l.course||'')}${l.city?' · '+esc(l.city):''} ${l.ago_min!=null?' · '+l.ago_min+' min ago':''}</small></div>`).join('')||'<div class="empty">Nothing here yet</div>');
 if(phone){const c=await api('/dashboard/api/chat?phone='+phone);if(!c)return;const L=c.lead||{};
  document.getElementById('head').innerHTML=`<b>${esc(L.name||phone)}</b><div>${esc(phone)} · ${esc(L.status||'')} ${L.city?'· '+esc(L.city):''} ${L.course?'· '+esc(L.course):''}</div>`;
  const f=document.getElementById('feed'),atEnd=f.scrollHeight-f.scrollTop-f.clientHeight<80;
  f.innerHTML=c.messages.map(m=>`<div class="msg ${m.direction}"><i>${m.direction=='in'?'Student':'Agent'} · ${m.type} · ${esc(m.time)}</i>${esc(m.text)}</div>`).join('');
  if(atEnd)f.scrollTop=f.scrollHeight}}
nav();refresh();setInterval(refresh,3000);
</script></body></html>"""


@app.get("/webhook")
def verify_webhook(request: Request):
    """Meta calls this once, when you save the webhook URL, to confirm you
    own this server. It must echo back the 'hub.challenge' value."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Meta calls this every time a user sends a WhatsApp message."""
    data = await request.json()
    try:
        entry = data["entry"][0]
        change = entry["changes"][0]["value"]
        messages = change.get("messages")

        if messages:
            message = messages[0]
            message_id = message.get("id")
            from_number = message["from"]  # sender's WhatsApp number
            msg_type = message.get("type")
            print(f"Incoming WhatsApp {msg_type} message from: {from_number}")

            if message_id in processed_message_ids:
                print(f"Duplicate delivery of message {message_id}, skipping.")
                return {"status": "duplicate, skipped"}
            processed_message_ids.add(message_id)

            user_text = None
            is_voice_message = False

            if msg_type == "text":
                user_text = message.get("text", {}).get("body", "")
            elif msg_type == "audio":
                is_voice_message = True
                media_id = message["audio"]["id"]
                try:
                    audio_bytes, mime_type = download_whatsapp_media(media_id)
                    user_text = transcribe_audio(audio_bytes, mime_type)
                    print(f"Transcribed voice message: {user_text}")
                except Exception as e:
                    print(f"Voice transcription failed after retries: {e}")
                    send_whatsapp_message(
                        from_number,
                        "Yeh voice note samajhne mein masla ho raha hai. "
                        "Dobara bhej kar dekh lein, ya text mein likh dein, "
                        f"ya seedha rabta karein: {TEAM_HEAD_NUMBER}",
                    )
                    user_text = None

            if user_text:
                log_message(
                    from_number, "in", "voice" if is_voice_message else "text", user_text
                )
                reply_text = rag_answer(user_text, user_id=from_number, voice=is_voice_message)
                log_message(
                    from_number, "out", "voice" if is_voice_message else "text", reply_text
                )
                record_forward_if_needed(from_number, reply_text)
                asyncio.create_task(refresh_lead(from_number))

                if is_voice_message or len(reply_text) > LONG_REPLY_CHARS:
                    # Voice in -> voice out; long answers also go as voice
                    # — feels like a real conversation
                    voice_sent = False
                    try:
                        audio, mime_type, filename = await synthesize_reply_audio(voice_version(reply_text))
                        voice_sent = send_whatsapp_voice(from_number, audio, mime_type, filename)
                    except Exception as e:
                        print(f"Voice reply failed: {e}")
                    if not voice_sent:
                        # Never leave the user without an answer
                        send_whatsapp_message(from_number, reply_text)
                    elif TEAM_HEAD_NUMBER in reply_text:
                        send_whatsapp_message(from_number, contact_card())
                else:
                    send_whatsapp_message(from_number, reply_text)
    except Exception as e:
        print("Error processing incoming WhatsApp message:", e)

    # Always return 200 quickly so Meta doesn't retry/resend the same message
    return {"status": "received"}


# Mount the Gradio chat UI at "/" for browser-based testing, alongside the
# WhatsApp webhook routes above.
# ---------- Google Sheet: automatic lead summary every 2 hours ----------
GOOGLE_SHEETS_CREDENTIALS_JSON = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON")
GOOGLE_SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
GOOGLE_SHEET_TAB_NAME = os.environ.get("GOOGLE_SHEET_TAB_NAME", "Leads")
SUMMARY_INTERVAL_SECONDS = 2 * 60 * 60  # 2 hours

_sheet = None
if GOOGLE_SHEETS_CREDENTIALS_JSON and GOOGLE_SHEET_ID:
    try:
        import json as _json

        import gspread
        from google.oauth2.service_account import Credentials as GoogleCredentials

        _creds_dict = _json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON)
        _scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        _gcreds = GoogleCredentials.from_service_account_info(_creds_dict, scopes=_scopes)
        _gc = gspread.authorize(_gcreds)
        _spreadsheet = _gc.open_by_key(GOOGLE_SHEET_ID)
        try:
            _sheet = _spreadsheet.worksheet(GOOGLE_SHEET_TAB_NAME)
        except gspread.WorksheetNotFound:
            _sheet = _spreadsheet.add_worksheet(GOOGLE_SHEET_TAB_NAME, rows=1000, cols=8)
            _sheet.append_row(
                ["Timestamp", "Phone Number", "Name", "City/Address", "Course Interested", "Status"]
            )
        print("Google Sheet connected for lead summaries.")
    except Exception as e:
        print(f"Google Sheet setup failed (summaries will be skipped): {e}")
else:
    print("GOOGLE_SHEETS_CREDENTIALS_JSON / GOOGLE_SHEET_ID not set — lead summary disabled.")


def extract_lead_info(phone, history):
    """Ask the AI to read a conversation and pull out structured lead info.
    Falls back to safe defaults if extraction fails for any reason."""
    convo_text = format_history(history)
    prompt = f"""Read this WhatsApp conversation between a potential student and
Leaders Academia's assistant. Extract the following as STRICT JSON only, no
other text:

{{
  "name": "the person's name if they mentioned it, else 'Not provided'",
  "address": "their city/address if mentioned, else 'Not provided'",
  "course_interested": "the specific course they showed interest in, else 'General inquiry'",
  "status": "one of: Interested, Not Interested, Follow-up Needed"
}}

CONVERSATION:
{convo_text}

Respond with ONLY the JSON object, nothing else."""

    try:
        import json as _json
        import re as _re

        raw = generate_text(prompt)
        match = _re.search(r"\{.*\}", raw, _re.DOTALL)
        data = _json.loads(match.group(0)) if match else {}
        return {
            "name": data.get("name", "Not provided"),
            "address": data.get("address", "Not provided"),
            "course_interested": data.get("course_interested", "General inquiry"),
            "status": data.get("status", "Follow-up Needed"),
        }
    except Exception as e:
        print(f"Lead extraction failed for {phone}: {e}")
        return {
            "name": "Not provided",
            "address": "Not provided",
            "course_interested": "General inquiry",
            "status": "Follow-up Needed",
        }


_last_summary_time = time.time()
_students_ws = None


def get_students_sheet():
    """Second tab: one live row per student (name, city, status, last message/reply)."""
    global _students_ws
    if _students_ws is not None:
        return _students_ws
    if not _sheet:
        return None
    try:
        _students_ws = _spreadsheet.worksheet("Students")
    except gspread.WorksheetNotFound:
        _students_ws = _spreadsheet.add_worksheet("Students", rows=2000, cols=8)
        _students_ws.append_row(["Updated", "Phone", "Name", "City", "Course", "Status",
                                 "Last Student Message", "Last Agent Reply"])
    return _students_ws


def upsert_student(phone):
    ws = get_students_sheet()
    if not ws:
        return
    lead = db("SELECT * FROM leads WHERE phone=?", (phone,))[0]
    last_in = db("SELECT text FROM messages WHERE phone=? AND direction='in' ORDER BY id DESC LIMIT 1", (phone,))
    last_out = db("SELECT text FROM messages WHERE phone=? AND direction='out' ORDER BY id DESC LIMIT 1", (phone,))
    row = [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), phone, lead["name"] or "Not provided",
           lead["city"] or "Not provided", lead["course"] or "General inquiry", lead["status"],
           (last_in[0]["text"] if last_in else "")[:300], (last_out[0]["text"] if last_out else "")[:300]]
    cell = ws.find(str(phone), in_column=2)
    if cell:
        ws.update(values=[row], range_name=f"A{cell.row}:H{cell.row}")
    else:
        ws.append_row(row)


def send_proactive(phone, name, text, template):
    """WhatsApp only allows free text within 24h of the student's last message.
    For 3-day / 2-day follow-ups an APPROVED TEMPLATE is required (set the env var)."""
    if template:
        url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        payload = {"messaging_product": "whatsapp", "to": phone, "type": "template",
                   "template": {"name": template, "language": {"code": TEMPLATE_LANG},
                                "components": [{"type": "body", "parameters": [
                                    {"type": "text", "text": name or "dear student"}]}]}}
        r = requests.post(url, headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
                                        "Content-Type": "application/json"}, json=payload, timeout=20)
        if r.status_code != 200:
            print("Template send failed:", r.status_code, r.text)
        return r.status_code == 200
    return send_whatsapp_message(phone, text).status_code == 200


async def run_periodic_lead_summary():
    """Every 2 hours: refresh every active student's row in the Students tab and add
    one summary line (how many numbers were contacted) to the Leads tab."""
    global _last_summary_time
    while True:
        await asyncio.sleep(SUMMARY_INTERVAL_SECONDS)
        try:
            cutoff = _last_summary_time
            _last_summary_time = time.time()
            phones = [r["phone"] for r in db(
                "SELECT phone FROM leads WHERE last_user_ts>? OR last_agent_ts>?", (cutoff, cutoff))]
            for phone in phones:
                await refresh_lead(phone)
                if _sheet:
                    await asyncio.to_thread(upsert_student, phone)
            if _sheet and phones:
                await asyncio.to_thread(_sheet.append_row, [
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "SUMMARY",
                    f"{len(phones)} number(s) chatted in the last 2 hours", "", "", ""])
        except Exception as e:
            print(f"Periodic lead summary failed: {e}")


async def run_followups():
    """Every 30 min: (1) interested student silent for 3 days -> greet + ask;
    (2) student forwarded to team head -> after 2 days ask if they spoke to him."""
    while True:
        await asyncio.sleep(1800)
        try:
            now = time.time()
            for r in db("SELECT phone,name,course FROM leads WHERE status='Interested' AND followup_sent=0"
                        " AND last_user_ts>0 AND ?-last_user_ts>?", (now, FOLLOWUP_AFTER_DAYS * 86400)):
                text = (f"Assalam o Alaikum {r['name'] or ''}! Aap ne {r['course'] or 'hamare courses'} mein "
                        "interest dikhaya tha. Koi sawal reh gaya ho to bata dein, main yahin hun.").replace("  ", " ")
                if await asyncio.to_thread(send_proactive, r["phone"], r["name"], text, FOLLOWUP_TEMPLATE):
                    db("UPDATE leads SET followup_sent=1 WHERE phone=?", (r["phone"],))
                    log_message(r["phone"], "out", "text", text)
            for r in db("SELECT phone,name FROM leads WHERE forwarded_at>0 AND forward_checked=0"
                        " AND ?-forwarded_at>?", (now, FORWARD_CHECK_AFTER_DAYS * 86400)):
                text = (f"Assalam o Alaikum {r['name'] or ''}! {TEAM_HEAD_NAME} se baat ho gayi? "
                        "Unhon ne kya kaha, aur aap ki enrollment ka kya plan bana?").replace("  ", " ")
                if await asyncio.to_thread(send_proactive, r["phone"], r["name"], text, FORWARD_CHECK_TEMPLATE):
                    db("UPDATE leads SET forward_checked=1 WHERE phone=?", (r["phone"],))
                    log_message(r["phone"], "out", "text", text)
        except Exception as e:
            print(f"Follow-up job failed: {e}")


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(run_periodic_lead_summary())
    asyncio.create_task(run_followups())


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
