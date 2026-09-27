"""
Leaders Academia — RAG Chatbot + WhatsApp Webhook
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> served three ways from ONE app:
  1) Gradio chat UI (browser testing)        -> GET  /
  2) WhatsApp Cloud API webhook (real chats) -> GET/POST /webhook
  3) Privacy Policy page (for Meta App Publish requirement) -> GET /privacy
"""

import os
import time

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, HTMLResponse
import uvicorn

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
import gradio as gr

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo

# NOTE: gemini-2.5-flash is not available to new accounts (Google's own API
# confirmed this). gemini-3.8-flash is the required model, but its free tier
# is only ~20 requests/day — expect "busy" replies after ~20 messages/day
# until billing is enabled on the Google AI Studio project for higher quota.
GEMINI_MODEL_NAME = "gemini-3.8-flash"

# Team head's contact number — given out when the bot can't answer something
TEAM_HEAD_NUMBER = "0335-5229587"

HUMAN_HANDOFF_KEYWORDS = [
    "human", "real person", "agent", "representative",
    "insan se baat", "banda se baat", "customer support",
]

# ---------- Load secrets from Railway's environment variables (never hardcode them) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it under this project's "
        "Variables tab in the Railway dashboard."
    )
client = genai.Client(api_key=GEMINI_API_KEY)

# WhatsApp Cloud API credentials — these match the variable names already set
# in Railway (Variables tab): WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID,
# WHATSAPP_VERIFY_TOKEN
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")       # "Access token" from Meta
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")   # "Phone Number ID" from Meta
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "leadersacademia2026")


# ---------- Build the knowledge base once, when the app starts ----------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=800, overlap=150):
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


# ---------- RAG logic ----------
def retrieve_chunks(query, top_k=4):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


def wants_human(text):
    text = text.lower()
    return any(keyword in text for keyword in HUMAN_HANDOFF_KEYWORDS)


def rag_answer(user_question, max_retries=3):
    # Direct handoff — skip the model entirely for this case
    if wants_human(user_question):
        return (
            "Zaroor! Aap hamari team head se seedha rabta kar sakte hain:\n"
            f"{TEAM_HEAD_NUMBER}"
        )

    context_chunks = retrieve_chunks(user_question, top_k=4)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are the official AI assistant for Leaders Academia. You are also a
warm, helpful, general-purpose conversational assistant — not limited to only
Leaders Academia topics.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

CONVERSATION RULE: For greetings, small talk, casual questions, or anything
NOT specifically about Leaders Academia (general knowledge, advice, everyday
chit-chat, etc.), reply naturally, warmly, and helpfully like a normal
friendly AI assistant would — do not refuse or restrict yourself to only
Leaders Academia subjects.

FACTUAL RULE (Leaders Academia specific topics only): For questions about
Leaders Academia's courses, instructors, pricing, schedules, admissions, or
the platform itself, answer ONLY using the CONTEXT below. If the specific
answer is not in the CONTEXT, say politely that you don't have that exact
detail right now, and offer to connect them with the team head for help:
"Iske liye behtar hoga aap hamari team head se rabta karein: {TEAM_HEAD_NUMBER}"
Never guess or invent Leaders Academia facts that aren't in the CONTEXT.

CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply clearly and concisely."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
            return response.text
        except Exception as e:
            error_text = str(e)
            print(f"Gemini error (attempt {attempt + 1}/{max_retries}): {error_text}")
            if "429" in error_text or "quota" in error_text.lower():
                if attempt < max_retries - 1:
                    time.sleep(15)  # back off and retry once on rate limit
                    continue
                return (
                    "Maaf kijiye, is waqt AI system busy hai (free usage "
                    "limit lag gayi hai). Thori dair mein dobara try karein, "
                    f"ya seedha rabta karein: {TEAM_HEAD_NUMBER}"
                )
            return (
                "Kuch masla aa gaya jawab generate karte waqt, dobara koshish karein. "
                f"Ya seedha rabta karein: {TEAM_HEAD_NUMBER}"
            )


# ---------- Gradio frontend (browser demo) ----------
def chat_fn(message, history):
    return rag_answer(message)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Leaders Academia — AI Assistant",
    description="Courses, instructors aur platform ke baare mein kuch bhi puchein — ya bas baat karein!",
)


# ======================================================================
# FastAPI app — hosts the WhatsApp webhook, privacy policy, and mounts Gradio
# ======================================================================
app = FastAPI()


PRIVACY_POLICY_HTML = """
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Privacy Policy — Leaders Academia AI Assistant</title></head>
<body style="font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; line-height: 1.6; color:#222;">
  <h1>Privacy Policy — Leaders Academia AI Assistant</h1>
  <p>This AI Assistant ("the Assistant") is operated by Leaders Academia to answer
  questions about our courses, instructors, pricing, and platform via our website
  chat widget and WhatsApp.</p>

  <h3>What we collect</h3>
  <p>When you message the Assistant (via our website or WhatsApp), we process the
  text of your message and your WhatsApp phone number solely to generate a
  relevant response and, where you request it, connect you with a human team
  member.</p>

  <h3>How we use it</h3>
  <p>Message content is sent to our AI language model provider (Google Gemini) to
  generate a reply. We do not sell or share your personal data with third
  parties for advertising purposes.</p>

  <h3>Data retention</h3>
  <p>Conversation data is retained only as needed to operate and improve the
  Assistant and is not used for any purpose beyond answering your questions and
  providing support.</p>

  <h3>Contact us</h3>
  <p>For any privacy questions or data deletion requests, contact us at:<br>
  Email: info@leadersacademia.com<br>
  Phone: (051) 9876543<br>
  WhatsApp: +92 311 1534344</p>

  <p style="color:#777; font-size: 0.9em;">Last updated: 2026</p>
</body>
</html>
"""


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    """Public privacy policy page — used as the 'Privacy Policy URL' when
    publishing the Meta app."""
    return PRIVACY_POLICY_HTML


def send_whatsapp_message(to_number, message_text):
    """Sends a plain-text reply back to a WhatsApp user via Meta's Cloud API."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("WhatsApp not configured (missing WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID) — skipping send.")
        return

    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        if resp.status_code >= 400:
            print("WhatsApp send failed:", resp.status_code, resp.text)
    except Exception as e:
        print("WhatsApp send error:", e)


@app.get("/webhook")
def verify_webhook(request: Request):
    """Meta calls this ONCE, when you click 'Verify and save' in the dashboard."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="Verification failed", status_code=403)


@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Meta calls this every time a WhatsApp user sends your number a message."""
    body = await request.json()
    try:
        entry = body["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        messages = value.get("messages")

        if messages:
            message = messages[0]
            sender_number = message["from"]
            user_text = message.get("text", {}).get("body", "")

            if user_text:
                reply = rag_answer(user_text)
                send_whatsapp_message(sender_number, reply)
    except Exception as e:
        print("Webhook processing note:", e)

    return {"status": "ok"}


# Mount the Gradio chat UI at the root path ("/") so the browser demo still works
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
