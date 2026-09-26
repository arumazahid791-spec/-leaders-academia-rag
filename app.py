"""
Leaders Academia — RAG Chatbot + WhatsApp Webhook
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> served two ways from ONE app:
  1) Gradio chat UI (browser testing)        -> GET  /
  2) WhatsApp Cloud API webhook (real chats) -> GET/POST /webhook
"""

import os
import time

import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
import gradio as gr

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.8-flash"

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

# WhatsApp Cloud API credentials — add these three in Railway -> Variables
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")              # "Access token" from Meta
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")            # "Phone Number ID" from Meta
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


# ---------- RAG logic (unchanged from your original) ----------
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
            "Zaroor! Aap hamari team se seedha rabta kar sakte hain:\n"
            "Phone: (051) 9876543\n"
            "WhatsApp Support: +92 311 1534344"
        )

    context_chunks = retrieve_chunks(user_question, top_k=4)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are the official AI assistant for Leaders Academia.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

CONVERSATION RULE: For greetings and small talk (e.g. "how are you", "hi",
"thanks") reply naturally and warmly like a human — never say information is
unavailable for these.

FACTUAL RULE: For questions about courses, instructors, pricing, schedules or
the platform, answer ONLY using the CONTEXT below. If the answer is not in the
CONTEXT, say clearly that this information is not available — never guess.

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
            if "429" in error_text or "quota" in error_text.lower():
                if attempt < max_retries - 1:
                    time.sleep(15)  # back off and retry once on rate limit
                    continue
                return (
                    "Maaf kijiye, is waqt AI system busy hai (free usage "
                    "limit lag gayi hai). Thori dair mein dobara try karein, "
                    "ya seedha rabta karein: +92 311 1534344 (WhatsApp)."
                )
            return "Kuch masla aa gaya jawab generate karte waqt, dobara koshish karein."


# ---------- Gradio frontend (browser demo — unchanged) ----------
def chat_fn(message, history):
    return rag_answer(message)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Leaders Academia — AI Assistant",
    description="Courses, instructors aur platform ke baare mein kuch bhi puchein.",
)


# ======================================================================
# NEW: FastAPI app — hosts the WhatsApp webhook and mounts the Gradio UI
# ======================================================================
app = FastAPI()


def send_whatsapp_message(to_number, message_text):
    """Sends a plain-text reply back to a WhatsApp user via Meta's Cloud API."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        print("WhatsApp not configured (missing WHATSAPP_TOKEN / PHONE_NUMBER_ID) — skipping send.")
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
        # Meta also sends non-message events (delivery/read receipts) — ignore those quietly
        print("Webhook processing note:", e)

    # Always return 200 quickly, or Meta will mark the webhook as failing
    return {"status": "ok"}


# Mount the Gradio chat UI at the root path ("/") so the browser demo still works
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
