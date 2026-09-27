"""
Leaders Academia — RAG Chatbot
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> Gradio chat UI.
"""

import os
import time

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
import gradio as gr
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.8-flash"

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
client = genai.Client(api_key=GEMINI_API_KEY)

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


# ---------- Gradio frontend ----------
def chat_fn(message, history):
    return rag_answer(message)


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


# ---------- FastAPI app: hosts the webhook AND the Gradio UI together ----------
app = FastAPI()


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
            from_number = message["from"]  # sender's WhatsApp number
            print(f"Incoming WhatsApp message from: {from_number}")  # debug line
            user_text = message.get("text", {}).get("body", "")

            if user_text:
                reply_text = rag_answer(user_text)
                send_whatsapp_message(from_number, reply_text)
    except Exception as e:
        print("Error processing incoming WhatsApp message:", e)

    # Always return 200 quickly so Meta doesn't retry/resend the same message
    return {"status": "received"}


# Mount the Gradio chat UI at "/" for browser-based testing, alongside the
# WhatsApp webhook routes above.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
