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
from google.genai import types
import gradio as gr
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.8-flash"

TEAM_HEAD_NUMBER = "0335-5229587"

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
            f"Zaroor! Aap hamari team head se seedha rabta kar sakte hain: "
            f"{TEAM_HEAD_NUMBER}"
        )

    context_chunks = retrieve_chunks(user_question, top_k=4)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are the official AI assistant for Leaders Academia, chatting with
someone on WhatsApp. You're warm, natural, and easy to talk to — like a real
person, not a scripted bot.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

GENERAL CHAT RULE: You're not limited to only Leaders Academia topics. If the
user makes small talk, asks a general question, or chats about something
unrelated, engage naturally and pleasantly — like any friendly, knowledgeable
person would. You don't need the CONTEXT for this.

LEADERS ACADEMIA RULE: For questions about courses, instructors, pricing,
schedules or the platform, answer using the CONTEXT below as if it's simply
what you know — speak naturally and confidently, the way a helpful human
staff member would. Never mention "context", "the information provided",
"based on the available data" or anything that reveals you're reading from a
document.

WHEN YOU DON'T KNOW: If a Leaders Academia question isn't answered by the
CONTEXT, don't guess or invent details. Instead say naturally, in your own
words, that you don't have that detail on hand and give them this number to
reach the team head directly: {TEAM_HEAD_NUMBER}

CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply clearly, briefly, and naturally — like a real person texting back, not
a formal report."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
            return response.text
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "503" in error_text or "quota" in error_text.lower() or "UNAVAILABLE" in error_text:
                if attempt < max_retries - 1:
                    time.sleep(15)  # back off and retry once on rate limit / busy server
                    continue
                return (
                    "Maaf kijiye, is waqt AI system busy hai (free usage "
                    f"limit lag gayi hai). Thori dair mein dobara try karein, "
                    f"ya seedha rabta karein: {TEAM_HEAD_NUMBER}"
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


def transcribe_audio(audio_bytes, mime_type, max_retries=3):
    """Send the voice note straight to Gemini and get back the spoken text."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    "Transcribe exactly what is said in this audio clip. Reply "
                    "with only the transcription, nothing else.",
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )
            return response.text.strip()
        except Exception as e:
            error_text = str(e)
            if "503" in error_text or "UNAVAILABLE" in error_text or "429" in error_text:
                if attempt < max_retries - 1:
                    print(f"Gemini busy during transcription, retrying... ({e})")
                    time.sleep(10)
                    continue
            raise


def text_to_speech(text):
    """Convert a text reply into an MP3 voice note using free gTTS."""
    from gtts import gTTS
    import io

    tts = gTTS(text=text, lang="ur")  # Urdu voice; works fine for Roman Urdu/English too
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    return buffer.read()


def send_whatsapp_voice(to_number, audio_bytes):
    """Upload an MP3 to WhatsApp's media store, then send it as a voice note."""
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

    upload_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    files = {"file": ("reply.mp3", audio_bytes, "audio/mpeg")}
    data = {"messaging_product": "whatsapp", "type": "audio/mpeg"}
    upload_response = requests.post(
        upload_url, headers=headers, files=files, data=data, timeout=30
    ).json()
    media_id = upload_response.get("id")
    if not media_id:
        print("Voice upload failed:", upload_response)
        return

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


# ---------- FastAPI app: hosts the webhook AND the Gradio UI together ----------
app = FastAPI()

# Track WhatsApp message IDs we've already replied to, so Meta's automatic
# retries (when our server is slow to respond) don't trigger duplicate replies.
processed_message_ids = set()


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
                        "Maaf kijiye, is waqt AI system busy hai. Thori dair "
                        "mein dobara voice note bhej kar try karein, ya text "
                        f"mein likh dein, ya seedha rabta karein: {TEAM_HEAD_NUMBER}",
                    )
                    user_text = None

            if user_text:
                reply_text = rag_answer(user_text)

                if is_voice_message:
                    # Voice in, voice out — feels like a real conversation
                    reply_audio = text_to_speech(reply_text)
                    send_whatsapp_voice(from_number, reply_audio)
                else:
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
