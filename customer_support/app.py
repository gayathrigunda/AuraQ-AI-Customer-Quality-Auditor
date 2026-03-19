import os
import csv
import asyncio
import httpx
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")  # goes up to root

# --- DEEPGRAM V3.x IMPORTS ---
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
TRANSCRIPT_FILE = "transcriptions_with_speakers.csv"
SUMMARY_FILE = "final_summaries.csv"
STATUS_FILE = "processing_status.json"  # NEW: tracks background job status

# Load from environment so the key is not stored in source control
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("Missing required env var: DEEPGRAM_API_KEY")

# ---------------- KEEP-ALIVE PING ----------------
# Render free tier spins down after 15 min of inactivity.
# This pings /health every 10 minutes to keep the service awake.
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")  # set this in Render env vars

async def keep_alive():
    """Ping own /health endpoint every 10 minutes to prevent Render sleep."""
    if not RENDER_URL:
        print("INFO: RENDER_EXTERNAL_URL not set — keep-alive ping disabled.")
        return
    await asyncio.sleep(60)  # wait 1 min after startup before first ping
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(f"{RENDER_URL}/health", timeout=10)
                print(f"Keep-alive ping → {r.status_code}")
            except Exception as e:
                print(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(600)  # ping every 10 minutes

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Start keep-alive background task on server startup
    task = asyncio.create_task(keep_alive())
    yield
    task.cancel()  # clean shutdown

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deepgram Client
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

# ---------------- UTILS ----------------

def format_for_ui(dg_raw_data):
    """
    Programmatically identifies roles to save time.
    UI expects 'Speaker 00' for Agent and 'Speaker 01' for Customer.
    """
    if not dg_raw_data:
        return []

    first_msg = dg_raw_data[0]['text'].lower()
    
    if any(word in first_msg for word in ["help", "issue", "problem", "broken"]):
        agent_label = "Speaker 1"
    else:
        agent_label = "Speaker 0"

    refined_transcript = []
    for d in dg_raw_data:
        is_agent = d['speaker'] == agent_label
        refined_transcript.append({
            "speaker": "Speaker 00" if is_agent else "Speaker 01",
            "text": d['text'],
            "start": d['start']
        })
    return refined_transcript


def set_status(status: str, detail: str = ""):
    """Write processing status to a JSON file so the frontend can poll it."""
    import json
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": status, "detail": detail, "updated_at": datetime.now().isoformat()}, f)


def process_audio_background(audio_data: bytes, filename: str):
    """
    Runs the heavy Deepgram call in the background so the HTTP request
    returns immediately — avoiding Render's 30-second timeout.
    """
    try:
        set_status("processing", f"Transcribing {filename}...")

        payload: FileSource = {"buffer": audio_data}

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=True,
            summarize="v2",
            punctuate=True,
        )

        print(f"DEBUG: Processing {filename}...")

        response = dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)

        # 1. Get the Instant Summary
        deepgram_summary = "No summary available."
        if hasattr(response.results, 'summary'):
            deepgram_summary = response.results.summary.short

        # 2. Extract Words and Group by Speaker
        words = response.results.channels[0].alternatives[0].words
        if not words:
            set_status("error", "Empty audio content.")
            return

        dg_raw = []
        curr_spk = words[0].speaker
        curr_start = words[0].start
        curr_txt = []

        for w in words:
            if w.speaker == curr_spk:
                curr_txt.append(w.word)
            else:
                dg_raw.append({
                    "speaker": f"Speaker {curr_spk}",
                    "text": " ".join(curr_txt),
                    "start": curr_start
                })
                curr_spk = w.speaker
                curr_start = w.start
                curr_txt = [w.word]

        dg_raw.append({
            "speaker": f"Speaker {curr_spk}",
            "text": " ".join(curr_txt),
            "start": curr_start
        })

        # 3. Fast Formatting
        refined_data = format_for_ui(dg_raw)

        # 4. Save to CSV
        df = pd.DataFrame(refined_data)
        df.to_csv(TRANSCRIPT_FILE, index=False)

        # 5. Update History
        file_exists = os.path.isfile(SUMMARY_FILE)
        with open(SUMMARY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "timestamp", "summary"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "file_name": filename,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "summary": deepgram_summary,
            })

        # 6. Mark as done — frontend polls /status to know when to fetch results
        set_status("done", deepgram_summary)
        print(f"DEBUG: Done processing {filename}")

    except Exception as e:
        print(f"Background processing error: {e}")
        set_status("error", str(e))


# ---------------- API ENDPOINTS ----------------

@app.post("/upload")
async def process_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Reads the file immediately, then hands off the heavy Deepgram work to a
    background task. Returns 202 Accepted instantly — no more timeout on Render.
    """
    try:
        audio_data = await file.read()  # read BEFORE the background task (stream closes after response)
        filename = file.filename

        set_status("queued", f"File '{filename}' received, queued for processing.")
        background_tasks.add_task(process_audio_background, audio_data, filename)

        return {
            "status": "queued",
            "message": f"'{filename}' is being processed. Poll /status for updates."
        }

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """
    Frontend polls this endpoint after upload to know when processing is complete.
    Returns: { status: 'queued' | 'processing' | 'done' | 'error', detail: str }
    """
    import json
    if not os.path.exists(STATUS_FILE):
        return {"status": "idle", "detail": "No job running."}
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except:
        return {"status": "unknown", "detail": "Could not read status."}


@app.get("/get-transcript")
async def get_transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        return []
    try:
        df = pd.read_csv(TRANSCRIPT_FILE)
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Transcript fetch error: {e}")
        return []


@app.get("/get-summary")
async def get_summary():
    if not os.path.exists(SUMMARY_FILE):
        return {"summary": "No summary available."}
    try:
        df = pd.read_csv(SUMMARY_FILE)
        if df.empty:
            return {"summary": "No data."}
        return {"summary": df["summary"].iloc[-1]}
    except:
        return {"summary": "No summary available."}


@app.get("/history")
async def get_history():
    if not os.path.exists(SUMMARY_FILE):
        return []
    try:
        df = pd.read_csv(SUMMARY_FILE)
        return df.tail(10).iloc[::-1].to_dict(orient="records")
    except:
        return []


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))