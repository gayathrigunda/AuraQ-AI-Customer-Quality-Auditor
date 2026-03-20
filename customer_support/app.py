import os
import csv
import asyncio
import httpx
import cloudinary
import cloudinary.uploader
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from contextlib import asynccontextmanager

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from deepgram import DeepgramClient, PrerecordedOptions
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
TRANSCRIPT_FILE = "transcriptions_with_speakers.csv"
SUMMARY_FILE    = "final_summaries.csv"
STATUS_FILE     = "processing_status.json"

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("Missing required env var: DEEPGRAM_API_KEY")

# Cloudinary credentials — add these 3 vars to Render's Environment tab
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY_   = os.environ.get("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET")

if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY_, CLOUDINARY_API_SECRET]):
    raise RuntimeError(
        "Missing Cloudinary env vars: CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
    )

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY_,
    api_secret=CLOUDINARY_API_SECRET,
)

dg_client  = DeepgramClient(DEEPGRAM_API_KEY)
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

# ---------------- KEEP-ALIVE PING ----------------

async def keep_alive():
    if not RENDER_URL:
        print("INFO: RENDER_EXTERNAL_URL not set - keep-alive ping disabled.")
        return
    await asyncio.sleep(60)
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(f"{RENDER_URL}/health", timeout=10)
                print(f"Keep-alive ping -> {r.status_code}")
            except Exception as e:
                print(f"Keep-alive ping failed: {e}")
            await asyncio.sleep(600)

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(keep_alive())
    yield
    task.cancel()

# ---------------- APP ----------------

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTILS ----------------

def format_for_ui(dg_raw_data):
    if not dg_raw_data:
        return []
    first_msg = dg_raw_data[0]['text'].lower()
    agent_label = "Speaker 1" if any(
        w in first_msg for w in ["help", "issue", "problem", "broken"]
    ) else "Speaker 0"
    return [
        {
            "speaker": "Speaker 00" if d['speaker'] == agent_label else "Speaker 01",
            "text": d['text'],
            "start": d['start']
        }
        for d in dg_raw_data
    ]

def set_status(status: str, detail: str = ""):
    import json
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": status, "detail": detail,
                   "updated_at": datetime.now().isoformat()}, f)

# ---------------- BACKGROUND TASK ----------------

async def process_audio_background(audio_data: bytes, filename: str):
    """
    Fix for Render 408 timeout:
      1. Upload audio -> Cloudinary (fast, same datacenter egress)
      2. Give Deepgram the HTTPS URL  (Deepgram fetches directly, not through Render)
      3. Delete the Cloudinary file after transcription
    """
    cloudinary_public_id = None
    try:
        # Step 1: Upload to Cloudinary
        set_status("processing", f"Uploading {filename} to storage...")
        print(f"DEBUG: Uploading {filename} to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            audio_data,
            resource_type="video",   # Cloudinary treats audio as "video" type
            public_id=f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            overwrite=True,
        )
        audio_url            = upload_result["secure_url"]
        cloudinary_public_id = upload_result["public_id"]
        print(f"DEBUG: Cloudinary URL -> {audio_url}")

        # Step 2: Pass URL to Deepgram (no buffer = no timeout)
        set_status("processing", f"Transcribing {filename}...")
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=True,
            summarize="v2",
            punctuate=True,
        )
        response = await dg_client.listen.asyncprerecorded.v("1").transcribe_url(
            {"url": audio_url}, options
        )

        # Step 3: Parse Deepgram response
        deepgram_summary = "No summary available."
        if hasattr(response.results, 'summary'):
            deepgram_summary = response.results.summary.short

        words = response.results.channels[0].alternatives[0].words
        if not words:
            set_status("error", "Empty audio content.")
            return

        dg_raw     = []
        curr_spk   = words[0].speaker
        curr_start = words[0].start
        curr_txt   = []

        for w in words:
            if w.speaker == curr_spk:
                curr_txt.append(w.word)
            else:
                dg_raw.append({"speaker": f"Speaker {curr_spk}",
                               "text": " ".join(curr_txt), "start": curr_start})
                curr_spk, curr_start, curr_txt = w.speaker, w.start, [w.word]

        dg_raw.append({"speaker": f"Speaker {curr_spk}",
                       "text": " ".join(curr_txt), "start": curr_start})

        # Step 4: Save to CSV
        pd.DataFrame(format_for_ui(dg_raw)).to_csv(TRANSCRIPT_FILE, index=False)

        file_exists = os.path.isfile(SUMMARY_FILE)
        with open(SUMMARY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "timestamp", "summary"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "file_name": filename,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "summary":   deepgram_summary,
            })

        set_status("done", deepgram_summary)
        print(f"DEBUG: Done processing {filename}")

    except Exception as e:
        print(f"Background processing error: {e}")
        set_status("error", str(e))

    finally:
        # Step 5: Delete temp Cloudinary file to keep storage free
        if cloudinary_public_id:
            try:
                cloudinary.uploader.destroy(cloudinary_public_id, resource_type="video")
                print(f"DEBUG: Deleted Cloudinary file {cloudinary_public_id}")
            except Exception as ce:
                print(f"Cloudinary cleanup warning (non-fatal): {ce}")

# ---------------- API ENDPOINTS ----------------

@app.post("/upload")
async def process_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        audio_data = await file.read()
        set_status("queued", f"File '{file.filename}' received.")
        background_tasks.add_task(process_audio_background, audio_data, file.filename)
        return {"status": "queued",
                "message": f"'{file.filename}' is being processed. Poll /status for updates."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
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
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    except Exception as e:
        print(f"Transcript fetch error: {e}")
        return []

@app.get("/get-summary")
async def get_summary():
    if not os.path.exists(SUMMARY_FILE):
        return {"summary": "No summary available."}
    try:
        df = pd.read_csv(SUMMARY_FILE)
        return {"summary": df["summary"].iloc[-1]} if not df.empty else {"summary": "No data."}
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