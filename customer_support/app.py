import os
import csv
import asyncio
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# --- DEEPGRAM V3.x IMPORTS ---
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# ---------------- CONFIG ----------------
TRANSCRIPT_FILE = "transcriptions_with_speakers.csv"
SUMMARY_FILE = "final_summaries.csv"
TRANSCRIPTS_DIR = "batch_transcripts"   # folder for per-file batch transcripts

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError("Missing required env var: DEEPGRAM_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Deepgram Client
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

# Ensure batch transcript directory exists
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


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


def safe_filename(name: str) -> str:
    """Strip unsafe characters from a filename for use as a path component."""
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


# ---------------- CORE TRANSCRIPTION (shared by single + batch) ----------------

async def transcribe_one(filename: str, audio_data: bytes) -> dict:
    """
    Transcribe a single audio file using Deepgram and return a result dict.
    Runs Deepgram call in a thread so it doesn't block the event loop.
    """
    payload: FileSource = {"buffer": audio_data}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        diarize=True,
        summarize="v2",
        punctuate=True,
    )

    print(f"[TRANSCRIBE] Starting: {filename}")

    # Run blocking Deepgram SDK call in a thread pool
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)
    )

    # 1. Extract summary
    deepgram_summary = "No summary available."
    if hasattr(response.results, 'summary'):
        deepgram_summary = response.results.summary.short

    # 2. Extract words and group by speaker
    words = response.results.channels[0].alternatives[0].words
    if not words:
        raise ValueError(f"Empty audio content in file: {filename}")

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

    # 3. Format for UI
    refined_data = format_for_ui(dg_raw)

    print(f"[TRANSCRIBE] Done: {filename} ({len(refined_data)} turns)")
    return {
        "filename": filename,
        "summary": deepgram_summary,
        "transcript": refined_data,
    }


# ---------------- API ENDPOINTS ----------------

@app.post("/upload")
async def process_upload(file: UploadFile = File(...)):
    """Single file upload — original behaviour preserved."""
    try:
        audio_data = await file.read()
        result = await transcribe_one(file.filename, audio_data)

        # Save transcript CSV (overwrites — single-file mode)
        df = pd.DataFrame(result["transcript"])
        df.to_csv(TRANSCRIPT_FILE, index=False)

        # Append to history
        file_exists = os.path.isfile(SUMMARY_FILE)
        with open(SUMMARY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "timestamp", "summary"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "file_name": file.filename,
                "timestamp": datetime.now().strftime("%I:%M %p"),
                "summary": result["summary"],
            })

        return {"status": "success", "summary": result["summary"]}

    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-batch")
async def process_batch_upload(files: List[UploadFile] = File(...)):
    """
    Batch transcription endpoint.

    Accepts multiple audio files, transcribes them concurrently using asyncio,
    saves per-file transcripts to ./batch_transcripts/<filename>.csv,
    appends all summaries to the shared history CSV, and also overwrites the
    shared TRANSCRIPT_FILE with the LAST successfully transcribed file so the
    existing single-file UI still works immediately after a batch run.

    Returns a per-file results list so the frontend can show a status table.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Read all file bytes up-front (UploadFile streams must be consumed in the request)
    file_payloads = []
    for f in files:
        data = await f.read()
        file_payloads.append((f.filename, data))
        print(f"[BATCH] Queued: {f.filename} ({len(data)} bytes)")

    # ── Transcribe all files concurrently ──────────────────────────────────────
    tasks = [transcribe_one(name, data) for name, data in file_payloads]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # ── Process results ────────────────────────────────────────────────────────
    batch_results = []
    last_successful_transcript = None

    file_exists = os.path.isfile(SUMMARY_FILE)
    summary_rows = []

    for (filename, _), outcome in zip(file_payloads, raw_results):
        if isinstance(outcome, Exception):
            print(f"[BATCH] FAILED: {filename} — {outcome}")
            batch_results.append({
                "file_name": filename,
                "status": "error",
                "summary": f"Failed: {str(outcome)}",
                "turns": 0,
            })
            continue

        # Save per-file transcript CSV
        safe_name = safe_filename(filename)
        per_file_csv = os.path.join(TRANSCRIPTS_DIR, f"{safe_name}.csv")
        df = pd.DataFrame(outcome["transcript"])
        df.to_csv(per_file_csv, index=False)

        # Track last successful transcript for shared CSV compatibility
        last_successful_transcript = outcome["transcript"]

        summary_rows.append({
            "file_name": filename,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "summary": outcome["summary"],
        })

        batch_results.append({
            "file_name": filename,
            "status": "success",
            "summary": outcome["summary"],
            "turns": len(outcome["transcript"]),
            "transcript_csv": per_file_csv,
        })

        print(f"[BATCH] Saved: {per_file_csv}")

    # ── Write all summaries to history in one pass ─────────────────────────────
    if summary_rows:
        with open(SUMMARY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "timestamp", "summary"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(summary_rows)

    # ── Overwrite shared transcript file with last successful result ───────────
    # This keeps backward compatibility so /get-transcript still returns something useful.
    if last_successful_transcript:
        pd.DataFrame(last_successful_transcript).to_csv(TRANSCRIPT_FILE, index=False)

    success_count = sum(1 for r in batch_results if r["status"] == "success")
    print(f"[BATCH] Complete: {success_count}/{len(files)} succeeded.")

    return {
        "status": "complete",
        "total": len(files),
        "succeeded": success_count,
        "failed": len(files) - success_count,
        "results": batch_results,
    }


@app.get("/batch-transcripts")
async def list_batch_transcripts():
    """
    Lists all per-file CSVs saved by /upload-batch.
    Returns filename, row count, and the transcript path for each file.
    """
    if not os.path.isdir(TRANSCRIPTS_DIR):
        return []

    entries = []
    for fname in sorted(os.listdir(TRANSCRIPTS_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(TRANSCRIPTS_DIR, fname)
        try:
            df = pd.read_csv(path)
            entries.append({
                "csv_file": fname,
                "turns": len(df),
                "path": path,
            })
        except Exception:
            entries.append({"csv_file": fname, "turns": 0, "path": path})

    return entries


@app.get("/batch-transcript/{csv_filename}")
async def get_batch_transcript(csv_filename: str):
    """
    Returns the transcript rows for a specific batch-processed file.
    csv_filename is the sanitised filename returned by /batch-transcripts.
    """
    # Prevent path traversal
    safe = safe_filename(csv_filename)
    path = os.path.join(TRANSCRIPTS_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Transcript not found: {csv_filename}")
    try:
        df = pd.read_csv(path)
        df = df.where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    uvicorn.run(app, host="127.0.0.1", port=8000)