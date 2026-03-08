import os
import shutil
import gc
import csv
import pandas as pd
from datetime import datetime
from groq import Groq
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisperx
from whisperx.diarize import DiarizationPipeline

# --- CONFIGURATION ---
TRANSCRIPT_FILE = "transcriptions_with_speakers.csv"
SUMMARY_FILE = "final_summaries.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEVICE = "cpu"  # Use "cuda" if you have GPU
MODEL_SIZE = "medium"  # "medium" or "large" for better accuracy
COMPUTE_TYPE = "int8"

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# Initialize WhisperX model
print(f"🚀 Loading WhisperX ({MODEL_SIZE})...")
model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)

# Initialize Pyannote diarization for 2 speakers
try:
    HF_TOKEN = os.getenv("HF_TOKEN")
    diarize_model = DiarizationPipeline(
        token=HF_TOKEN,
        device=DEVICE,
        model_name="pyannote/speaker-diarization-3.1",
        num_speakers=2  # Force two speakers
    )
except Exception as e:
    print(f"⚠️ Diarization unavailable: {e}")
    diarize_model = None

# History for sidebar
analysis_history = []

# --- FUNCTION TO GENERATE AI SUMMARY ---
def generate_ai_summary(text):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional call logger. Summarize the conversation in EXACTLY one sentence "
                        "using this format: [Name] called to [Action] from [Business], resulting in [Outcome].\n"
                        "Example: Brando Thomas called to order a dozen long-stem red roses from Martha's Flores, "
                        "resulting in a successful transaction and shipment confirmation within 24 hours."
                    )
                },
                {"role": "user", "content": f"Transcript: {text[:4000]}"}
            ],
            temperature=0.1,
            max_tokens=100
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return "Summary currently unavailable due to API limits."

# --- UPLOAD ENDPOINT ---
@app.post("/upload")
async def process_upload(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    file_extension = file.filename.split('.')[-1].lower()

    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        full_text_for_summary = ""
        formatted_data = []

        # --- TEXT FILES (.txt or .csv) ---
        if file_extension in ['txt', 'csv']:
            speaker_map = {}
            speakers_detected = 0
            with open(temp_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    if ":" in line:
                        raw_speaker, text_content = line.split(":", 1)
                        raw_speaker = raw_speaker.strip()
                        text_content = text_content.strip()

                        # Track only 2 speakers
                        if raw_speaker not in speaker_map and speakers_detected < 2:
                            if speakers_detected == 0:
                                speaker_map[raw_speaker] = "Customer"
                            else:
                                speaker_map[raw_speaker] = "Agent"
                            speakers_detected += 1

                        current_speaker = speaker_map.get(raw_speaker, "UNKNOWN")
                    else:
                        text_content = line
                        current_speaker = "UNKNOWN"

                    full_text_for_summary += f" {text_content}"
                    formatted_data.append({
                        "speaker": current_speaker,
                        "text": text_content,
                        "start": i,
                        "end": i + 1
                    })

            pd.DataFrame(formatted_data).to_csv(TRANSCRIPT_FILE, index=False)

        # --- AUDIO FILES ---
        else:
            audio = whisperx.load_audio(temp_file)
            result = model.transcribe(audio, batch_size=4)  # Increase batch_size if GPU

            # Alignment
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE)

            # Diarization
            if diarize_model:
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            for seg in result["segments"]:
                text_content = seg.get("text", "").strip()
                speaker_raw = seg.get("speaker", "UNKNOWN")

                # Map speakers to Customer / Agent
                if speaker_raw == "SPEAKER_0":
                    current_speaker = "Customer"
                elif speaker_raw == "SPEAKER_1":
                    current_speaker = "Agent"
                else:
                    current_speaker = "UNKNOWN"

                full_text_for_summary += f" {text_content}"
                formatted_data.append({
                    "speaker": current_speaker,
                    "text": text_content,
                    "start": round(seg.get("start", 0), 2),
                    "end": round(seg.get("end", 0), 2)
                })

            pd.DataFrame(formatted_data).to_csv(TRANSCRIPT_FILE, index=False)

        # --- GENERATE SUMMARY ---
        summary_text = generate_ai_summary(full_text_for_summary)

        file_exists = os.path.isfile(SUMMARY_FILE)
        with open(SUMMARY_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "text", "summary"])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "file_name": file.filename,
                "text": full_text_for_summary[:500],
                "summary": summary_text
            })

        # Update history
        analysis_history.insert(0, {
            "id": len(analysis_history) + 1,
            "name": file.filename,
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "status": "Ready"
        })

        return {"status": "success", "summary": summary_text}

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        gc.collect()

# --- GET LATEST SUMMARY ---
@app.get("/get-summary")
async def get_summary():
    if not os.path.exists(SUMMARY_FILE):
        return {"summary": "No summary available."}
    try:
        df = pd.read_csv(SUMMARY_FILE)
        if not df.empty:
            return {"summary": df["summary"].iloc[-1]}
    except:
        pass
    return {"summary": "Error reading summary file."}

# --- GET FULL TRANSCRIPT ---
@app.get("/get-transcript")
async def get_transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        return []
    return pd.read_csv(TRANSCRIPT_FILE).to_dict(orient="records")

# --- GET HISTORY ---
@app.get("/history")
async def get_history():
    return analysis_history

# --- RUN APP ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)