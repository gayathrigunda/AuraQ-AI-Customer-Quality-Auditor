import os, re, json, time, csv, shutil, gc, asyncio
import httpx
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from groq import Groq
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List

# ============================================================
# CONFIG
# ============================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTS_DIR  = os.path.join(BASE_DIR, "file_transcripts")
SUMMARIES_DIR    = os.path.join(BASE_DIR, "file_summaries")
SCORES_DIR       = os.path.join(BASE_DIR, "file_scores")
BATCH_DIR        = os.path.join(BASE_DIR, "batch_transcripts")

AUDIO_TRANSCRIPT_FILE = "/tmp/audio_transcript.csv"
TEXT_TRANSCRIPT_FILE  = "/tmp/text_transcript.csv"
AUDIO_SUMMARY_FILE    = "/tmp/audio_summaries.csv"
TEXT_SUMMARY_FILE     = "/tmp/text_summaries.csv"
SCORES_FILE           = os.path.join(BASE_DIR, "quality_scores.json")
SCORES_HISTORY_FILE   = os.path.join(BASE_DIR, "quality_scores_history.json")

for d in [TRANSCRIPTS_DIR, SUMMARIES_DIR, SCORES_DIR, BATCH_DIR]:
    os.makedirs(d, exist_ok=True)

def ist_now(fmt="%Y-%m-%d %H:%M:%S"):
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime(fmt)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "").strip()

if not DEEPGRAM_API_KEY:
    raise RuntimeError("Missing DEEPGRAM_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY")

dg_client = DeepgramClient(DEEPGRAM_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# UTILS — safe filename
# ============================================================
def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

# ============================================================
# AUDIO — transcription helpers
# ============================================================
def format_for_ui(dg_raw_data):
    if not dg_raw_data:
        return []
    first_msg = dg_raw_data[0]['text'].lower()
    agent_label = "Speaker 1" if any(
        w in first_msg for w in ["help", "issue", "problem", "broken"]
    ) else "Speaker 0"
    result = []
    for d in dg_raw_data:
        result.append({
            "speaker": "Speaker 00" if d['speaker'] == agent_label else "Speaker 01",
            "text":    d['text'],
            "start":   d['start'],
        })
    return result

async def transcribe_one(filename: str, audio_data: bytes) -> dict:
    payload: FileSource = {"buffer": audio_data}
    options = PrerecordedOptions(
        model="nova-2", smart_format=True,
        diarize=True, summarize="v2", punctuate=True,
    )
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)
    )
    summary = "No summary available."
    if hasattr(response.results, 'summary'):
        summary = response.results.summary.short

    words = response.results.channels[0].alternatives[0].words
    if not words:
        raise ValueError(f"Empty audio: {filename}")

    dg_raw, curr_spk = [], words[0].speaker
    curr_start, curr_txt = words[0].start, []
    for w in words:
        if w.speaker == curr_spk:
            curr_txt.append(w.word)
        else:
            dg_raw.append({"speaker": f"Speaker {curr_spk}",
                           "text": " ".join(curr_txt), "start": curr_start})
            curr_spk, curr_start, curr_txt = w.speaker, w.start, [w.word]
    dg_raw.append({"speaker": f"Speaker {curr_spk}",
                   "text": " ".join(curr_txt), "start": curr_start})

    return {"filename": filename, "summary": summary,
            "transcript": format_for_ui(dg_raw)}

# ============================================================
# AUDIO ENDPOINTS
# ============================================================
@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        data   = await file.read()
        result = await transcribe_one(file.filename, data)

        df = pd.DataFrame(result["transcript"])
        df.to_csv(AUDIO_TRANSCRIPT_FILE, index=False)

        # per-file transcript
        sn = safe_name(file.filename)
        df.to_csv(os.path.join(TRANSCRIPTS_DIR, f"{sn}.csv"), index=False)

        # history
        fe = os.path.isfile(AUDIO_SUMMARY_FILE)
        with open(AUDIO_SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name","timestamp","summary"])
            if not fe: w.writeheader()
            w.writerow({"file_name": file.filename,
                        "timestamp": ist_now(),
                        "summary":   result["summary"]})

        # per-file summary
        with open(os.path.join(SUMMARIES_DIR, f"{sn}.json"), "w") as f:
            json.dump({"filename": file.filename, "summary": result["summary"],
                       "saved_at": ist_now()}, f, indent=4)

        return {"status": "success", "summary": result["summary"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    payloads = [(f.filename, await f.read()) for f in files]
    tasks    = [transcribe_one(name, data) for name, data in payloads]
    results  = await asyncio.gather(*tasks, return_exceptions=True)

    batch_results, last_ok, summary_rows = [], None, []
    fe = os.path.isfile(AUDIO_SUMMARY_FILE)

    for (filename, _), outcome in zip(payloads, results):
        if isinstance(outcome, Exception):
            batch_results.append({"file_name": filename, "status": "error",
                                   "summary": str(outcome), "turns": 0})
            continue
        sn  = safe_name(filename)
        df  = pd.DataFrame(outcome["transcript"])
        df.to_csv(os.path.join(BATCH_DIR, f"{sn}.csv"), index=False)
        df.to_csv(os.path.join(TRANSCRIPTS_DIR, f"{sn}.csv"), index=False)
        last_ok = outcome["transcript"]
        summary_rows.append({"file_name": filename, "timestamp": ist_now(),
                              "summary": outcome["summary"]})
        batch_results.append({"file_name": filename, "status": "success",
                               "summary": outcome["summary"],
                               "turns": len(outcome["transcript"])})

    if summary_rows:
        with open(AUDIO_SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name","timestamp","summary"])
            if not fe: w.writeheader()
            w.writerows(summary_rows)

    if last_ok:
        pd.DataFrame(last_ok).to_csv(AUDIO_TRANSCRIPT_FILE, index=False)

    ok = sum(1 for r in batch_results if r["status"] == "success")
    return {"status": "complete", "total": len(files),
            "succeeded": ok, "failed": len(files) - ok, "results": batch_results}

@app.get("/get-transcript")
async def get_transcript():
    if not os.path.exists(AUDIO_TRANSCRIPT_FILE): return []
    try:
        df = pd.read_csv(AUDIO_TRANSCRIPT_FILE)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    except: return []

@app.get("/get-file-transcript/{filename:path}")
async def get_file_transcript(filename: str):
    from urllib.parse import unquote
    path = os.path.join(TRANSCRIPTS_DIR, f"{safe_name(unquote(filename))}.csv")
    if not os.path.exists(path): return []
    try:
        df = pd.read_csv(path)
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    except: return []

@app.get("/get-summary")
async def get_audio_summary():
    if not os.path.exists(AUDIO_SUMMARY_FILE):
        return {"summary": "No summary available."}
    try:
        df = pd.read_csv(AUDIO_SUMMARY_FILE)
        return {"summary": df["summary"].iloc[-1] if not df.empty else "No data."}
    except: return {"summary": "No summary available."}

@app.get("/get-file-summary/{filename:path}")
async def get_file_summary(filename: str):
    from urllib.parse import unquote
    path = os.path.join(SUMMARIES_DIR, f"{safe_name(unquote(filename))}.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {"summary": "No summary available."}

@app.get("/history")
async def get_audio_history():
    results = []
    for fpath, key in [(AUDIO_SUMMARY_FILE, "audio"), (TEXT_SUMMARY_FILE, "text")]:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath).fillna("")
                results.extend(df.to_dict(orient="records"))
            except: pass
    results.sort(key=lambda x: x.get("timestamp",""), reverse=True)
    return results

@app.post("/clear-history")
async def clear_history():
    try:
        for fpath in [AUDIO_SUMMARY_FILE, TEXT_SUMMARY_FILE,
                      AUDIO_TRANSCRIPT_FILE, TEXT_TRANSCRIPT_FILE]:
            if os.path.exists(fpath): os.remove(fpath)
        for folder in [SCORES_DIR, SUMMARIES_DIR, TRANSCRIPTS_DIR]:
            if os.path.exists(folder):
                shutil.rmtree(folder); os.makedirs(folder)
        return {"status": "cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================
# TEXT CHAT ENDPOINTS
# ============================================================
def parse_chat_to_turns(text):
    lines, turns = text.strip().split('\n'), []
    if len(lines) > 1:
        pat = re.compile(r'^([A-Za-z][A-Za-z0-9_ ]{0,30}?)\s*:\s*(.+)$')
        for line in lines:
            m = pat.match(line.strip())
            if m: turns.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
    if not turns:
        candidates = re.findall(r'\b([A-Za-z][A-Za-z0-9_ ]{0,30}?)\s*:', text)
        counts   = Counter(c.strip() for c in candidates)
        speakers = list(counts.keys())
        if speakers:
            escaped = [re.escape(s) for s in sorted(speakers, key=len, reverse=True)]
            parts   = re.split(r'(' + '|'.join(escaped) + r')\s*:', text)
            i = 1
            while i < len(parts) - 1:
                spk, msg = parts[i].strip(), parts[i+1].strip()
                if spk in speakers and msg:
                    turns.append({"speaker": spk, "text": msg})
                i += 2
    return turns

def format_chat_for_ui(turns):
    if not turns: return []
    first_msg    = turns[0]['text'].lower()
    all_speakers = list(dict.fromkeys(t['speaker'] for t in turns))
    agent_spk    = (all_speakers[1] if len(all_speakers) > 1 else all_speakers[0]) \
                   if any(w in first_msg for w in ["help","issue","problem","broken","error"]) \
                   else all_speakers[0]
    return [{"speaker": "Speaker 00" if t['speaker'].lower() == agent_spk.lower()
             else "Speaker 01", "text": t['text']} for t in turns]

def summarize_with_deepgram(text: str) -> str:
    try:
        resp = requests.post(
            "https://api.deepgram.com/v1/read?summarize=true&language=en",
            headers={"Authorization": f"Token {DEEPGRAM_API_KEY}",
                     "Content-Type": "application/json"},
            json={"text": text}, timeout=30)
        if resp.status_code != 200:
            return f"Summary failed: HTTP {resp.status_code}"
        return resp.json()["results"]["summary"]["text"]
    except Exception as e:
        return f"Summary failed: {str(e)}"

@app.post("/upload-text")
async def upload_text(file: UploadFile = File(...)):
    tmp = f"/tmp/upload_{file.filename}"
    try:
        with open(tmp, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        with open(tmp, "r", encoding="utf-8") as f:
            content = f.read()

        summary   = summarize_with_deepgram(content)
        turns     = parse_chat_to_turns(content)
        formatted = format_chat_for_ui(turns) or [{"speaker":"Speaker 00","text":content}]

        df = pd.DataFrame(formatted)
        df.to_csv(TEXT_TRANSCRIPT_FILE, index=False)

        sn = safe_name(file.filename)
        df.to_csv(os.path.join(TRANSCRIPTS_DIR, f"{sn}.csv"), index=False)

        fe = os.path.isfile(TEXT_SUMMARY_FILE)
        with open(TEXT_SUMMARY_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_name","timestamp","summary"])
            if not fe: w.writeheader()
            w.writerow({"file_name": file.filename,
                        "timestamp": ist_now(), "summary": summary})

        with open(os.path.join(SUMMARIES_DIR, f"{sn}.json"), "w") as f:
            json.dump({"filename": file.filename, "summary": summary,
                       "saved_at": ist_now()}, f, indent=4)

        return {"status": "success", "summary": summary}
    finally:
        if os.path.exists(tmp): os.remove(tmp)
        gc.collect()

@app.get("/get-text-transcript")
async def get_text_transcript():
    if not os.path.exists(TEXT_TRANSCRIPT_FILE): return []
    return pd.read_csv(TEXT_TRANSCRIPT_FILE).to_dict(orient="records")

@app.get("/get-text-summary")
async def get_text_summary():
    if not os.path.exists(TEXT_SUMMARY_FILE):
        return JSONResponse(content={"summary": "No summary found."})
    df = pd.read_csv(TEXT_SUMMARY_FILE)
    return JSONResponse(content={"summary": str(df.iloc[-1]["summary"]) if not df.empty else "Empty."})

# ============================================================
# SCORING — bias / fairness / efficiency
# ============================================================
_SPEAKER_PREFIX_RE = re.compile(r'^([^:]{1,30}):', re.IGNORECASE)

def normalize_speakers(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    seen: list = []
    for line in lines:
        m = _SPEAKER_PREFIX_RE.match(line)
        if m:
            key = m.group(1).strip().lower()
            if key not in seen: seen.append(key)
    if seen and seen[0] in ("agent", "customer"): return text
    label_map = {}
    for i, key in enumerate(seen):
        label_map[key] = "AGENT" if i == 0 else ("CUSTOMER" if i == 1 else f"SPEAKER_{i}")
    result = []
    for line in lines:
        m = _SPEAKER_PREFIX_RE.match(line)
        if m:
            key  = m.group(1).strip().lower()
            rest = line[m.end():].strip()
            result.append(f"{label_map.get(key, m.group(1).strip())}: {rest}")
        else:
            result.append(line)
    return "\n".join(result)

def compute_bias_scores(text: str) -> dict:
    lines       = [l.strip() for l in text.splitlines() if l.strip()]
    agent_lines = [l for l in lines if l.lower().startswith("agent")]
    cust_lines  = [l for l in lines if l.lower().startswith("customer")]
    agent_text  = " ".join(agent_lines).lower()

    personal_patterns = [r"\bsir\b", r"\bma'?am\b", r"\bmr\.?\b", r"\bms\.?\b"]
    greeting_name = len(re.findall(
        r"(?:thank you|hello|hi|bye)[,\s]+[a-z]{2,15}\b", agent_text))
    personal_count = sum(len(re.findall(p, agent_text)) for p in personal_patterns)
    total_personal = personal_count + greeting_name
    name_score = 5 if total_personal == 0 else (9 if total_personal <= 3 else (7 if total_personal <= 6 else 4))

    negative_phrases = [
        r"\byou (always|never|should(n'?t)? have)\b",
        r"\bthat'?s (not |your )?(my )?(problem|fault)\b",
        r"\bi (can'?t|cannot|won'?t) help\b",
        r"\bnot my (department|problem)\b",
        r"\byou should\b", r"\byou'?ll need to\b",
        r"\bactually\b", r"\blike i said\b",
    ]
    assumption_words = ["obviously", "clearly", "simply", "basic", "easy"]
    neg_hits        = sum(1 for p in negative_phrases if re.search(p, agent_text))
    assumption_hits = sum(agent_text.count(w) for w in assumption_words)
    lang_score      = max(1, 10 - (neg_hits * 2) - assumption_hits)

    polite_open  = ["thank you for calling","how can i help","how may i assist",
                    "good morning","good afternoon","hello","hi"]
    polite_close = ["thank you","have a great","is there anything else",
                    "you're welcome","glad i could","goodbye","take care"]
    dismissive   = ["hold on","wait a moment","i said","as i mentioned"]
    first_agent  = agent_lines[0].lower() if agent_lines else ""
    last_agent   = agent_lines[-1].lower() if agent_lines else ""
    open_ok      = any(p in first_agent for p in polite_open)
    close_ok     = any(p in last_agent  for p in polite_close)
    dismiss_cnt  = sum(agent_text.count(p) for p in dismissive)
    tone_score   = max(1, (9 if (open_ok and close_ok) else 7 if (open_ok or close_ok) else 3) - dismiss_cnt)

    effort_signals = [
        r"\bi (will|can|am going to|have|checked|found|escalat|process|transfer|update)\b",
        r"\blet me\b", r"\bi understand\b", r"\bi apologize\b", r"\bi'?m sorry\b",
        r"\bfor you\b", r"\bright away\b",
    ]
    effort_hits = sum(1 for p in effort_signals if re.search(p, agent_text))
    agent_wc    = len(" ".join(agent_lines).split())
    cust_wc     = max(len(" ".join(cust_lines).split()), 1)
    ratio       = agent_wc / cust_wc
    base_effort = min(7, round(ratio * 4)) if ratio >= 0.5 else 2
    effort_score = min(10, base_effort + min(effort_hits, 3))

    overall = round((name_score + lang_score + tone_score + effort_score) / 4)
    return {"name_neutrality": min(name_score,10), "language_neutrality": min(lang_score,10),
            "tone_consistency": min(tone_score,10), "equal_effort": min(effort_score,10),
            "overall_fairness": min(overall,10)}

def estimate_efficiency(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip() and any(c.isalpha() for c in l)]
    turns = max(len(lines), 1)
    words = len(text.split())
    wpt   = words / turns
    s1 = 4 if wpt < 15 else (3 if wpt < 30 else (2 if wpt < 50 else 1))
    s2 = 4 if turns <= 8 else (3 if turns <= 16 else (2 if turns <= 24 else 1))
    return min(s1 + s2, 10), 0.0

# ============================================================
# SCORING — LLM prompt
# ============================================================
def build_prompt(conversation_text: str) -> str:
    return f"""You are a strict Quality Assurance evaluator for customer-service interactions.
Read ONLY the CUSTOMER's lines carefully to detect emotion and satisfaction.
Every call is different — your scores MUST reflect what is actually said.

CRITICAL RULES:
1. Base every score on specific words/events in THIS conversation.
2. Do NOT default to Neutral or Satisfied.
3. Casual small talk = resolution 1, compliance 1, all sub-scores 1.
4. Resolution 9-10 ONLY if agent confirmed fix AND customer explicitly agreed.
5. Compliance 9-10 ONLY if identity verified AND full protocol followed.

EMPATHY (1-10): 1-3 cold/robotic | 4-6 polite but generic | 7-8 acknowledged issue | 9-10 exceptional warmth
COMPLIANCE (1-10): 1-2 no protocol | 3-5 some steps | 6-7 mostly correct | 8-10 full protocol
RESOLUTION (1-10): 1-2 not addressed | 3-4 incomplete | 5-6 partial | 7-8 resolved | 9-10 confirmed+followup
EFFICIENCY (1-10): 1-3 long/repetitive | 4-5 moderate | 6-7 concise | 8-10 minimal turns

CUSTOMER EMOTION — choose ONE: Happy | Satisfied | Neutral | Anxious | Frustrated | Angry | Sad | Confused
CUSTOMER SATISFACTION — choose ONE: Not Satisfied | Somewhat Satisfied | Neutral | Satisfied | Highly Satisfied
satisfaction_percentage: 0-100 based on how resolved and positive customer was at end.

=== CONVERSATION ===
{conversation_text}

Return ONLY valid JSON:
{{
  "empathy": <1-10>, "compliance": <1-10>, "resolution": <1-10>, "efficiency": <1-10>,
  "empathy_timeline": [
    {{"stage":"Start","score":<1-10>}},{{"stage":"Mid","score":<1-10>}},{{"stage":"End","score":<1-10>}}
  ],
  "compliance_steps": [
    {{"step":"ID Verify","score":<1-10>}},{{"step":"Protocol","score":<1-10>}},{{"step":"Closing","score":<1-10>}}
  ],
  "resolution_progress": [
    {{"stage":"Discovery","score":<1-10>}},{{"stage":"Fixing","score":<1-10>}},{{"stage":"Solved","score":<1-10>}}
  ],
  "customer_emotion": "<emotion>",
  "emotion_confidence": <0-100>,
  "customer_satisfaction": "<satisfaction>",
  "satisfaction_percentage": <0-100>,
  "satisfaction_confidence": <0-100>,
  "reasoning": "<2-3 sentences justifying scores with specific examples from the call>"
}}"""

EMOTION_EMOJI = {"happy":"😊","satisfied":"😌","neutral":"😐","anxious":"😰",
                 "frustrated":"😤","angry":"😡","sad":"😢","confused":"😕"}
SATISFACTION_EMOJI = {"not satisfied":"😞","somewhat satisfied":"😟","neutral":"😐",
                      "satisfied":"😌","highly satisfied":"😊"}
SATISFACTION_PCT_FALLBACK = {"not satisfied":15,"somewhat satisfied":45,
                             "neutral":60,"satisfied":75,"highly satisfied":92}

def enrich_emotion(result: dict) -> dict:
    emotion      = result.get("customer_emotion","Neutral").strip().lower()
    satisfaction = result.get("customer_satisfaction","Neutral").strip().lower()
    result["customer_emotion_emoji"]      = EMOTION_EMOJI.get(emotion,"😐")
    result["customer_satisfaction_emoji"] = SATISFACTION_EMOJI.get(satisfaction,"😐")
    pct = result.get("satisfaction_percentage")
    if pct is None or not isinstance(pct,(int,float)):
        pct = SATISFACTION_PCT_FALLBACK.get(satisfaction,60)
    result["satisfaction_percentage"] = max(0, min(100, int(pct)))
    return result

def _error_result(msg=""):
    return {
        "empathy":0,"compliance":0,"resolution":0,"efficiency":0,
        "efficiency_score":0,"avg_response_time":0.0,
        "empathy_timeline":   [{"stage":"Start","score":0},{"stage":"Mid","score":0},{"stage":"End","score":0}],
        "compliance_steps":   [{"step":"ID Verify","score":0},{"step":"Protocol","score":0},{"step":"Closing","score":0}],
        "resolution_progress":[{"stage":"Discovery","score":0},{"stage":"Fixing","score":0},{"stage":"Solved","score":0}],
        "customer_emotion":"Neutral","customer_emotion_emoji":"😐",
        "customer_satisfaction":"Neutral","customer_satisfaction_emoji":"😐",
        "emotion_confidence":0,"satisfaction_confidence":0,"satisfaction_percentage":0,
        "bias":{"name_neutrality":0,"language_neutrality":0,"tone_consistency":0,
                "equal_effort":0,"overall_fairness":0},
        "reasoning": f"Analysis failed: {msg}",
    }

def _append_history(result: dict, filename: str):
    history = []
    if os.path.exists(SCORES_HISTORY_FILE):
        try:
            with open(SCORES_HISTORY_FILE) as f: history = json.load(f)
        except: pass
    history.append({"file_name": filename, "timestamp": ist_now(), **result})
    with open(SCORES_HISTORY_FILE,"w") as f: json.dump(history, f, indent=2)

async def _run_scoring(conversation_text: str, filename: str) -> dict:
    if not conversation_text.strip():
        raise ValueError("Empty conversation")

    conversation_text = normalize_speakers(conversation_text)
    bias_scores       = compute_bias_scores(conversation_text)
    eff, avg_rt       = estimate_efficiency(conversation_text)

    anon = re.sub(r"\b\d{6,}\b","[ACCOUNT]", conversation_text)
    anon = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b","[EMAIL]", anon)
    anon = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b","[PHONE]", anon)

    response = groq_client.chat.completions.create(
        messages=[{"role":"user","content": build_prompt(anon[:8000])}],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        response_format={"type":"json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    result["efficiency_score"]  = eff
    result["avg_response_time"] = avg_rt
    result["bias"]              = bias_scores
    result = enrich_emotion(result)

    with open(SCORES_FILE,"w") as f: json.dump(result, f, indent=4)
    _append_history(result, filename)

    # save per-file scores
    sn = safe_name(filename)
    result["original_filename"] = filename
    result["saved_at"]          = ist_now()
    with open(os.path.join(SCORES_DIR, f"{sn}.json"),"w") as f:
        json.dump(result, f, indent=4)

    return result

# ============================================================
# SCORING ENDPOINTS
# ============================================================
@app.post("/analyze-quality")
async def analyze_quality(
    file: UploadFile = File(...),
    original_filename: str = Form(None),
):
    try:
        display_name   = original_filename if original_filename else file.filename
        original_lower = display_name.lower().strip()
        is_audio = any(original_lower.endswith(ext) for ext in [".mp3",".wav",".m4a",".mp4"])

        if is_audio:
            # Wait for audio transcript to be saved by /upload
            time.sleep(10)
            sn   = safe_name(display_name)
            path = os.path.join(TRANSCRIPTS_DIR, f"{sn}.csv")
            if not os.path.exists(path):
                path = AUDIO_TRANSCRIPT_FILE
            if not os.path.exists(path):
                return _error_result("No transcript found. Upload audio first.")
            df   = pd.read_csv(path)
            conv = "\n".join(
                f"{row['speaker']}: {row['text']}"
                for _, row in df.iterrows()
                if str(row.get('text','')).strip()
            )
        else:
            raw = await file.read()
            if not raw:
                return _error_result("Empty file.")
            try:    conv = raw.decode("utf-8")
            except: conv = raw.decode("latin-1")

        return await _run_scoring(conv, display_name)

    except Exception as e:
        err = _error_result(str(e))
        with open(SCORES_FILE,"w") as f: json.dump(err, f, indent=4)
        return err

class TranscriptRow(BaseModel):
    speaker: str
    text: str
    start: float | None = None

class DirectScoreRequest(BaseModel):
    filename: str
    transcript: list[TranscriptRow]

@app.post("/analyze-quality-direct")
async def analyze_quality_direct(req: DirectScoreRequest):
    try:
        rows = [f"{r.speaker}: {r.text}" for r in req.transcript if r.text.strip()]
        if not rows: raise ValueError("Empty transcript.")
        return await _run_scoring("\n".join(rows), req.filename)
    except Exception as e:
        err = _error_result(str(e))
        with open(SCORES_FILE,"w") as f: json.dump(err, f, indent=4)
        return err

@app.get("/get-quality-scores")
async def get_quality_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f: return json.load(f)
    return _error_result("No scores yet.")

@app.get("/get-file-scores/{filename:path}")
async def get_file_scores(filename: str):
    from urllib.parse import unquote
    path = os.path.join(SCORES_DIR, f"{safe_name(unquote(filename))}.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return _error_result()

@app.get("/list-file-scores")
async def list_file_scores():
    try:
        files = []
        for fname in os.listdir(SCORES_DIR):
            if not fname.endswith(".json"): continue
            with open(os.path.join(SCORES_DIR, fname)) as f:
                d = json.load(f)
            files.append({"filename":   d.get("original_filename", fname),
                          "saved_at":   d.get("saved_at",""),
                          "empathy":    d.get("empathy",0),
                          "compliance": d.get("compliance",0),
                          "resolution": d.get("resolution",0)})
        files.sort(key=lambda x: x["saved_at"], reverse=True)
        return files
    except Exception as e:
        return []

@app.get("/get-aggregate-scores")
async def get_aggregate_scores():
    if os.path.exists(SCORES_HISTORY_FILE):
        try:
            with open(SCORES_HISTORY_FILE) as f: history = json.load(f)
            if history:
                n   = len(history)
                agg = {}
                for key in ["empathy","compliance","resolution","efficiency_score",
                            "satisfaction_percentage"]:
                    vals = [e[key] for e in history if isinstance(e.get(key),(int,float))]
                    agg[key] = round(sum(vals)/len(vals),2) if vals else 0
                agg["file_count"] = n
                agg["reasoning"]  = history[-1].get("reasoning","")
                emotion = Counter(e.get("customer_emotion","Neutral") for e in history)
                sat     = Counter(e.get("customer_satisfaction","Neutral") for e in history)
                agg["customer_emotion"]      = emotion.most_common(1)[0][0]
                agg["customer_satisfaction"] = sat.most_common(1)[0][0]
                agg["customer_emotion_emoji"]      = EMOTION_EMOJI.get(agg["customer_emotion"].lower(),"😐")
                agg["customer_satisfaction_emoji"] = SATISFACTION_EMOJI.get(agg["customer_satisfaction"].lower(),"😐")
                return agg
        except: pass
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f:
            d = json.load(f); d.setdefault("file_count",1); return d
    return _error_result("No data yet.")

@app.get("/get-analysis")
async def get_analysis():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE) as f: return json.load(f)
    return _error_result("No analysis yet.")

@app.post("/start-session")
async def start_session():
    removed = []
    for fpath in [SCORES_HISTORY_FILE, SCORES_FILE]:
        if os.path.exists(fpath): os.remove(fpath); removed.append(fpath)
    return {"status":"session_started","removed":removed}

@app.delete("/clear-scores-history")
async def clear_scores_history():
    removed = []
    for fpath in [SCORES_HISTORY_FILE, SCORES_FILE]:
        if os.path.exists(fpath): os.remove(fpath); removed.append(fpath)
    return {"status":"cleared","removed":removed}

@app.get("/scores-history")
async def scores_history():
    if not os.path.exists(SCORES_HISTORY_FILE): return []
    try:
        with open(SCORES_HISTORY_FILE) as f: history = json.load(f)
        return [{"file_name":e.get("file_name",""),"timestamp":e.get("timestamp",""),
                 "empathy":e.get("empathy",0),"compliance":e.get("compliance",0),
                 "resolution":e.get("resolution",0)} for e in history]
    except Exception as e: return {"error":str(e)}

# ============================================================
# HEALTH
# ============================================================
@app.get("/")
@app.get("/health")
async def health():
    return {"status":"ok","service":"AuraQ Combined Server"}

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)