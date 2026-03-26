import os
import re
import json
import time
import uuid
import shutil
import asyncio
import httpx
import pandas as pd
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


# ── RAG: LangChain + Pinecone + local embeddings ─────────────────────────────
try:
    # langchain_text_splitters is the correct package in LangChain >= 0.2
    # Fallback to the old path for older installs
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec
    _RAG_LIBS_AVAILABLE = True
except ImportError:
    _RAG_LIBS_AVAILABLE = False
    print("[RAG] WARNING: RAG libraries not installed. "
          "Run: pip install -r requirements.txt  "
          "Policy upload will be disabled until then.")

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing required env var: GROQ_API_KEY")

TRANSCRIPT_FILE     = "transcriptions_with_speakers.csv"
SCORES_FILE         = "quality_scores.json"
SCORES_HISTORY_FILE = "quality_scores_history.json"

# ── Pinecone config (optional — RAG is disabled if key is absent) ─────────────
PINECONE_API_KEY    = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "auraq-policy")
EMBEDDING_MODEL     = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM       = 384          # must match your Pinecone index dimension
CHUNK_SIZE          = 512
CHUNK_OVERLAP       = 64
TOP_K               = 5
UPSERT_BATCH        = 100

UPLOAD_TEMP_DIR     = "rag_uploads"
POLICY_META_FILE    = "policy_meta.json"
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance):
    """
    Pre-warm the embedding model and Pinecone connection at startup.
    This means the first /upload-policy request won't have to wait 20+ seconds
    for the model to download — it will already be cached and ready.
    """
    if _rag_available():
        print("[RAG] Pre-warming embedding model and Pinecone at startup…")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_embed_model)   # download/cache model
        await loop.run_in_executor(None, _get_pinecone_index)  # connect to Pinecone
        print("[RAG] Pre-warm complete — policy upload is ready.")
    yield  # server runs here
    # (nothing to clean up on shutdown)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = Groq(api_key=GROQ_API_KEY)

GROQ_CONCURRENCY = int(os.environ.get("GROQ_CONCURRENCY", "3"))
_GROQ_SEMAPHORE  = None

def get_groq_semaphore() -> asyncio.Semaphore:
    global _GROQ_SEMAPHORE
    if _GROQ_SEMAPHORE is None:
        _GROQ_SEMAPHORE = asyncio.Semaphore(GROQ_CONCURRENCY)
    return _GROQ_SEMAPHORE


# ==============================================================================
# RAG — PINECONE + LOCAL EMBEDDINGS (integrated, no separate server needed)
# ==============================================================================

_pinecone_index  = None   # Pinecone Index object, initialised lazily
_embed_model     = None   # SentenceTransformer, loaded once on first use


def _rag_available() -> bool:
    """True only if RAG libs are installed AND a Pinecone API key is configured."""
    return _RAG_LIBS_AVAILABLE and bool(PINECONE_API_KEY)


def _get_embed_model():
    """Load the embedding model once and cache it."""
    global _embed_model
    if _embed_model is None:
        print(f"[RAG] Loading embedding model '{EMBEDDING_MODEL}'…")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print("[RAG] Embedding model ready.")
    return _embed_model


def _get_pinecone_index():
    """Connect to Pinecone and return the index object (lazy init)."""
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    if not _rag_available():
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            print(f"[RAG] Creating Pinecone index '{PINECONE_INDEX_NAME}'…")
            pc.create_index(
                name      = PINECONE_INDEX_NAME,
                dimension = EMBEDDING_DIM,
                metric    = "cosine",
                spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print(f"[RAG] Index '{PINECONE_INDEX_NAME}' created.")
        else:
            print(f"[RAG] Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        return _pinecone_index
    except Exception as e:
        print(f"[RAG] Pinecone init error: {e}")
        return None


def _embed(texts: list) -> list:
    return _get_embed_model().encode(texts, normalize_embeddings=True).tolist()


def _make_namespace(filename: str) -> str:
    base = Path(filename).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base).lower()


def _load_policy_meta() -> dict:
    if os.path.exists(POLICY_META_FILE):
        try:
            with open(POLICY_META_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_policy_meta(meta: dict):
    with open(POLICY_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def _policy_loaded() -> bool:
    """Returns True if a policy has been uploaded and vectors exist in Pinecone."""
    if not _rag_available():
        return False
    meta = _load_policy_meta()
    return bool(meta.get("namespace"))


async def _fetch_policy_context(conversation_text: str) -> list:
    """
    If a policy is loaded in Pinecone, retrieve the top-K most relevant
    policy chunks via cosine similarity.

    Strategy: run THREE focused queries covering different compliance angles,
    then deduplicate. This dramatically improves recall compared to a single
    agent-lines-only query.
    """
    if not _policy_loaded():
        return []

    try:
        index = _get_pinecone_index()
        if index is None:
            return []

        meta      = _load_policy_meta()
        namespace = meta.get("namespace", "")

        lines       = [l for l in conversation_text.splitlines() if l.strip()]
        agent_lines = [l for l in lines if l.lower().startswith("agent")]
        cust_lines  = [l for l in lines if l.lower().startswith("customer")]

        # Three complementary queries:
        # 1. Agent procedure lines — what did the agent DO
        # 2. Full conversation summary — overall context
        # 3. Customer request — what was the PURPOSE of the call
        queries = [
            " ".join(agent_lines[:6]) if agent_lines else conversation_text[:400],
            conversation_text[:600],
            " ".join(cust_lines[:4]) if cust_lines else conversation_text[400:800],
        ]
        queries = [q.strip() for q in queries if q.strip()]

        loop = asyncio.get_event_loop()

        # Embed all queries in one batch
        query_vectors = await loop.run_in_executor(None, lambda: _embed(queries))

        # Query Pinecone for each, collect unique chunks by text
        seen_texts: set = set()
        all_chunks: list = []

        for qv in query_vectors:
            result = await loop.run_in_executor(
                None,
                lambda v=qv: index.query(
                    vector=v,
                    top_k=4,
                    namespace=namespace,
                    include_metadata=True,
                )
            )
            for match in result["matches"]:
                text = match.get("metadata", {}).get("text", "").strip()
                # Only include if score > 0.3 (relevance threshold) and not duplicate
                if text and match["score"] >= 0.3 and text not in seen_texts:
                    seen_texts.add(text)
                    all_chunks.append((match["score"], text))

        # Sort by score descending, take top 5
        all_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [text for _, text in all_chunks[:5]]

        top_score = all_chunks[0][0] if all_chunks else 0
        print(f"  [RAG] {len(top_chunks)} policy chunks retrieved "
              f"(top score={round(top_score, 3)}, ns='{namespace}')")
        return top_chunks

    except Exception as e:
        print(f"  [RAG] Retrieval failed: {e} — falling back to generic rubrics")
        return []


# ==============================================================================
# SPEAKER NORMALIZER
# Fix: audio transcripts come out as "Speaker 00 / Speaker 01" not "agent/customer"
# so every downstream function gets correct AGENT: / CUSTOMER: prefixes
# ==============================================================================

_SPEAKER_PREFIX_RE = re.compile(r'^([^:]{1,30}):', re.IGNORECASE)

def normalize_speakers(text: str) -> str:
    """
    Rewrite any speaker format to AGENT: / CUSTOMER: so bias scoring,
    efficiency and LLM prompt all work correctly.

    Handles:
      Speaker 00 / Speaker 01  (Whisper transcription default)
      speaker_00 / speaker_01
      S1 / S2
      Agent / Customer         (already correct — pass through)
      Any two-speaker format   (first encountered = AGENT, second = CUSTOMER)
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Collect unique speaker keys in order of first appearance
    seen: list[str] = []
    for line in lines:
        m = _SPEAKER_PREFIX_RE.match(line)
        if m:
            key = m.group(1).strip().lower()
            if key not in seen:
                seen.append(key)

    # Already normalised?
    if seen and seen[0] in ("agent", "customer"):
        return text

    # Build map: first speaker → AGENT, second → CUSTOMER, rest → SPEAKER_N
    label_map: dict[str, str] = {}
    for i, key in enumerate(seen):
        if i == 0:
            label_map[key] = "AGENT"
        elif i == 1:
            label_map[key] = "CUSTOMER"
        else:
            label_map[key] = f"SPEAKER_{i}"

    normalised: list[str] = []
    for line in lines:
        m = _SPEAKER_PREFIX_RE.match(line)
        if m:
            key  = m.group(1).strip().lower()
            rest = line[m.end():].strip()
            normalised.append(f"{label_map.get(key, m.group(1).strip())}: {rest}")
        else:
            normalised.append(line)

    return "\n".join(normalised)


# ==============================================================================
# EFFICIENCY MODULE
# ==============================================================================

def calculate_response_time(messages: list) -> float:
    times = []
    for i in range(len(messages) - 1):
        if (messages[i].get("speaker", "").lower() == "customer"
                and messages[i + 1].get("speaker", "").lower() == "agent"):
            ts1 = messages[i].get("timestamp")
            ts2 = messages[i + 1].get("timestamp")
            if ts1 and ts2:
                try:
                    diff = (datetime.strptime(ts2, "%H:%M:%S")
                            - datetime.strptime(ts1, "%H:%M:%S")).total_seconds()
                    if diff >= 0:
                        times.append(diff)
                except ValueError:
                    pass
    return round(sum(times) / len(times), 2) if times else 0.0


def efficiency_score_from_messages(messages: list) -> tuple[int, float]:
    avg_rt = calculate_response_time(messages)
    total  = len(messages)
    score  = 0
    if avg_rt == 0:   score += 3
    elif avg_rt < 30: score += 4
    elif avg_rt < 60: score += 3
    else:             score += 1
    if total <= 6:    score += 4
    elif total <= 12: score += 3
    elif total <= 20: score += 2
    else:             score += 1
    return min(score, 10), avg_rt


def estimate_efficiency_from_text(text: str) -> tuple[int, float]:
    """
    Efficiency from plain text (audio / txt / csv).
    Uses dialogue-line count and words-per-turn as proxies.
    Scores will differ because each call has a different length and verbosity.
    """
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and any(c.isalpha() for c in l)]
    turns = max(len(lines), 1)
    words = len(text.split())
    wpt   = words / turns   # average words per turn

    score = 0
    # Verbosity sub-score
    if wpt < 15:   score += 4
    elif wpt < 30: score += 3
    elif wpt < 50: score += 2
    else:          score += 1
    # Turn-count sub-score
    if turns <= 8:    score += 4
    elif turns <= 16: score += 3
    elif turns <= 24: score += 2
    else:             score += 1

    return min(score, 10), 0.0


# ==============================================================================
# BIAS / FAIRNESS MODULE
# Fix: now works on normalised text so Speaker 00/01 format is handled
# Fix: added missing negative-language patterns that were not triggering
# ==============================================================================

def compute_bias_scores(conversation_text: str) -> dict:
    """
    Four content-driven fairness sub-scores based on agent behaviour signals.
    Requires AGENT: / CUSTOMER: prefixes — call normalize_speakers() first.
    Scores vary per conversation because they are computed from actual dialogue.
    """
    lines       = [l.strip() for l in conversation_text.splitlines() if l.strip()]
    agent_lines = [l for l in lines if l.lower().startswith("agent")]
    cust_lines  = [l for l in lines if l.lower().startswith("customer")]
    agent_text  = " ".join(agent_lines).lower()
    cust_text   = " ".join(cust_lines).lower()

    # ── 1. Name Neutrality ───────────────────────────────────────────────────
    # Did agent personalise appropriately? 0 = cold, 1-3 = ideal, 6+ = robotic
    personal_patterns = [r"\bsir\b", r"\bma'?am\b", r"\bmr\.?\b", r"\bms\.?\b", r"\bmrs\.?\b"]
    greeting_name     = len(re.findall(
        r"(?:thank you|hello|hi|bye|goodbye|welcome)[,\s]+[a-z]{2,15}\b", agent_text))
    personal_count    = sum(len(re.findall(p, agent_text)) for p in personal_patterns)
    total_personal    = personal_count + greeting_name

    if total_personal == 0:   name_score = 5
    elif total_personal <= 3: name_score = 9
    elif total_personal <= 6: name_score = 7
    else:                     name_score = 4

    # ── 2. Language Neutrality ───────────────────────────────────────────────
    # Did agent avoid dismissive, blame-shifting or condescending language?
    negative_phrases = [
        # Original patterns
        r"\byou (always|never|should(n'?t)? have|must have)\b",
        r"\bthat'?s (not |your |)(my )?(problem|fault)\b",
        r"\bi (can'?t|cannot|won'?t|will not) help\b",
        r"\bwe don'?t do that\b",
        r"\bnot my (department|problem)\b",
        r"\bthere'?s nothing (i|we) can do\b",
        # Fix: added patterns that were missing — common dismissive real-world phrases
        r"\byou should\b",                  # "you should restart", "you should know"
        r"\bnot much (i|we) can do\b",      # "there's not much I can do"
        r"\byou'?ll need to\b",             # "you'll need to call back / submit online"
        r"\bjust (restart|reboot|reset|submit|go online|call back|check)\b",  # oversimplification
        r"\bactually\b",                    # condescending correction opener
        r"\blike i said\b",                 # dismissive repetition
        r"\bthat'?s not how (it|this) works\b",
    ]
    assumption_words = ["obviously", "clearly", "simply", "basic", "easy", "just try"]
    neg_hits         = sum(1 for p in negative_phrases if re.search(p, agent_text))
    assumption_hits  = sum(agent_text.count(w) for w in assumption_words)
    lang_score       = max(1, 10 - (neg_hits * 2) - assumption_hits)

    # ── 3. Tone Consistency ──────────────────────────────────────────────────
    # Did agent open AND close professionally? Penalise dismissive mid-call phrases.
    polite_open  = ["thank you for calling", "how can i help", "how may i assist",
                    "good morning", "good afternoon", "good evening", "hello", "hi"]
    polite_close = ["thank you", "have a great", "have a good", "is there anything else",
                    "my pleasure", "you're welcome", "glad i could", "take care", "goodbye",
                    "have a wonderful", "appreciate your call"]
    dismissive   = ["hold on", "wait a moment", "one moment", "look,", "listen,",
                    "i said", "as i mentioned", "i already told you"]

    first_agent   = agent_lines[0].lower() if agent_lines else ""
    last_agent    = agent_lines[-1].lower() if agent_lines else ""
    open_polite   = any(p in first_agent for p in polite_open)
    close_polite  = any(p in last_agent  for p in polite_close)
    dismiss_count = sum(agent_text.count(p) for p in dismissive)

    if open_polite and close_polite:  tone_score = max(1, 9 - dismiss_count)
    elif open_polite or close_polite: tone_score = max(1, 7 - dismiss_count)
    else:                             tone_score = max(1, 3 - dismiss_count)

    # ── 4. Equal Effort ──────────────────────────────────────────────────────
    # Did agent invest effort? Count action-verb signals + agent/customer word ratio.
    effort_signals = [
        r"\bi (will|can|am going to|have|checked|looked|found|escalat|process|transfer|update)\b",
        r"\blet me\b", r"\bi understand\b", r"\bi apologize\b", r"\bi'?m sorry\b",
        r"\bwe (can|will|have)\b", r"\bfor you\b", r"\bright away\b", r"\bimmediately\b",
        r"\bi'?ll (take|make sure|ensure|follow up|call|send|process|check)\b",
    ]
    effort_hits  = sum(1 for p in effort_signals if re.search(p, agent_text))
    agent_wc     = len(" ".join(agent_lines).split())
    cust_wc      = max(len(" ".join(cust_lines).split()), 1)
    ratio        = agent_wc / cust_wc
    base_effort  = min(7, round(ratio * 4)) if ratio >= 0.5 else 2
    effort_score = min(10, base_effort + min(effort_hits, 3))

    overall = round((name_score + lang_score + tone_score + effort_score) / 4)

    return {
        "name_neutrality":     min(name_score,   10),
        "language_neutrality": min(lang_score,   10),
        "tone_consistency":    min(tone_score,   10),
        "equal_effort":        min(effort_score, 10),
        "overall_fairness":    min(overall,      10),
    }


# ==============================================================================
# LLM PROMPT
# ==============================================================================

def build_prompt(conversation_text: str, policy_chunks: list[str] | None = None) -> str:
    # Build the policy context section if chunks are available
    policy_section = ""
    if policy_chunks:
        formatted = "\n\n".join(f"[Policy Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(policy_chunks))
        policy_section = f"""
=== COMPANY POLICY CONTEXT ===
The following excerpts are from the company's official policy document.
Use them to evaluate COMPLIANCE. If the agent followed these rules, score higher.
If the agent violated or skipped them, penalise the compliance score accordingly.

{formatted}

IMPORTANT: When policy context is provided, compliance scoring MUST reference it.
A score of 8-10 requires the agent to have followed the specific policy steps above.
A score of 1-4 means the agent clearly violated or ignored these policy rules.

"""

    return f"""
You are a strict Quality Assurance evaluator for customer-service interactions.
Read ONLY the CUSTOMER's lines carefully to detect emotion and satisfaction.
Every call is different — your scores MUST reflect what is actually said.

CRITICAL RULES:
1. Base every score on specific words/events in THIS conversation.
2. Do NOT default to Neutral or Satisfied. Read the customer's actual words.
3. Casual small talk = resolution 1, compliance 1, all sub-scores 1.
4. Resolution 9-10 ONLY if agent confirmed fix AND customer explicitly agreed.
5. Compliance 9-10 ONLY if identity verified AND full protocol followed.
6. SERVICE calls (orders, bookings, inquiries) are NOT casual chat — if the agent
   successfully completed the customer's request, resolution must be at least 7.
{policy_section}
=== RUBRICS ===

EMPATHY (1-10):
1-3: Cold, robotic, ignored feelings entirely.
4-5: Polite but generic, no personalisation.
6-7: Acknowledged issue, used "I understand".
8-9: Used customer's name, showed genuine care.
10: Exceptional warmth throughout.

COMPLIANCE (1-10):
1-2: No verification or protocol at all.
3-5: Some steps done, others skipped.
6-7: Mostly correct, minor gaps.
8-9: Full verification + structured handling.
10: Perfect best-practice execution.
{
    "→ COMPLIANCE NOTE: Score strictly against the COMPANY POLICY CONTEXT above, not just generic best practice."
    if policy_chunks else ""
}

RESOLUTION (1-10):
1-2: Issue not addressed.
3-4: Tried but incomplete.
5-6: Partially resolved.
7-8: Resolved clearly.
9-10: Resolved + confirmed by customer + follow-up offered.
→ Pure casual chat with no request and no issue = 1. 
→ Completed service requests (order placed, info given, booking made) = 7 minimum.

EFFICIENCY (1-10):
1-3: Long, repetitive, circular.
4-5: Moderate, unnecessary turns.
6-7: Reasonably concise.
8-9: On-point, minimal turns.
10: Resolved in fewest possible turns.

=== CUSTOMER EMOTION ===
Read ONLY the CUSTOMER's messages. Identify their DOMINANT emotional state across the whole call.
Choose EXACTLY ONE from this list:

- Happy      → Customer is positive, cheerful, grateful. Example: "This is amazing! Thank you so much!"
- Satisfied  → Customer is content, calm, pleased at end. Example: "Great, that works. Thanks."
- Neutral    → Customer is matter-of-fact, no strong emotion. Example: "Okay. Is that all?"
- Anxious    → Customer expresses worry, nervousness, urgency. Example: "I'm really worried about this." / "I need this fixed urgently."
- Frustrated → Customer shows repeated irritation, sighing, complaint. Example: "This is the third time I'm calling!"
- Angry      → Customer is hostile, aggressive, threatening. Example: "This is unacceptable! I want to cancel!"
- Sad        → Customer sounds distressed, upset, or defeated. Example: "I don't know what to do anymore."
- Confused   → Customer is lost, asking repeated clarifying questions. Example: "Wait, I don't understand what you mean."

IMPORTANT: Do NOT pick Neutral unless the customer truly shows zero emotional signal.
IMPORTANT: Pick the emotion that best describes the MAJORITY of the call, not just the end.

=== CUSTOMER SATISFACTION (after the call ends) ===
Based on: (a) was their issue resolved? (b) how did they sound at the END of the call?
Choose EXACTLY ONE label AND provide a percentage (0-100):

- Not Satisfied   (0-30%)   → Issue unresolved, customer still upset or complained at end.
- Somewhat Satisfied (31-55%) → Issue partially resolved or customer lukewarm.
- Neutral         (56-65%)  → No strong signal either way.
- Satisfied       (66-85%)  → Issue resolved, customer acknowledged it positively.
- Highly Satisfied (86-100%) → Customer explicitly thanked, praised, or expressed delight.

satisfaction_percentage rules:
- If customer said something like "this is terrible" at end → 5-15%
- If customer said "okay thanks" without enthusiasm → 50-60%
- If customer said "thank you so much, that's perfect" → 85-95%
- If customer said nothing about outcome → 50%

=== DETAIL SCORING ===

empathy_timeline — score agent empathy at each stage independently:
  Start: Opening/greeting tone
  Mid:   Core conversation tone
  End:   Closing tone

compliance_steps — score each independently (1 if it didn't happen):
  ID Verify: Was customer identity verified?
  Protocol:  Was correct process/script followed?
  Closing:   Was resolution confirmed and call closed properly?

resolution_progress — score each independently (1 for casual chat):
  Discovery: Did agent identify/understand the issue?
  Fixing:    Did agent take concrete action?
  Solved:    Was issue confirmed closed and customer agreed?

=== CONVERSATION ===
{conversation_text}

=== OUTPUT — return ONLY valid JSON, no markdown, no text outside the object ===
{{
  "empathy":    <integer 1-10>,
  "compliance": <integer 1-10>,
  "resolution": <integer 1-10>,
  "efficiency": <integer 1-10>,
  "empathy_timeline": [
    {{"stage": "Start", "score": <integer 1-10>}},
    {{"stage": "Mid",   "score": <integer 1-10>}},
    {{"stage": "End",   "score": <integer 1-10>}}
  ],
  "compliance_steps": [
    {{"step": "ID Verify", "score": <integer 1-10>}},
    {{"step": "Protocol",  "score": <integer 1-10>}},
    {{"step": "Closing",   "score": <integer 1-10>}}
  ],
  "resolution_progress": [
    {{"stage": "Discovery", "score": <integer 1-10>}},
    {{"stage": "Fixing",    "score": <integer 1-10>}},
    {{"stage": "Solved",    "score": <integer 1-10>}}
  ],
  "customer_emotion":          "<Happy|Satisfied|Neutral|Anxious|Frustrated|Angry|Sad|Confused>",
  "emotion_confidence":        <integer 0-100>,
  "customer_satisfaction":     "<Not Satisfied|Somewhat Satisfied|Neutral|Satisfied|Highly Satisfied>",
  "satisfaction_percentage":   <integer 0-100>,
  "satisfaction_confidence":   <integer 0-100>,
  "reasoning": "<2-3 sentences quoting specific customer lines to justify emotion and satisfaction scores>",
  "policy_violations": "<brief list of policy rules that were broken, or 'None' if all rules followed>"
}}
"""


async def call_llama_async(prompt_text: str, max_retries: int = 5) -> dict:
    """
    Async Groq call with:
    - Semaphore to cap concurrent requests (GROQ_CONCURRENCY)
    - Exponential backoff retry on rate-limit (429) errors
    This makes batch scoring of 100 calls safe on the free tier.
    """
    async with get_groq_semaphore():
        loop = asyncio.get_event_loop()
        for attempt in range(1, max_retries + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_text}],
                        model="llama-3.1-8b-instant",
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** attempt   # 2s, 4s, 8s, 16s
                    print(f"  [GROQ] Rate limit hit (attempt {attempt}), retrying in {wait}s…")
                    await asyncio.sleep(wait)
                    continue
                raise  # re-raise on non-rate-limit errors or exhausted retries


def call_llama(prompt_text: str) -> dict:
    """Sync wrapper kept for backward compatibility."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_text}],
        model="llama-3.1-8b-instant",
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ==============================================================================
# EMOTION / SATISFACTION HELPERS
# ==============================================================================

EMOTION_EMOJI = {
    "happy":      "😊",
    "satisfied":  "😌",
    "neutral":    "😐",
    "anxious":    "😰",
    "frustrated": "😤",
    "angry":      "😡",
    "sad":        "😢",
    "confused":   "😕",
}

SATISFACTION_EMOJI = {
    "not satisfied":       "😞",
    "somewhat satisfied":  "😟",
    "neutral":             "😐",
    "satisfied":           "😌",
    "highly satisfied":    "😊",
}

# Map satisfaction label → fallback percentage midpoint if LLM forgets to return one
SATISFACTION_PCT_FALLBACK = {
    "not satisfied":      15,
    "somewhat satisfied": 45,
    "neutral":            60,
    "satisfied":          75,
    "highly satisfied":   92,
}


def enrich_emotion(result: dict) -> dict:
    emotion      = result.get("customer_emotion",      "Neutral").strip().lower()
    satisfaction = result.get("customer_satisfaction", "Neutral").strip().lower()

    result["customer_emotion_emoji"]      = EMOTION_EMOJI.get(emotion,      "😐")
    result["customer_satisfaction_emoji"] = SATISFACTION_EMOJI.get(satisfaction, "😐")

    # Ensure satisfaction_percentage is always present and in valid range
    pct = result.get("satisfaction_percentage")
    if pct is None or not isinstance(pct, (int, float)):
        pct = SATISFACTION_PCT_FALLBACK.get(satisfaction, 60)
    result["satisfaction_percentage"] = max(0, min(100, int(pct)))

    # Log so we can see what the LLM actually decided
    print(f"  emotion={emotion} ({result.get('emotion_confidence','?')}%) | "
          f"satisfaction={satisfaction} {result['satisfaction_percentage']}% "
          f"({result.get('satisfaction_confidence','?')}%)")
    return result


# ==============================================================================
# HISTORY + AGGREGATE HELPERS
# ==============================================================================

# Numeric score keys that can be meaningfully averaged
_SCALAR_KEYS = ["empathy", "compliance", "resolution", "efficiency", "efficiency_score",
                "avg_response_time", "emotion_confidence", "satisfaction_confidence",
                "satisfaction_percentage"]

# Sub-array keys — list of {label_key: str, score: int} dicts
_ARRAY_KEYS = [
    ("empathy_timeline",    "stage"),
    ("compliance_steps",    "step"),
    ("resolution_progress", "stage"),
]

# Bias sub-keys
_BIAS_KEYS = ["name_neutrality", "language_neutrality", "tone_consistency",
              "equal_effort", "overall_fairness"]


def _append_to_history(result: dict, filename: str) -> None:
    """Append a single call's scores to the history file."""
    history: list = []
    if os.path.exists(SCORES_HISTORY_FILE):
        try:
            with open(SCORES_HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append({
        "file_name": filename,
        "timestamp": datetime.now().isoformat(),
        **result,
    })
    with open(SCORES_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History now has {len(history)} entries.")


def _compute_aggregate(history: list) -> dict:
    """
    Average all numeric scores across every entry in history.
    Sub-arrays (empathy_timeline etc.) are averaged element-by-element.
    Categorical fields (emotion, satisfaction) use the most-common value.
    The most-recent reasoning string is kept as-is.
    """
    if not history:
        return {}

    n = len(history)

    # ── Scalar averages ───────────────────────────────────────────────────────
    agg: dict = {}
    for key in _SCALAR_KEYS:
        vals = [e[key] for e in history if isinstance(e.get(key), (int, float))]
        agg[key] = round(sum(vals) / len(vals), 2) if vals else 0

    # ── Sub-array averages ────────────────────────────────────────────────────
    for arr_key, label_key in _ARRAY_KEYS:
        # Collect all entries that have this array
        arrays = [e[arr_key] for e in history if isinstance(e.get(arr_key), list)]
        if not arrays:
            continue
        # Use first entry as template for labels
        template = arrays[0]
        averaged = []
        for item in template:
            lbl    = item.get(label_key) or item.get("step") or item.get("stage")
            scores = [
                a[i]["score"]
                for a in arrays
                for i, elem in enumerate(a)
                if (elem.get(label_key) or elem.get("step") or elem.get("stage")) == lbl
                and isinstance(elem.get("score"), (int, float))
            ]
            averaged.append({
                label_key: lbl,
                "score": round(sum(scores) / len(scores), 1) if scores else 0,
            })
        agg[arr_key] = averaged

    # ── Bias sub-key averages ─────────────────────────────────────────────────
    bias_entries = [e["bias"] for e in history if isinstance(e.get("bias"), dict)]
    if bias_entries:
        bias_agg: dict = {}
        for bk in _BIAS_KEYS:
            vals = [b[bk] for b in bias_entries if isinstance(b.get(bk), (int, float))]
            bias_agg[bk] = round(sum(vals) / len(vals), 1) if vals else 0
        agg["bias"] = bias_agg

    # ── Categorical: most-common value ───────────────────────────────────────
    from collections import Counter
    for cat_key in ["customer_emotion", "customer_satisfaction"]:
        vals = [e[cat_key] for e in history if isinstance(e.get(cat_key), str)]
        if vals:
            agg[cat_key] = Counter(vals).most_common(1)[0][0]

    # ── Emoji: derive from the most-common label ──────────────────────────────
    emotion      = agg.get("customer_emotion", "Neutral").strip().lower()
    satisfaction = agg.get("customer_satisfaction", "Neutral").strip().lower()
    agg["customer_emotion_emoji"]      = EMOTION_EMOJI.get(emotion, "😐")
    agg["customer_satisfaction_emoji"] = SATISFACTION_EMOJI.get(satisfaction, "😐")

    # ── Meta ──────────────────────────────────────────────────────────────────
    agg["file_count"] = n
    agg["reasoning"]  = (
        history[-1].get("reasoning", "")
        or f"Aggregate of {n} call{'s' if n != 1 else ''}."
    )

    return agg


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.post("/analyze-quality")
async def analyze_quality(file: UploadFile = File(...)):
    print(f"\n--- ANALYZING: {file.filename} ---")
    try:
        file_ext  = file.filename.lower().rsplit(".", 1)[-1]
        raw_bytes = await file.read()   # read ONCE — avoids empty second-read bug

        conversation_text = ""
        messages_list: list = []

        # ── Route by file type ───────────────────────────────────────────────
        if file_ext == "json":
            data          = json.loads(raw_bytes.decode("utf-8"))
            messages_list = data.get("messages", [])
            conversation_text = "\n".join(
                f"{m.get('speaker', '?').upper()}: {m.get('text', '')}"
                for m in messages_list
            )

        elif file_ext in ["txt", "csv"]:
            try:
                conversation_text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                conversation_text = raw_bytes.decode("latin-1")
            if not conversation_text.strip():
                raise ValueError("Uploaded text file is empty.")

        else:   # audio — fetch fresh transcript from port-8000 after it finishes
            print("Waiting for audio transcript sync…")
            time.sleep(6)

            # ── PRIMARY: fetch from port-8000 /get-transcript with retry ────
            # getaddrinfo / connection errors are transient on loopback —
            # retry up to 3 times with a short back-off before giving up.
            transcript_fetched = False
            AUDIO_API_URL      = "http://127.0.0.1:8000/get-transcript"
            MAX_RETRIES        = 3
            RETRY_DELAY        = 2   # seconds between retries

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    resp = httpx.get(AUDIO_API_URL, timeout=10)
                    if resp.status_code == 200:
                        entries = resp.json()          # [{speaker, text, start?}, …]
                        if entries:
                            rows = [
                                f"{e.get('speaker', 'Speaker 00')}: {e.get('text', '')}"
                                for e in entries if str(e.get("text", "")).strip()
                            ]
                            if rows:
                                conversation_text  = "\n".join(rows)
                                transcript_fetched = True
                                print(f"  Fetched {len(entries)} turns from port-8000 "
                                      f"(attempt {attempt})")
                                break
                        else:
                            print(f"  Port-8000 returned empty transcript "
                                  f"(attempt {attempt}), waiting…")
                    else:
                        print(f"  Port-8000 returned HTTP {resp.status_code} "
                              f"(attempt {attempt}), waiting…")
                except (httpx.ConnectError,
                        httpx.TimeoutException,
                        httpx.NetworkError,
                        OSError) as fetch_err:
                    # Catches getaddrinfo failed, connection refused, timeouts
                    print(f"  Port-8000 fetch error (attempt {attempt}): {fetch_err}")

                if not transcript_fetched and attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

            # ── FALLBACK: CSV written by port-8000 transcription pipeline ───
            if not transcript_fetched:
                print("  All port-8000 retries failed — trying CSV fallback…")
                if os.path.exists(TRANSCRIPT_FILE):
                    try:
                        df   = pd.read_csv(TRANSCRIPT_FILE)
                        rows = []
                        for _, row in df.iterrows():
                            spk  = str(row.get("speaker",
                                        row.get("Speaker", "Speaker 00"))).strip()
                            text = str(row.get("text",
                                        row.get("Text", ""))).strip()
                            if text:
                                rows.append(f"{spk}: {text}")
                        if rows:
                            conversation_text  = "\n".join(rows)
                            transcript_fetched = True
                            print(f"  Loaded {len(rows)} rows from CSV fallback")
                        else:
                            print("  CSV exists but is empty.")
                    except Exception as csv_err:
                        print(f"  CSV read failed: {csv_err}")

            if not transcript_fetched:
                raise RuntimeError(
                    "Could not retrieve audio transcript from port-8000 or CSV. "
                    "Make sure the audio transcription server is running on port 8000."
                )

        if not conversation_text.strip():
            raise ValueError("No conversation content to analyse.")

        return await _run_scoring(conversation_text, file.filename, messages_list or None)

    except Exception as e:
        print("ERROR:", str(e))
        err = _ERROR_RESULT(str(e))
        with open(SCORES_FILE, "w") as f:
            json.dump(err, f, indent=4)
        return err


# ==============================================================================
# SHARED SCORING CORE  (used by both /analyze-quality and /analyze-quality-direct)
# ==============================================================================

async def _run_scoring(conversation_text: str, filename: str,
                       messages_list: list | None = None) -> dict:
    """
    Run the full scoring pipeline on already-resolved conversation text.
    Returns the result dict and also persists it to disk + history.
    """
    if not conversation_text.strip():
        raise ValueError("No conversation content to analyse.")

    conversation_text = normalize_speakers(conversation_text)
    print(f"  Normalised transcript ({len(conversation_text)} chars, "
          f"{len([l for l in conversation_text.splitlines() if l.strip()])} lines)")

    print("Running bias analysis…")
    bias_scores = compute_bias_scores(conversation_text)
    print(f"  bias → {bias_scores}")

    if messages_list:
        eff, avg_rt = efficiency_score_from_messages(messages_list)
    else:
        eff, avg_rt = estimate_efficiency_from_text(conversation_text)
    print(f"  efficiency={eff}, avg_response_time={avg_rt}s")

    anon = re.sub(r"\b\d{6,}\b", "[ACCOUNT]", conversation_text)
    anon = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", anon)
    anon = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", anon)
    llm_input = anon[:8000]

    print("Running LLM evaluation…")
    # ── RAG: fetch relevant policy chunks for this conversation ───────────────
    policy_chunks = await _fetch_policy_context(conversation_text)
    policy_used   = len(policy_chunks) > 0
    if policy_used:
        print(f"  [RAG] Injecting {len(policy_chunks)} policy chunks into prompt")
    else:
        print("  [RAG] No policy loaded — using generic rubrics")

    result_json = await call_llama_async(build_prompt(llm_input, policy_chunks))
    result_json["policy_context_used"] = policy_used
    result_json["policy_chunks_count"] = len(policy_chunks)

    result_json["efficiency_score"]  = eff
    result_json["avg_response_time"] = avg_rt
    result_json["bias"]              = bias_scores
    result_json = enrich_emotion(result_json)

    with open(SCORES_FILE, "w") as f:
        json.dump(result_json, f, indent=4)

    _append_to_history(result_json, filename)

    log_keys = ["empathy", "compliance", "resolution", "efficiency_score",
                "customer_emotion", "customer_satisfaction",
                "emotion_confidence", "satisfaction_confidence"]
    print("SUCCESS →", {k: result_json.get(k) for k in log_keys if k in result_json})
    return result_json


_ERROR_RESULT = lambda msg: {
    "empathy": 0, "compliance": 0, "resolution": 0, "efficiency": 0,
    "efficiency_score": 0, "avg_response_time": 0.0,
    "empathy_timeline":    [{"stage": "Start", "score": 0}, {"stage": "Mid", "score": 0}, {"stage": "End", "score": 0}],
    "compliance_steps":    [{"step": "ID Verify", "score": 0}, {"step": "Protocol", "score": 0}, {"step": "Closing", "score": 0}],
    "resolution_progress": [{"stage": "Discovery", "score": 0}, {"stage": "Fixing", "score": 0}, {"stage": "Solved", "score": 0}],
    "customer_emotion": "Neutral",      "customer_emotion_emoji":      "😐",
    "customer_satisfaction": "Neutral", "customer_satisfaction_emoji": "😐",
    "emotion_confidence": 0,            "satisfaction_confidence": 0,
    "satisfaction_percentage": 0,
    "bias": {"name_neutrality": 0, "language_neutrality": 0,
             "tone_consistency": 0, "equal_effort": 0, "overall_fairness": 0},
    "reasoning": f"Analysis failed: {msg}",
}


# ==============================================================================
# NEW: Direct transcript scoring endpoint
# Frontend passes transcript rows it already has — no port-8000 fetch, no races
# ==============================================================================

from pydantic import BaseModel

class TranscriptRow(BaseModel):
    speaker: str
    text: str
    start: float | None = None

class DirectScoreRequest(BaseModel):
    filename: str
    transcript: list[TranscriptRow]


@app.post("/analyze-quality-direct")
async def analyze_quality_direct(req: DirectScoreRequest):
    """
    Score a call using a transcript the frontend already has.
    Accepts { filename, transcript: [{speaker, text, start?}, ...] }.
    Eliminates the port-8000 fetch race condition entirely.
    """
    print(f"\n--- DIRECT SCORING: {req.filename} ({len(req.transcript)} turns) ---")
    try:
        rows = [f"{r.speaker}: {r.text}" for r in req.transcript if r.text.strip()]
        if not rows:
            raise ValueError("Transcript is empty.")
        return await _run_scoring("\n".join(rows), req.filename)
    except Exception as e:
        print("ERROR:", str(e))
        err = _ERROR_RESULT(str(e))
        with open(SCORES_FILE, "w") as f:
            json.dump(err, f, indent=4)
        return err


# ── In-memory scoring job store (mirrors transcription job pattern) ──────────
_SCORE_JOBS: dict = {}


class BatchScoreRequest(BaseModel):
    files: list[DirectScoreRequest]


@app.post("/score-batch")
async def score_batch(req: BatchScoreRequest):
    """
    Submit a batch of transcripts for async quality scoring.
    Returns a score_job_id immediately — poll /score-job/{id} for progress.
    Groq calls are semaphore-limited (GROQ_CONCURRENCY=3 default) with
    exponential backoff on rate limits — safe for 100 calls on free tier.
    """
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(req.files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 files per batch.")

    job_id = str(uuid.uuid4())[:8]
    _SCORE_JOBS[job_id] = {
        "status":  "queued",
        "total":   len(req.files),
        "done":    0,
        "failed":  0,
        "results": [],
    }

    async def _run_score_job():
        job = _SCORE_JOBS[job_id]
        job["status"] = "running"

        async def score_one(item: DirectScoreRequest):
            try:
                rows = [f"{r.speaker}: {r.text}" for r in item.transcript if r.text.strip()]
                if not rows:
                    raise ValueError("Empty transcript")
                result = await _run_scoring("\n".join(rows), item.filename)
                job["results"].append({"filename": item.filename, "status": "success", "scores": result})
                job["done"] += 1
                print(f"[SCORE JOB {job_id}] {job['done']}/{job['total']} — {item.filename}")
            except Exception as e:
                job["results"].append({"filename": item.filename, "status": "error", "error": str(e)})
                job["failed"] += 1
                print(f"[SCORE JOB {job_id}] FAILED — {item.filename}: {e}")

        # All fire — semaphore inside call_llama_async caps real concurrency
        await asyncio.gather(*[score_one(item) for item in req.files])
        job["status"] = "complete"
        print(f"[SCORE JOB {job_id}] Complete — {job['done']} ok, {job['failed']} failed")

    asyncio.create_task(_run_score_job())
    print(f"[SCORE BATCH] Job {job_id} started — {len(req.files)} files, concurrency={GROQ_CONCURRENCY}")
    return {"score_job_id": job_id, "total": len(req.files), "status": "queued",
            "message": f"Poll /score-job/{job_id} for progress."}


@app.get("/score-job/{job_id}")
async def get_score_job(job_id: str):
    """
    Poll for scoring progress.
    Returns { score_job_id, status, total, done, failed, percent, results[] }
    When complete, results[] contains per-file scores.
    """
    job = _SCORE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Score job {job_id} not found.")
    total   = max(job["total"], 1)
    percent = round(((job["done"] + job["failed"]) / total) * 100)
    return {"score_job_id": job_id, "status": job["status"], "total": job["total"],
            "done": job["done"], "failed": job["failed"],
            "percent": percent, "results": job["results"]}


@app.get("/get-quality-scores")
async def get_scores():
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r") as f:
            return json.load(f)
    return {
        "empathy": 0, "compliance": 0, "resolution": 0, "efficiency": 0,
        "efficiency_score": 0, "avg_response_time": 0.0,
        "empathy_timeline":    [{"stage": "Start", "score": 0}, {"stage": "Mid", "score": 0}, {"stage": "End", "score": 0}],
        "compliance_steps":    [{"step": "ID Verify", "score": 0}, {"step": "Protocol", "score": 0}, {"step": "Closing", "score": 0}],
        "resolution_progress": [{"stage": "Discovery", "score": 0}, {"stage": "Fixing", "score": 0}, {"stage": "Solved", "score": 0}],
        "customer_emotion": "Neutral",      "customer_emotion_emoji":      "😐",
        "customer_satisfaction": "Neutral", "customer_satisfaction_emoji": "😐",
        "emotion_confidence": 0,            "satisfaction_confidence": 0,
        "satisfaction_percentage": 0,
        "bias": {"name_neutrality": 0, "language_neutrality": 0,
                 "tone_consistency": 0, "equal_effort": 0, "overall_fairness": 0},
        "reasoning": "No analysis data found. Please upload a file.",
    }


@app.get("/get-aggregate-scores")
async def get_aggregate_scores():
    """
    Returns scores averaged across ALL calls in history.
    The response includes a `file_count` field so the UI can show
    'Based on N calls'.  Falls back to the latest single-file scores
    if no history exists yet.
    """
    if os.path.exists(SCORES_HISTORY_FILE):
        try:
            with open(SCORES_HISTORY_FILE, "r") as f:
                history = json.load(f)
            if history:
                agg = _compute_aggregate(history)
                print(f"Aggregate served: {agg.get('file_count')} files")
                return agg
        except Exception as e:
            print(f"Aggregate error: {e}")

    # Fallback — no history yet, serve latest single-file result
    if os.path.exists(SCORES_FILE):
        with open(SCORES_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("file_count", 1)
        return data

    return {
        "empathy": 0, "compliance": 0, "resolution": 0, "efficiency": 0,
        "efficiency_score": 0, "avg_response_time": 0.0, "file_count": 0,
        "empathy_timeline":    [{"stage": "Start", "score": 0}, {"stage": "Mid", "score": 0}, {"stage": "End", "score": 0}],
        "compliance_steps":    [{"step": "ID Verify", "score": 0}, {"step": "Protocol", "score": 0}, {"step": "Closing", "score": 0}],
        "resolution_progress": [{"stage": "Discovery", "score": 0}, {"stage": "Fixing", "score": 0}, {"stage": "Solved", "score": 0}],
        "customer_emotion": "Neutral",      "customer_emotion_emoji":      "😐",
        "customer_satisfaction": "Neutral", "customer_satisfaction_emoji": "😐",
        "emotion_confidence": 0,            "satisfaction_confidence": 0,
        "satisfaction_percentage": 0,
        "bias": {"name_neutrality": 0, "language_neutrality": 0,
                 "tone_consistency": 0, "equal_effort": 0, "overall_fairness": 0},
        "reasoning": "No analysis data found. Please upload a file.",
    }


@app.post("/start-session")
async def start_session():
    """
    Call this BEFORE sending files for a new upload session (single or batch).
    Clears history so the aggregate only reflects the current session's calls,
    not every call ever uploaded since the server started.
    """
    removed = []
    for fpath in [SCORES_HISTORY_FILE, SCORES_FILE]:
        if os.path.exists(fpath):
            os.remove(fpath)
            removed.append(fpath)
    print(f"[SESSION] History cleared — fresh session started. Removed: {removed}")
    return {"status": "session_started", "removed": removed}


@app.delete("/clear-scores-history")
async def clear_scores_history():
    """Wipe history so aggregate resets to zero."""
    removed = []
    for fpath in [SCORES_HISTORY_FILE, SCORES_FILE]:
        if os.path.exists(fpath):
            os.remove(fpath)
            removed.append(fpath)
    return {"status": "cleared", "removed": removed}


@app.get("/scores-history")
async def list_scores_history():
    """Return raw per-file history (file_name, timestamp, core scores) for debugging."""
    if not os.path.exists(SCORES_HISTORY_FILE):
        return []
    try:
        with open(SCORES_HISTORY_FILE, "r") as f:
            history = json.load(f)
        return [
            {
                "file_name":  e.get("file_name", "unknown"),
                "timestamp":  e.get("timestamp", ""),
                "empathy":    e.get("empathy", 0),
                "compliance": e.get("compliance", 0),
                "resolution": e.get("resolution", 0),
                "efficiency_score": e.get("efficiency_score", 0),
                "customer_emotion":      e.get("customer_emotion", "—"),
                "customer_satisfaction": e.get("customer_satisfaction", "—"),
            }
            for e in history
        ]
    except Exception as e:
        return {"error": str(e)}
async def evaluate_chat_file(file: UploadFile = File(...)):
    """Efficiency + bias only — no LLM call."""
    try:
        data          = json.loads((await file.read()).decode("utf-8"))
        messages_list = data.get("messages", [])
        raw_text      = "\n".join(
            f"{m.get('speaker', '?').upper()}: {m.get('text', '')}"
            for m in messages_list
        )
        full_text   = normalize_speakers(raw_text)
        bias        = compute_bias_scores(full_text)
        eff, avg_rt = efficiency_score_from_messages(messages_list)
        return {
            "avg_response_time": avg_rt,
            "efficiency_score":  eff,
            "total_messages":    len(messages_list),
            "bias":              bias,
        }
    except Exception as e:
        return {"error": str(e)}


# ==============================================================================
# RAG / POLICY ENDPOINTS  (integrated directly — no separate rag_server needed)
# ==============================================================================

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    """
    Upload a policy document (PDF or TXT).

    If PINECONE_API_KEY is set and RAG libs are installed:
      → Chunks the doc, embeds locally (sentence-transformers), upserts to Pinecone.
      → All future scoring calls will retrieve relevant chunks and score compliance
        against YOUR policy instead of generic best-practice rubrics.

    If RAG is not configured:
      → Returns a clear error explaining what's missing.
    """
    if not _rag_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline not available. "
                "Ensure PINECONE_API_KEY is set in .env and RAG libraries are installed "
                "(pip install -r requirements.txt)."
            )
        )

    # Robustly extract extension — handles Windows paths like C:\...\file.pdf
    clean_name = file.filename.replace("\\", "/").split("/")[-1] if file.filename else ""
    ext = clean_name.lower().rsplit(".", 1)[-1] if "." in clean_name else ""
    if ext not in ("pdf", "txt", "docx"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Only PDF, TXT, and DOCX files are supported."
        )

    temp_path = os.path.join(UPLOAD_TEMP_DIR, clean_name)
    raw_bytes = await file.read()
    with open(temp_path, "wb") as fh:
        fh.write(raw_bytes)

    print(f"[RAG] Processing policy: {file.filename} ({len(raw_bytes):,} bytes)")

    try:
        # 1. Load document
        loader   = PyPDFLoader(temp_path) if ext == "pdf" else TextLoader(temp_path, encoding="utf-8")
        raw_docs = loader.load()
        full_text = " ".join(d.page_content for d in raw_docs)
        print(f"[RAG] Loaded {len(raw_docs)} page(s), {len(full_text):,} chars")

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        if not chunks:
            raise ValueError("Document produced no usable text chunks.")
        print(f"[RAG] Split into {len(chunks)} chunks")

        # 3. Embed locally
        texts   = [c.page_content for c in chunks]
        loop    = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, lambda: _embed(texts))
        print(f"[RAG] Embedded {len(vectors)} chunks (dim={EMBEDDING_DIM})")

        # 4. Upsert to Pinecone
        index     = _get_pinecone_index()
        namespace = _make_namespace(clean_name)

        # Clear existing vectors in this namespace (clean replace)
        try:
            index.delete(delete_all=True, namespace=namespace)
        except Exception:
            pass  # namespace didn't exist yet

        records = [
            {
                "id":     f"{namespace}_{i}",
                "values": vectors[i],
                "metadata": {"text": texts[i], "filename": clean_name, "chunk_idx": i},
            }
            for i in range(len(vectors))
        ]
        for start in range(0, len(records), UPSERT_BATCH):
            batch = records[start : start + UPSERT_BATCH]
            await loop.run_in_executor(
                None,
                lambda b=batch: index.upsert(vectors=b, namespace=namespace)
            )
            print(f"[RAG] Upserted {start}–{start + len(batch) - 1}")

        # 5. Save metadata
        meta = {
            "filename":        clean_name,
            "namespace":       namespace,
            "uploaded_at":     datetime.now().isoformat(),
            "chunk_count":     len(chunks),
            "char_count":      len(full_text),
            "embedding_model": EMBEDDING_MODEL,
            "pinecone_index":  PINECONE_INDEX_NAME,
        }
        _save_policy_meta(meta)

        print(f"[RAG] Done — namespace='{namespace}', {len(chunks)} vectors in Pinecone")
        return {
            "status":    "success",
            "filename":  clean_name,
            "namespace": namespace,
            "chunks":    len(chunks),
            "message":   (
                f"Policy '{clean_name}' indexed into {len(chunks)} chunks. "
                f"Scoring will now check compliance against this policy."
            ),
        }

    except Exception as e:
        print(f"[RAG] Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/policy-status")
async def policy_status():
    """
    Returns the currently active policy info and live Pinecone vector count.
    The UI can poll this to show whether RAG-based scoring is active.
    """
    if not _rag_available():
        return {
            "loaded":   False,
            "rag_ready": False,
            "message":  "RAG not configured (PINECONE_API_KEY missing or libs not installed).",
        }

    meta = _load_policy_meta()
    if not meta:
        return {"loaded": False, "rag_ready": True,
                "message": "No policy uploaded yet. Upload a PDF or TXT to enable policy-aware scoring."}

    # Live vector count from Pinecone
    vector_count = 0
    try:
        index  = _get_pinecone_index()
        stats  = index.describe_index_stats()
        ns_info = stats.get("namespaces", {}).get(meta.get("namespace", ""), {})
        vector_count = ns_info.get("vector_count", 0)
    except Exception:
        pass

    return {
        "loaded":          True,
        "rag_ready":       True,
        "filename":        meta.get("filename"),
        "namespace":       meta.get("namespace"),
        "uploaded_at":     meta.get("uploaded_at"),
        "chunk_count":     meta.get("chunk_count", 0),
        "char_count":      meta.get("char_count", 0),
        "vector_count":    vector_count,
        "embedding_model": meta.get("embedding_model"),
        "pinecone_index":  meta.get("pinecone_index"),
    }


@app.get("/list-policies")
async def list_policies():
    """Lists all uploaded policy namespaces in Pinecone. Useful for switching between policies."""
    if not _rag_available():
        raise HTTPException(status_code=503, detail="RAG not configured.")
    try:
        index      = _get_pinecone_index()
        stats      = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        active_ns  = _load_policy_meta().get("namespace", "")
        return {
            "policies": [
                {
                    "namespace":    ns,
                    "vector_count": info.get("vector_count", 0),
                    "active":       ns == active_ns,
                }
                for ns, info in namespaces.items()
            ],
            "active_namespace": active_ns,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/switch-policy/{namespace}")
async def switch_policy(namespace: str):
    """
    Switch the active policy to a previously uploaded namespace.
    No re-upload needed — vectors already exist in Pinecone.
    """
    if not _rag_available():
        raise HTTPException(status_code=503, detail="RAG not configured.")
    try:
        index = _get_pinecone_index()
        stats = index.describe_index_stats()
        if namespace not in stats.get("namespaces", {}):
            raise HTTPException(status_code=404, detail=f"Namespace '{namespace}' not found.")
        meta = _load_policy_meta()
        meta["namespace"] = namespace
        _save_policy_meta(meta)
        return {"status": "switched", "active_namespace": namespace}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-policy")
async def clear_policy():
    """Delete the active policy namespace from Pinecone and clear local metadata."""
    meta      = _load_policy_meta()
    namespace = meta.get("namespace", "")
    if namespace and _rag_available():
        try:
            index = _get_pinecone_index()
            index.delete(delete_all=True, namespace=namespace)
            print(f"[RAG] Deleted namespace '{namespace}' from Pinecone")
        except Exception as e:
            print(f"[RAG] Could not delete namespace: {e}")
    if os.path.exists(POLICY_META_FILE):
        os.remove(POLICY_META_FILE)
    return {"status": "cleared", "namespace": namespace}


@app.get("/alerts")
async def get_alerts():
    """
    Evaluates ONLY the current session's scores and returns compliance alerts.

    Source priority:
      1. SCORES_HISTORY_FILE — used when batch was run (multiple files this session)
      2. SCORES_FILE         — used for single-file uploads (latest call only)

    This means alerts always match what the Reports page is showing right now.

    Severity (strict less-than — score AT threshold is NOT an alert):
      critical  — score < 4   → immediate action required
      warning   — score < 6   → needs attention
      info      — score < 8   → minor improvement needed
    """
    alerts = []

    # ── Thresholds (strict less-than boundaries) ───────────────────────────────
    THRESHOLDS = {
        "compliance":               {"critical": 4, "warning": 6, "info": 8},
        "empathy":                  {"critical": 4, "warning": 6, "info": 8},
        "resolution":               {"critical": 4, "warning": 6, "info": 8},
        "efficiency_score":         {"critical": 4, "warning": 6, "info": 8},
        "bias.overall_fairness":    {"critical": 4, "warning": 6, "info": 8},
        "bias.language_neutrality": {"critical": 4, "warning": 6, "info": 8},
        "bias.tone_consistency":    {"critical": 4, "warning": 6, "info": 8},
    }

    ALERT_META = {
        "compliance": {
            "category": "Compliance",
            "critical": ("Critical Compliance Failure", "Agent severely failed to follow required protocols. Immediate review needed.",
                         "🔴 Escalate to supervisor immediately. Pull call recording and conduct 1-on-1 coaching session."),
            "warning":  ("Compliance Gap Detected",    "Agent missed key compliance steps. Training recommended.",
                         "🟡 Schedule targeted compliance training. Review ID verification and closing scripts with agent."),
            "info":     ("Minor Compliance Issue",     "Agent skipped some steps.",
                         "🔵 Remind agent of full protocol checklist in next team meeting."),
        },
        "empathy": {
            "category": "Empathy",
            "critical": ("Very Low Empathy",   "Agent showed little to no empathy. Customer experience severely impacted.",
                         "🔴 Mandatory empathy coaching required. Use call recording as training example."),
            "warning":  ("Low Empathy Score",  "Agent empathy was below standard.",
                         "🟡 Coach agent on active listening and personalisation. Practice empathy phrases."),
            "info":     ("Empathy Needs Work", "Agent empathy could be improved.",
                         "🔵 Encourage agent to use customer name and acknowledge feelings more consistently."),
        },
        "resolution": {
            "category": "Resolution",
            "critical": ("Issue Unresolved",       "Customer issue was not addressed. Escalation may be needed.",
                         "🔴 Follow up with customer immediately. Review why agent could not resolve and escalate if needed."),
            "warning":  ("Poor Resolution Rate",   "Agent failed to resolve the customer issue properly.",
                         "🟡 Coach agent on problem-solving steps. Review resolution workflow and escalation triggers."),
            "info":     ("Incomplete Resolution",  "Issue was only partially resolved.",
                         "🔵 Remind agent to confirm resolution with customer before closing."),
        },
        "efficiency_score": {
            "category": "Efficiency",
            "critical": ("Very Inefficient Call",  "Call was excessively long and circular. Process review needed.",
                         "🔴 Identify root cause — knowledge gap or process issue. Provide call handling framework."),
            "warning":  ("Low Efficiency",         "Call had unnecessary turns and repetition.",
                         "🟡 Coach agent on concise communication. Review call structure and use of knowledge base."),
            "info":     ("Efficiency Improvement", "Call could have been resolved more concisely.",
                         "🔵 Suggest agent use standard scripts to reduce unnecessary back-and-forth."),
        },
        "bias.overall_fairness": {
            "category": "Fairness",
            "critical": ("Critical Fairness Violation", "Severe bias or unfair treatment detected. Immediate review required.",
                         "🔴 Mandatory bias and inclusion training. HR review may be required."),
            "warning":  ("Fairness Concern",            "Agent showed signs of inconsistent treatment.",
                         "🟡 Discuss with agent. Review call recording and coach on consistent, neutral behaviour."),
            "info":     ("Fairness Improvement",        "Minor fairness issues detected.",
                         "🔵 Reinforce equal-effort standards in next team review."),
        },
        "bias.language_neutrality": {
            "category": "Language",
            "critical": ("Prohibited Language Detected", "Agent used dismissive or non-neutral language.",
                         "🔴 Immediate coaching on prohibited phrases. Flag call for compliance record."),
            "warning":  ("Language Neutrality Issue",    "Agent used potentially dismissive language.",
                         "🟡 Share approved language guide with agent. Highlight specific phrases to avoid."),
            "info":     ("Language Suggestion",          "Agent language could be more neutral.",
                         "🔵 Suggest agent replace informal/assumption-based phrasing with neutral alternatives."),
        },
        "bias.tone_consistency": {
            "category": "Tone",
            "critical": ("Inconsistent Tone",    "Agent tone was unprofessional throughout the call.",
                         "🔴 Immediate tone and professionalism training. Review full call recording."),
            "warning":  ("Tone Issue Detected",  "Agent did not maintain required professional tone.",
                         "🟡 Coach agent on maintaining consistent tone from greeting to closing."),
            "info":     ("Tone Improvement",     "Agent tone was inconsistent at some points.",
                         "🔵 Remind agent about required greeting and closing phrases per script."),
        },
    }

    def _get_severity(key: str, score: float) -> str | None:
        t = THRESHOLDS.get(key, {})
        # Strict less-than: score AT the threshold is NOT an alert
        if score < t.get("critical", 0): return "critical"
        if score < t.get("warning",  0): return "warning"
        if score < t.get("info",     0): return "info"
        return None

    def _make_alert(key: str, score: float, file_name: str = "", idx: int = 0) -> dict | None:
        severity = _get_severity(key, score)
        if not severity:
            return None
        meta  = ALERT_META.get(key, {})
        parts = meta.get(severity, ("Issue Detected", "Score below threshold.", "Review this area."))
        title, message, action = parts if len(parts) == 3 else (*parts, "Review this area.")
        return {
            "id":        f"{key}_{file_name}_{idx}",
            "severity":  severity,
            "category":  meta.get("category", key),
            "title":     title,
            "message":   message,
            "action":    action,
            "score":     round(score, 1),
            "threshold": THRESHOLDS[key][severity],
            "file_name": file_name,
        }

    def _extract_scores_from_entry(entry: dict) -> list[tuple[str, float]]:
        pairs = []
        for key in ["compliance", "empathy", "resolution", "efficiency_score"]:
            val = entry.get(key)
            if isinstance(val, (int, float)) and val > 0:
                pairs.append((key, float(val)))
        bias = entry.get("bias", {})
        if isinstance(bias, dict):
            for bkey in ["overall_fairness", "language_neutrality", "tone_consistency"]:
                val = bias.get(bkey)
                if isinstance(val, (int, float)) and val > 0:
                    pairs.append((f"bias.{bkey}", float(val)))
        return pairs

    # ── Use SCORES_HISTORY_FILE if it exists (current session / batch) ──────────
    # This file is cleared at /start-session so it always reflects THIS session only
    if os.path.exists(SCORES_HISTORY_FILE):
        try:
            with open(SCORES_HISTORY_FILE, "r") as f:
                history = json.load(f)
            for idx, entry in enumerate(history):
                fname = entry.get("file_name", f"file_{idx+1}")
                for key, score in _extract_scores_from_entry(entry):
                    alert = _make_alert(key, score, fname, idx)
                    if alert:
                        alerts.append(alert)
        except Exception as e:
            print(f"[ALERTS] History read error: {e}")

    # ── Fallback: SCORES_FILE only (single upload, no history yet) ──────────────
    if not alerts and os.path.exists(SCORES_FILE):
        try:
            with open(SCORES_FILE, "r") as f:
                entry = json.load(f)
            fname = entry.get("file_name", "latest")
            for key, score in _extract_scores_from_entry(entry):
                alert = _make_alert(key, score, fname, 0)
                if alert:
                    alerts.append(alert)
        except Exception as e:
            print(f"[ALERTS] Scores file read error: {e}")

    # Sort: critical → warning → info, then by score ascending within same severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 3), a["score"]))

    # Deduplicate: if same key+file appears multiple times keep the worst (lowest score)
    seen: dict = {}
    deduped = []
    for a in alerts:
        dedup_key = f"{a['category']}_{a['file_name']}"
        if dedup_key not in seen or a["score"] < seen[dedup_key]:
            seen[dedup_key] = a["score"]
            deduped.append(a)
    alerts = deduped

    critical_count = sum(1 for a in alerts if a["severity"] == "critical")
    warning_count  = sum(1 for a in alerts if a["severity"] == "warning")

    print(f"[ALERTS] {len(alerts)} alerts — {critical_count} critical, {warning_count} warning")
    return {
        "alerts":         alerts,
        "total":          len(alerts),
        "critical_count": critical_count,
        "warning_count":  warning_count,
        "has_critical":   critical_count > 0,
    }



@app.get("/health")
async def health_check():
    meta = _load_policy_meta()
    return {
        "status":           "ok",
        "rag_available":    _rag_available(),
        "policy_loaded":    _policy_loaded(),
        "active_namespace": meta.get("namespace", "none") if meta else "none",
        "pinecone_index":   PINECONE_INDEX_NAME if _rag_available() else "n/a",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)