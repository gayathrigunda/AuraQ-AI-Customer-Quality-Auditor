import os
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import the individual app modules
import app as audio_module          # audio transcription server
import chat_app as chat_module      # text/chat analysis server
import scoring_server as scoring_module  # LLM scoring + RAG server

# ── Lifespan: pre-warm RAG at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(main_app: FastAPI):
    """
    Run the scoring_server's startup logic (embed model + Pinecone pre-warm)
    before the combined server starts accepting requests.
    """
    print("[AuraQ] Starting combined server...")

    # Pre-warm RAG from scoring_server if available
    if hasattr(scoring_module, '_rag_available') and scoring_module._rag_available():
        print("[AuraQ] Pre-warming RAG pipeline...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, scoring_module._get_embed_model)
        await loop.run_in_executor(None, scoring_module._get_pinecone_index)
        print("[AuraQ] RAG pipeline ready.")

    print("[AuraQ] Combined server is ready!")
    yield
    print("[AuraQ] Shutting down...")

# ── Create main app ────────────────────────────────────────────────────────────
main_app = FastAPI(
    title="AuraQ Combined Server",
    version="1.0.0",
    lifespan=lifespan
)

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ───────────────────────────────────────────────────────────────
@main_app.get("/")
@main_app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "AuraQ Combined Server",
        "modules": ["audio", "chat", "scoring+RAG"]
    }

# ── Skip these system paths when copying routes ────────────────────────────────
SKIP_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

# ── Mount routes from audio module (app.py) ────────────────────────────────────
for route in audio_module.app.routes:
    if hasattr(route, "path") and route.path not in SKIP_PATHS:
        main_app.router.routes.append(route)

# ── Mount routes from chat module (chat_app.py) ───────────────────────────────
for route in chat_module.app.routes:
    if hasattr(route, "path") and route.path not in SKIP_PATHS:
        main_app.router.routes.append(route)

# ── Mount routes from scoring module (scoring_server.py) ──────────────────────
for route in scoring_module.app.routes:
    if hasattr(route, "path") and route.path not in SKIP_PATHS:
        main_app.router.routes.append(route)

app = main_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
