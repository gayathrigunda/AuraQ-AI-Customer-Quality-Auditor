# At the very top of main.py, before any other imports
import os
from fastapi import FastAPI

debug_app = FastAPI()

@debug_app.get("/")
@debug_app.get("/health") 
def health():
    return {
        "status": "ok",
        "DEEPGRAM_API_KEY": "SET" if os.environ.get("DEEPGRAM_API_KEY") else "MISSING",
        "GROQ_API_KEY": "SET" if os.environ.get("GROQ_API_KEY") else "MISSING",
        "PINECONE_API_KEY": "SET" if os.environ.get("PINECONE_API_KEY") else "MISSING",
        "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME", "MISSING"),
    }

app = debug_app  

import os
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from contextlib import asynccontextmanager

# Load .env FIRST before importing anything
load_dotenv(find_dotenv(), override=True)

# Now import modules
import app as audio_module
import chat_app as chat_module
import scoring_server as scoring_module

@asynccontextmanager
async def lifespan(application: FastAPI):
    # ✅ Start heavy loading in background — don't block port binding
    asyncio.create_task(_background_startup(application))
    yield

async def _background_startup(application):
    try:
        async with scoring_module.lifespan(application):
            # Keep the context alive — this will run until server shuts down
            await asyncio.Event().wait()
    except Exception as e:
        print(f"[STARTUP] Background init error: {e}")

main_app = FastAPI(lifespan=lifespan)

main_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@main_app.get("/")
@main_app.get("/health")
async def health():
    return {"status": "ok", "service": "AuraQ Combined Server"}

for route in audio_module.app.routes:
    if hasattr(route, "path") and route.path not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        main_app.router.routes.append(route)

for route in chat_module.app.routes:
    if hasattr(route, "path") and route.path not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        main_app.router.routes.append(route)

for route in scoring_module.app.routes:
    if hasattr(route, "path") and route.path not in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        main_app.router.routes.append(route)

app = main_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)