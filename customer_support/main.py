import os
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
    async with scoring_module.lifespan(application):
        yield

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
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)