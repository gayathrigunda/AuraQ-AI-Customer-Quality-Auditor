import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRouter

# Import routers from each app
import app as audio_module
import chat_app as chat_module
import scoring_server as scoring_module

main_app = FastAPI()

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



# Mount all routes from each module's app
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)


