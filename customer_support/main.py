from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the 'app' objects from your existing files
from app import app as audio_app
from chat_app import app as chat_app
from scoring_server import app as scoring_app

# Create the main master app
app = FastAPI()

# Add CORS so your frontend can talk to it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your sub-apps to different "folders"
app.mount("/audio", audio_app)
app.mount("/chat", chat_app)
app.mount("/scoring", scoring_app)

@app.get("/")
async def health_check():
    return {"status": "all systems go", "services": ["audio", "chat", "scoring"]}