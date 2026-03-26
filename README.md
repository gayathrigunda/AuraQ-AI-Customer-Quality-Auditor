# 🎯 AuraQ — AI Customer Quality Auditor

> Automate your customer support QA process with AI-powered transcription, scoring, emotion detection, and audit intelligence.

![Version](https://img.shields.io/badge/version-v1.0.0-blue)
![Status](https://img.shields.io/badge/status-Live-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Sprint](https://img.shields.io/badge/sprints-4-orange)
![Team](https://img.shields.io/badge/team-3%20members-purple)

🔗 **Live Demo:** [aura-q-ai-customer-quality-auditor-eight.vercel.app](https://aura-q-ai-customer-quality-auditor-eight.vercel.app)

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Team](#team)
- [License](#license)

---

## 📖 About the Project

**AuraQ** is an AI-powered Customer Quality Auditor that automates the evaluation of customer-agent interactions. It accepts both audio call recordings and text/chat logs, analyses them using advanced AI models, and delivers deep quality insights through an interactive dashboard.

Built using **Agile methodology** across **4 sprints (8 weeks)** by a team of 3 developers.

---

## ❗ Problem Statement

Customer support organisations rely on manual QA processes where supervisors randomly sample only **5–10% of interactions**. This leads to:

- ❌ Inconsistent and biased agent evaluations
- ❌ Missed compliance violations
- ❌ No real-time visibility into customer sentiment
- ❌ Inability to make data-driven decisions at scale

**AuraQ solves this by automating 100% of QA reviews with AI.**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Audio Transcription | Upload call recordings — auto-transcribed with speaker diarisation (Agent/Customer split) using Deepgram Nova-2 |
| 💬 Chat Log Analysis | Upload text/chat logs — dynamically parsed and summarised using Deepgram Text Intelligence |
| 📊 Quality Scoring | Empathy, Compliance, Resolution scored 1–10 using Groq LLaMA 3.1 8B |
| 😊 Emotion Detection | Customer emotion detected across 8 categories with 0–100% satisfaction score |
| ⚖️ Fairness Analysis | Bias/fairness analysis with 4 sub-scores for equitable agent evaluation |
| 🧠 RAG Intelligence | LangChain + Pinecone for contextual transcript retrieval and policy-aware scoring |
| 📁 Policy Document Upload | Upload company policy docs — agents scored against real policies via RAG pipeline |
| 📈 React Dashboard | Interactive charts, call history, batch upload, and IST timestamps |
| 📄 Report Generation | Downloadable PDF and DOC audit reports per call |
| 📱 Mobile Responsive | Fully responsive UI with bottom navigation for mobile access |
| ☁️ Cloud Deployed | Frontend on Vercel, Backend on Render with UptimeRobot monitoring |

---

## 🛠️ Tech Stack

### Frontend
- **React + Vite** — Single Page Application (SPA)
- **Tailwind CSS** — Utility-first responsive styling
- **Recharts** — Interactive data visualisation

### Backend
- **Python + FastAPI** — Unified REST API (main.py)

### AI / ML Services
- **Deepgram Nova-2** — Audio transcription + speaker diarisation
- **Deepgram Text Intelligence** — Chat summarisation
- **Groq LLaMA 3.1 8B** — Quality scoring + emotion detection
- **LangChain + Pinecone** — RAG pipeline for contextual retrieval

### Deployment & Monitoring
- **Vercel** — Frontend hosting
- **Render** — Backend hosting
- **UptimeRobot** — 5-minute uptime monitoring

---

## 📁 Project Structure

```
-Customer-support-quality-auditor-/
│
├── clean-project/                          # React + Vite Frontend (SPA)
│   ├── src/
│   │   ├── components/
│   │   │   ├── AppLayout.tsx               # Main layout wrapper
│   │   │   ├── DashboardHeader.tsx         # Header with search & notifications
│   │   │   ├── Sidebar.tsx                 # Navigation sidebar
│   │   │   ├── ScoreCards.tsx              # Quality score display cards
│   │   │   ├── ScoreRadarChart.tsx         # Radar chart visualization
│   │   │   ├── ScoreTrendChart.tsx         # Trend chart visualization
│   │   │   ├── StatsOverview.tsx           # Statistics overview component
│   │   │   ├── SummaryCard.tsx             # Summary card component
│   │   │   ├── TranscriptViewer.tsx        # Transcript display viewer
│   │   │   ├── UploadPanel.tsx             # File upload interface
│   │   │   ├── AuditHistory.tsx            # Call history component
│   │   │   ├── NavLink.tsx                 # Navigation link component
│   │   │   └── ui/                         # shadcn/ui components (30+ components)
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx               # Main dashboard page
│   │   │   ├── Calls.tsx                   # Calls history page
│   │   │   ├── Reports.tsx                 # Reports page
│   │   │   ├── Landing.tsx                 # Landing page
│   │   │   └── NotFound.tsx                # 404 page
│   │   ├── hooks/
│   │   │   ├── use-toast.ts                # Toast notification hook
│   │   │   └── use-mobile.tsx              # Mobile detection hook
│   │   ├── lib/
│   │   │   ├── mock-data.ts                # Mock data for development
│   │   │   └── utils.ts                    # Utility functions
│   │   ├── test/
│   │   │   ├── example.test.ts             # Example tests
│   │   │   └── setup.ts                    # Test setup
│   │   ├── App.tsx                         # Root app component
│   │   ├── main.tsx                        # React entry point
│   │   ├── index.css                       # Global styles
│   │   └── vite-env.d.ts                   # Vite environment types
│   ├── public/
│   │   └── robots.txt                      # SEO robots file
│   ├── index.html                          # HTML entry point
│   ├── package.json                        # Dependencies & scripts
│   ├── vite.config.ts                      # Vite configuration (port: 8081)
│   ├── tsconfig.json                       # TypeScript config
│   ├── tailwind.config.ts                  # Tailwind CSS config
│   ├── eslint.config.js                    # ESLint configuration
│   ├── playwright.config.ts                # E2E test config
│   ├── vitest.config.ts                    # Unit test config
│   ├── postcss.config.js                   # PostCSS config
│   ├── components.json                     # Component registry
│  
│
├── customer_support/                       # Python FastAPI Backend
│   ├── main.py                             # Unified FastAPI server entry
│   ├── app.py                              # Audio processing & transcription
│   ├── chat_app.py                         # Chat/text analysis
│   ├── scoring_server.py                   # LLM quality scoring service
│   ├── requirements.txt                    # Python dependencies
│   ├── calls/                              # Processed call data storage
│   ├── quality_scores.json                 # Per-call quality scores
│   ├── quality_scores_history.json         # Historical score tracking
│   ├── final_summaries.csv                 # Exported AI summaries
│   ├── text_summaries.csv                  # Text analysis summaries
│   ├── text_transcript.csv                 # Text transcriptions
│   ├── transcriptions_with_speakers.csv    # Speaker-diarized transcriptions
│   ├── processing_status.json              # Current processing status
│   ├── chat.txt                            # Sample chat logs
│   ├── human_chat.txt                      # Human chat sample
│   ├── .env                                # Environment variables
│   ├── .git/                               # Git repository
│   ├── package-lock.json                   # NPM lock file
│   └── __pycache__/                        # Python cache (ignored)
│
├── newenv/                                 # Python virtual environment
│   ├── Scripts/                            # Executables (activate, deactivate)
│   ├── Lib/                                # Installed packages
│   └── Include/                            # Package headers
│
├── bin/                                    # Utility scripts
├── AuraQ_Agile_Document.xlsx               # Agile sprint documentation
├── package.json                            # Root-level npm config
├── package-lock.json                       # Dependency lock file
├── requirements.txt                        # Root Python requirements
├── .env                                    # Environment variables (ignored)
├── .gitignore                              # Git ignore rules
├── .git/                                   # Git repository
├── clean.txt                               # Project cleanup notes
├── quality_scores.json                     # Root-level scores backup
└── README.md                               # This file
```

### 📂 Key Directories

| Directory | Purpose |
|---|---|
| `clean-project/src/components` | Reusable React UI components |
| `clean-project/src/pages` | Page-level components for routing |
| `customer_support/` | FastAPI backend with AI/ML services |
| `customer_support/batch_transcripts/` | Audio files awaiting processing |
| `customer_support/rag_uploads/` | Policy documents for RAG context |
| `newenv/` | Python virtual environment with dependencies |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:
- **Node.js** (v18+)
- **Python** (v3.10+)
- **pip**

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Siddhi-Dhamal/-Customer-support-quality-auditor-.git
cd -Customer-support-quality-auditor-
```

---

### 2️⃣ Setup Backend

The backend consists of **3 separate FastAPI servers** running on different ports. Open **3 separate terminals** and run each command:

#### Terminal 1 — Audio Transcription Server (Port 8000)
```bash
cd customer_support
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

#### Terminal 2 — Chat Analysis Server (Port 8001)
```bash
cd customer_support
uvicorn chat_app:app --reload --port 8001
```

#### Terminal 3 — Quality Scoring Server (Port 8002)
```bash
cd customer_support
uvicorn scoring_server:app --reload --port 8002
```

**Service Ports:**
- 🎙️ Audio Service: `http://localhost:8000`
- 💬 Chat Service: `http://localhost:8001`
- 📊 Scoring Service: `http://localhost:8002`

**Main Unified API** (mounts all 3 services):
```bash
cd customer_support
uvicorn main:app --reload --port 8000  # (optional, combines all 3)
```

Main API: `http://localhost:8000` with endpoints at `/audio`, `/chat`, `/scoring` prefixes

---

### 3️⃣ Setup Frontend

```bash
cd clean-project
npm install
npm run dev
```

Frontend runs at: `http://localhost:8081/`

---

## 🔐 Environment Variables

Create a `.env` file in the root directory or `customer_support/` folder and add the following:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
PINECONE_INDEX=your_pinecone_index_name
```

> ⚠️ Never commit your `.env` file to GitHub. Add it to `.gitignore`.

---

## 📡 API Endpoints

> **Access Methods:**
> - **Standalone**: Call each service directly on its port (e.g., `http://localhost:8000`, `http://localhost:8001`, `http://localhost:8002`)
> - **Endpoint paths** below show routes WITHOUT the service prefix (add `/audio`, `/chat`, or `/scoring` when using unified API)

### 🎙️ Audio Transcription Endpoints (Port 8000)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Upload single audio file for transcription |
| POST | `/upload-batch` | Upload batch of audio files |
| GET | `/batch-transcripts` | Get list of all batch transcripts |
| GET | `/batch-transcript/{csv_filename}` | Get specific batch transcript |
| GET | `/get-transcript` | Get latest transcript |
| GET | `/get-summary` | Get latest summary |
| GET | `/history` | Get audio processing history |
| GET | `/health` | Audio service health check |

### 💬 Chat/Text Analysis Endpoints (Port 8001)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-text` | Upload text/chat log for analysis |
| GET | `/get-text-transcript` | Get text transcript |
| GET | `/get-text-summary` | Get text summary |
| GET | `/history` | Get chat processing history |
| GET | `/health` | Chat service health check |

### 📊 Quality Scoring Endpoints (Port 8002)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/analyze-quality` | Analyze quality of transcript/chat |
| POST | `/analyze-quality-direct` | Direct quality analysis |
| POST | `/score-batch` | Score a batch of files |
| GET | `/score-job/{job_id}` | Get scoring job status |
| GET | `/get-quality-scores` | Get quality scores |
| GET | `/get-aggregate-scores` | Get aggregate quality scores |
| POST | `/start-session` | Start a new scoring session |
| GET | `/scores-history` | Get scoring history |
| DELETE | `/clear-scores-history` | Clear score history |
| GET | `/alerts` | Get quality compliance alerts |

### 📋 Policy/RAG Endpoints (Port 8002)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-policy` | Upload policy document for RAG |
| GET | `/policy-status` | Get current policy status |
| GET | `/list-policies` | List all available policies |
| POST | `/switch-policy/{namespace}` | Switch to different policy |
| DELETE | `/clear-policy` | Clear current policy |

### 🏥 Health & System Endpoints

| Method | Endpoint | Full URL |
|---|---|---|
| GET | `/health` | `http://localhost:8000/health` (Audio) |
| GET | `/health` | `http://localhost:8001/health` (Chat) |
| GET | `/health` | `http://localhost:8002/health` (Scoring) |

---

## 👥 Team

| Name | Role |
|---|---|
| **Gunda Lakshmi Gayathri** | Full-Stack Developer — Text server, emotion detection, dashboard, reports, deployment |
| **Siddhi Dhamal** | AI/ML Engineer — Audio server, Speaker diarization, Summary generation, batch transcribing, LLM scoring, RAG pipeline, Render deployment |
| **Sowmya** | QA & Integration — Fairness analysis, charts, call history, monitoring |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Deepgram](https://deepgram.com) — Audio transcription API
- [Groq](https://groq.com) — Ultra-fast LLM inference
- [LangChain](https://langchain.com) — RAG pipeline framework
- [Pinecone](https://pinecone.io) — Vector database
- [Vercel](https://vercel.com) — Frontend deployment
- [Render](https://render.com) — Backend deployment

---

<p align="center">Made with ❤️ by Team AuraQ</p>

