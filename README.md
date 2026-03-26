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
AuraQ/
│
├── frontend/                  # React + Vite SPA
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Dashboard, History, Reports
│   │   └── App.jsx            # Main app entry
│   ├── public/
│   └── package.json
│
├── backend/                   # FastAPI Backend
│   ├── main.py                # Unified server entry point
│   ├── app.py                 # Audio transcription server
│   ├── chat_app.py            # Text/chat analysis server
│   ├── scoring/               # LLM scoring modules
│   ├── rag/                   # LangChain + Pinecone RAG pipeline
│   ├── file_scores/           # Per-file quality scores storage
│   └── requirements.txt
│
└── README.md
```

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
git clone https://github.com/your-username/auraq.git
cd auraq
```

---

### 2️⃣ Setup Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at: `http://localhost:8000`

---

### 3️⃣ Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## 🔐 Environment Variables

Create a `.env` file in the `backend/` folder and add the following:

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

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload-audio` | Upload audio call file for transcription |
| POST | `/upload-chat` | Upload text/chat log for summarisation |
| GET | `/scores/{file_id}` | Get quality scores for a specific file |
| GET | `/history` | Get call history with timestamps |
| POST | `/upload-policy` | Upload policy document to RAG pipeline |
| GET | `/rag-summary/{file_id}` | Get RAG-powered audit summary |
| GET | `/report/{file_id}` | Download PDF/DOC report |

---

## 👥 Team

| Name | Role |
|---|---|
| **Gunda Lakshmi Gayathri** | Full-Stack Developer — Text server, emotion detection, dashboard, reports, deployment |
| **Siddhi Dhamal** | AI/ML Engineer — Audio server, LLM scoring, RAG pipeline, Render deployment |
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
