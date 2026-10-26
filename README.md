# EchoBook 📖
**"Living Memory, One Story at a Time"**

EchoBook is a creative storyteller agent that conducts warm, emotionally intelligent voice interviews with elderly people—and generates a fully illustrated, narrated life story book in real-time as they speak. 

Every story becomes a page. Every memory becomes an illustrated scene. The book builds itself simultaneously alongside the conversation.

*Built for the "Creative Storyteller" Hackathon track.*

## 🌟 The Problem
53 million Americans provide unpaid care for a family member with dementia or age-related cognitive decline. Reminiscence therapy is clinically proven to reduce anxiety and maintain identity, but creating physical "life story books" is prohibitively time-consuming and expensive. 

Families want to preserve their loved ones' stories, but the interview and transcription process is emotionally draining and intimidating.

## 🚀 The Solution
EchoBook replaces the barrier of manual transcription and illustration with an empathetic AI experience:
1. **Emotionally Intelligent Agent:** Powered by Gemini Live API, the agent listens, paces itself for elderly speakers, and asks poignant follow-up questions.
2. **Real-Time Generation:** As a memory is spoken, our pipeline extracts the vignette, prompts Imagen (or Pollinations API), and visually inserts the memory into a living book.
3. **Session Continuity:** Books are persisted in Firestore. The agent remembers past sessions and guides the storyteller through an ongoing journey.

## ⚙️ How We Built It
- **Frontend**: Next.js (React) application with TypeScript and Tailwind CSS. Features dynamic audio visualizers and a real-time "living book" interface.
- **Backend API**: Python FastAPI application coordinating real-time WebSockets, LLM prompting (Gemini Live), and database operations.
- **Voice Intelligence**: **Gemini Live API (`gemini-2.5-flash-native-audio-latest`)** for streaming audio-to-audio empathic interviews.
- **Illustration**: **Pollinations API** used for generating nostalgic, watercolor-style illustrations dynamically.
- **Authentication**: **Firebase Auth** for secure user login and session management.
- **Database**: **Google Cloud Firestore** to persist storyteller sessions and illustrated stories.

## 🏃‍♂️ How to Run the App (Locally)

### Requirements
- Python 3.12+
- Node.js 18+
- Gemini API Key ([Get it here](https://aistudio.google.com/apikey))
- Firebase Project (for Authentication and Firestore)

### 1. Set Up Environment

#### Backend
Copy `backend/.env.example` to `backend/.env` and fill in:
```bash
GEMINI_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

#### Frontend
Create `frontend/.env.local` and add your Firebase configuration:
```env
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

### 2. Start the Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080
```

### 3. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to view the app.

## ☁️ Deployment
The backend can be deployed to Google Cloud Run, and the frontend to Vercel or similar platforms.


---
*Created over 48 hours. Let's make sure no memory is ever lost.*
