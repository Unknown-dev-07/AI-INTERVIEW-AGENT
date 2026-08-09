# AI Development & Prompts Log

## Project: AI Interview Agent

### 1. Initial Architecture & Setup
- **Prompt:** "How do I set up a FastAPI project for an AI Interview Agent with Groq integration and a dark-themed chat frontend?"
- **Outcome:** Created `main.py` containing the FastAPI backend, asynchronous endpoints, and integrated the Groq API handler with Pydantic models.

### 2. Frontend Interface Design
- **Prompt:** "Create a clean, responsive HTML/CSS/JS single-page UI for the chat interface that communicates with the FastAPI backend."
- **Outcome:** Built `index.html` featuring a modern chat window and dynamic message rendering.

### 3. Deployment Configuration
- **Prompt:** "How do I configure this application to deploy successfully on Render with Python and pip?"
- **Outcome:** Created `requirements.txt` (`fastapi`, `uvicorn`, `requests`, `python-dotenv`, `pydantic`), configured the start command (`python -m uvicorn main:app --host 0.0.0.0 --port $PORT`), and successfully deployed.

### 4. State Management & Database Persistence
- **Prompt:** "How do I implement a local SQLite database (`interview_memory.db`) to persist the conversation history and turn count across the session without relying on in-memory dictionaries?"
- **Outcome:** Successfully mapped the session IDs to database rows, ensuring the Groq AI receives full conversation context on every turn and seamlessly tracks the user's progress through the curriculum (Days 7-28).

### 5. API Compliance & Structured Feedback
- **Prompt:** "How can I force the Groq LLM to terminate the interview upon reaching the final turn and return a strictly formatted JSON object containing arrays for strengths, gaps, and next steps?"
- **Outcome:** Refactored the `POST /api/interview` logic to dynamically switch from conversational output to a structured evaluation schema that perfectly matches the `TECHNICAL-SPECS.MD` requirements.

### 6. UI/UX Refactoring & Polish
- **Prompt:** "Rewrite the frontend HTML file using Tailwind CSS via CDN to create a modern, dark-mode (Slate/Indigo/Emerald) enterprise UI, including dynamic glassmorphic rendering for the final feedback report."
- **Outcome:** Swapped the basic frontend for a responsive, interactive UI with bouncing typing indicators and properly styled status badges, seamlessly served by the FastAPI backend.