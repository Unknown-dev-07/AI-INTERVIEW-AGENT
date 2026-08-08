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