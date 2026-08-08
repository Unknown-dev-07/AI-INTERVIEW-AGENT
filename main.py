"""
AI Interview Agent — FastAPI backend
Implements POST /api/interview per technical-spec.md

Session flow:
  1. First request (candidate present)      -> init session, welcome reply
  2. Subsequent requests (message present)   -> conversational turn, tracks
                                                 turn_count + days_covered
  3. Once turn_count >= 8 AND len(days_covered) >= 4 -> final structured
                                                          feedback, done=True

External calls (Claude + Breeth) are both wrapped in try/except so the route
never crashes mid-interview; on failure we fall back to a safe default.
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
import logging
from typing import Optional, List, Dict, Any, Set

import requests
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("interview-agent")

app = FastAPI()
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount your frontend directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

BREETH_API_BASE = os.environ.get("BREETH_API_BASE", "https://api.thebreeth.com")
BREETH_API_KEY = os.environ.get("BREETH_API_KEY", "")

REQUIRED_TURNS = 8
REQUIRED_DAYS = 4

# In-memory session store (fine for hackathon MVP; swap for Redis/SQLite later)
sessions: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Load curriculum once at startup (used to ground Claude's questions)
# ---------------------------------------------------------------------------

try:
    with open("curriculum.json") as f:
        CURRICULUM = json.load(f)
    CURRICULUM_DAYS = {d["day"]: d for d in CURRICULUM.get("days", [])}
except FileNotFoundError:
    logger.warning("curriculum.json not found — proceeding with empty curriculum")
    CURRICULUM = {}
    CURRICULUM_DAYS = {}


# ---------------------------------------------------------------------------
# Pydantic schemas (unchanged from spec)
# ---------------------------------------------------------------------------

class CandidateMember(BaseModel):
    id: str
    name: str
    jobRole: str
    yearsExperience: int
    education: str
    status: str


class Mission(BaseModel):
    day: int
    title: str
    passed: Optional[bool] = None
    skipped: Optional[bool] = None
    attempts: Optional[int] = None


class CandidateProfile(BaseModel):
    member: CandidateMember
    missions: List[Mission]
    signals: Dict[str, int]


class InterviewRequest(BaseModel):
    sessionId: str
    candidate: Optional[CandidateProfile] = None
    message: Optional[str] = None


class FeedbackSchema(BaseModel):
    summary: str
    strengths: List[str]
    gaps: List[str]
    next: List[str]


class InterviewResponse(BaseModel):
    reply: str
    done: bool
    feedback: Optional[FeedbackSchema] = None


# ---------------------------------------------------------------------------
# Claude helper
# ---------------------------------------------------------------------------

def _call_claude_sync(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Optional[str]:
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not set")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        # Groq/OpenAI format includes the system prompt inside the messages list
        formatted_messages = [{"role": "system", "content": system}] + messages

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=formatted_messages,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception:
        logger.exception("Unexpected error calling Groq API")
        return None
    except requests.Timeout:
        logger.error("Claude API call timed out")
        return None
    except requests.HTTPError as e:
        logger.error("Claude API HTTP error: %s — %s", e.response.status_code, e.response.text)
        return None
    except Exception:
        logger.exception("Unexpected error calling Claude API")
        return None


async def call_claude(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 600,
) -> Optional[str]:
    """
    Calls the Anthropic Messages API via `requests`, off the event loop.
    Returns the text of the first text block, or None on any failure
    (caller must handle the fallback).
    """
    return await run_in_threadpool(_call_claude_sync, system, messages, max_tokens)


def parse_claude_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction from a Claude text response. Handles the
    common case where the model wraps JSON in a code fence.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to locate the outermost { ... } block
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        logger.error("Could not parse JSON from Claude response: %s", raw[:300])
        return None


# ---------------------------------------------------------------------------
# Breeth helper (memory store)
# ---------------------------------------------------------------------------
def _breeth_add_episode_sync(user_content: str, assistant_content: str, session_id: str) -> bool:
    return True


async def breeth_add_episode(
    user_content: str,
    session_id: str,
    assistant_content: str = "",
) -> bool:
    """
    Writes a turn to Breeth per the documented /v1/episodes schema.
    Returns True/False for success; never raises.
    """
    return await run_in_threadpool(
        _breeth_add_episode_sync, user_content, assistant_content, session_id
    )


def _breeth_search_sync(query: str, session_id: str, limit: int) -> List[str]:
        return []

   

async def breeth_search(query: str, session_id: str, limit: int = 5) -> List[str]:
    """
    Retrieves prior context from Breeth per the documented /v1/search
    schema. Returns a list of text snippets (possibly empty) — never raises.
    """
    return await run_in_threadpool(_breeth_search_sync, query, session_id, limit)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(
    candidate: Dict[str, Any],
    days_covered: Set[int],
    breeth_context: List[str],
) -> str:
    member = candidate["member"]
    missions = candidate.get("missions", [])

    completed_days = [
        m["day"] for m in missions if m.get("passed") and m["day"] in CURRICULUM_DAYS
    ]
    day_titles = {d: CURRICULUM_DAYS[d]["title"] for d in completed_days}

    context_block = ""
    if breeth_context:
        joined = "\n".join(f"- {c}" for c in breeth_context)
        context_block = f"\nRelevant prior context from memory:\n{joined}\n"

    covered_str = ", ".join(str(d) for d in sorted(days_covered)) or "none yet"

    return f"""You are a technical interviewer conducting a live conversational
interview for {member['name']}, a {member['jobRole']} with {member['yearsExperience']}
years of experience, evaluating their AI Cohort curriculum work.

Candidate completed these curriculum days: {sorted(completed_days)}
Day titles: {json.dumps(day_titles)}
{context_block}
Days already covered by questions so far this interview: {covered_str}

Your task each turn:
- Ask ONE focused technical question about a curriculum day the candidate
  completed, prioritizing days NOT already in the "covered" list above.
- Keep questions conversational, not robotic.
- Base follow-ups on the candidate's previous answer when relevant.

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"reply": "<your question or conversational response>", "day_covered": <int or null>}}

Set "day_covered" to the curriculum day number your reply is questioning
them about. Use null only if your reply is not about a specific day
(e.g. a generic follow-up)."""


def build_feedback_prompt(
    candidate: Dict[str, Any],
    history: List[Dict[str, str]],
    days_covered: Set[int],
    breeth_context: List[str],
) -> str:
    member = candidate["member"]
    transcript = "\n".join(
        f"{'Interviewer' if h['role'] == 'assistant' else 'Candidate'}: {h['content']}"
        for h in history
    )
    context_block = ""
    if breeth_context:
        joined = "\n".join(f"- {c}" for c in breeth_context)
        context_block = f"\nAdditional context from memory:\n{joined}\n"

    return f"""You are evaluating {member['name']}'s technical interview transcript
below. Curriculum days covered: {sorted(days_covered)}.
{context_block}
Transcript:
{transcript}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{
  "summary": "<2-3 sentence overall summary>",
  "strengths": ["<concise point>", "..."],
  "gaps": ["<concise point>", "..."],
  "next": ["<concise actionable recommendation>", "..."]
}}"""


# ---------------------------------------------------------------------------
# Fallbacks (used if Claude call/parse fails, so the route never crashes)
# ---------------------------------------------------------------------------

def fallback_reply(days_covered: Set[int]) -> Dict[str, Any]:
    remaining = [d for d in CURRICULUM_DAYS if d not in days_covered]
    day = remaining[0] if remaining else None
    title = CURRICULUM_DAYS.get(day, {}).get("title", "your recent work")
    return {
        "reply": f"Could you walk me through your approach to {title}?",
        "day_covered": day,
    }


def fallback_feedback() -> FeedbackSchema:
    return FeedbackSchema(
        summary="The interview completed, but automated feedback generation "
                "was unavailable. Please review the transcript manually.",
        strengths=["Transcript available for manual review"],
        gaps=["Automated evaluation could not be generated"],
        next=["Re-run evaluation or review responses manually"],
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.post("/api/interview", response_model=InterviewResponse)
async def handle_interview(req: InterviewRequest):
    session_id = req.sessionId

    # 1. Start Interview Turn
    if req.candidate is not None:
        sessions[session_id] = {
            "candidate": req.candidate.model_dump(),
            "turn_count": 0,
            "days_covered": set(),
            "history": [],
        }

        await breeth_add_episode(
            user_content=(
                f"Interview started for candidate {req.candidate.member.name} "
                f"({req.candidate.member.jobRole})."
            ),
            assistant_content="Interview session initialized.",
            session_id=session_id,
        )

        first_reply = (
            f"Welcome {req.candidate.member.name}. Let's begin your technical "
            f"interview on the AI Cohort curriculum."
        )
        sessions[session_id]["history"].append(
            {"role": "assistant", "content": first_reply}
        )
        return InterviewResponse(reply=first_reply, done=False)

    # 2. Ongoing Conversation Turn
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid or uninitialized sessionId")

    session = sessions[session_id]
    candidate = session["candidate"]
    user_message = req.message or ""

    # Grab the last assistant reply before we mutate history, so the Breeth
    # write reflects "candidate said X in response to Y".
    prior_assistant_reply = ""
    for h in reversed(session["history"]):
        if h["role"] == "assistant":
            prior_assistant_reply = h["content"]
            break

    session["history"].append({"role": "user", "content": user_message})
    session["turn_count"] += 1

    await breeth_add_episode(
        user_content=user_message,
        assistant_content=prior_assistant_reply,
        session_id=session_id,
    )

    days_covered: Set[int] = session["days_covered"]
    turn_count: int = session["turn_count"]

    # ---- Termination check: >= 8 turns AND >= 4 distinct curriculum days ----
    if turn_count >= REQUIRED_TURNS and len(days_covered) >= REQUIRED_DAYS:
        breeth_context = await breeth_search(
            query="interview strengths gaps performance", session_id=session_id
        )
        feedback_prompt = build_feedback_prompt(
            candidate, session["history"], days_covered, breeth_context
        )
        raw = await call_claude(
            system="You are a precise technical evaluator. Output only valid JSON.",
            messages=[{"role": "user", "content": feedback_prompt}],
            max_tokens=800,
        )
        parsed = parse_claude_json(raw)

        if parsed:
            try:
                feedback = FeedbackSchema(**parsed)
            except Exception:
                logger.exception("Feedback JSON did not match schema: %s", parsed)
                feedback = fallback_feedback()
        else:
            feedback = fallback_feedback()

        final_reply = "Thank you for completing the interview. Here is your structured feedback."

        await breeth_add_episode(
            user_content=f"Interview completed for {candidate['member']['name']}.",
            assistant_content=f"Final summary: {feedback.summary}",
            session_id=session_id,
        )

        return InterviewResponse(reply=final_reply, done=True, feedback=feedback)

    # ---- Standard conversational turn ----
    breeth_context = await breeth_search(
        query=user_message or "candidate interview turn", session_id=session_id
    )
    system_prompt = build_system_prompt(candidate, days_covered, breeth_context)

    # Give Claude the recent conversation as context (trim to last ~6 turns)
    recent_history = session["history"][-6:]
    raw = await call_claude(system=system_prompt, messages=recent_history)
    parsed = parse_claude_json(raw)

    if parsed and "reply" in parsed:
        reply_text = parsed["reply"]
        day_covered = parsed.get("day_covered")
    else:
        fb = fallback_reply(days_covered)
        reply_text = fb["reply"]
        day_covered = fb["day_covered"]

    if isinstance(day_covered, int) and day_covered in CURRICULUM_DAYS:
        days_covered.add(day_covered)

    session["history"].append({"role": "assistant", "content": reply_text})

    return InterviewResponse(reply=reply_text, done=False)
