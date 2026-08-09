"""
AI Interview Agent — FastAPI backend
Implements POST /api/interview per technical-spec.md

Session flow:
  1. First request (candidate present)      -> init session, welcome reply
  2. Subsequent requests (message present)   -> conversational turn, tracks
                                                 turn_count + days_covered
  3. Once turn_count >= 8 AND len(days_covered) >= 4 -> final structured
                                                          feedback, done=True

LLM calls go through the Groq SDK. All conversation state (per-turn
messages + session metadata) is persisted in a local SQLite database
(`interview_memory.db`) so the agent has full context on every turn and
survives process restarts — no external DB, no in-memory dict.

Dependencies: fastapi, pydantic, python-dotenv, groq
  pip install fastapi pydantic python-dotenv groq
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
import sqlite3
import logging
from typing import Optional, List, Dict, Any, Set

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from groq import Groq

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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

DB_PATH = os.environ.get("INTERVIEW_DB_PATH", "interview_memory.db")

REQUIRED_TURNS = 8
REQUIRED_DAYS = 4

# ---------------------------------------------------------------------------
# Load curriculum once at startup (used to ground the interviewer's questions)
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
# SQLite — conversation memory + session metadata
# ---------------------------------------------------------------------------
#
# `messages`: the append-only conversation log per session_id, in the
#             role/content shape the Groq API expects.
# `sessions`: small metadata row per session_id (candidate profile,
#             which curriculum days have been covered, turn count) so
#             the interview's progress also survives a restart.

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                candidate_json TEXT NOT NULL,
                days_covered_json TEXT NOT NULL DEFAULT '[]',
                turn_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


def _save_message_sync(session_id: str, role: str, content: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
    finally:
        conn.close()


async def save_message(session_id: str, role: str, content: str) -> None:
    await run_in_threadpool(_save_message_sync, session_id, role, content)


def _get_history_sync(session_id: str) -> List[Dict[str, str]]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()


async def get_history(session_id: str) -> List[Dict[str, str]]:
    """Full conversation history for session_id, oldest first, in the
    {"role": "user"/"assistant", "content": "..."} shape the Groq API wants."""
    return await run_in_threadpool(_get_history_sync, session_id)


def _create_session_sync(session_id: str, candidate: Dict[str, Any]) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, candidate_json, days_covered_json, turn_count)
            VALUES (?, ?, '[]', 0)
            """,
            (session_id, json.dumps(candidate)),
        )
        conn.commit()
    finally:
        conn.close()


async def create_session(session_id: str, candidate: Dict[str, Any]) -> None:
    await run_in_threadpool(_create_session_sync, session_id, candidate)


def _get_session_sync(session_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "candidate": json.loads(row["candidate_json"]),
            "days_covered": set(json.loads(row["days_covered_json"])),
            "turn_count": row["turn_count"],
        }
    finally:
        conn.close()


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return await run_in_threadpool(_get_session_sync, session_id)


def _update_session_sync(session_id: str, days_covered: Set[int], turn_count: int) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE sessions SET days_covered_json = ?, turn_count = ? WHERE session_id = ?",
            (json.dumps(sorted(days_covered)), turn_count, session_id),
        )
        conn.commit()
    finally:
        conn.close()


async def update_session(session_id: str, days_covered: Set[int], turn_count: int) -> None:
    await run_in_threadpool(_update_session_sync, session_id, days_covered, turn_count)


# ---------------------------------------------------------------------------
# Groq helper
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_groq_client() -> Optional[Groq]:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not set")
            return None
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _call_groq_sync(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Optional[str]:
    client = _get_groq_client()
    if client is None:
        return None
    try:
        # Groq's chat.completions API takes the system prompt as a normal
        # message with role "system", followed by the full conversation.
        formatted_messages = [{"role": "system", "content": system}] + messages
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception:
        logger.exception("Unexpected error calling Groq API")
        return None


async def call_groq(
    system: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 600,
) -> Optional[str]:
    """
    Calls the Groq chat completions API, off the event loop. Returns the
    text of the response, or None on any failure (caller must handle the
    fallback).
    """
    return await run_in_threadpool(_call_groq_sync, system, messages, max_tokens)


def parse_llm_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction from the model's text response. Handles the
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
        logger.error("Could not parse JSON from model response: %s", raw[:300])
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(
    candidate: Dict[str, Any],
    days_covered: Set[int],
) -> str:
    member = candidate["member"]
    missions = candidate.get("missions", [])

    completed_days = [
        m["day"] for m in missions if m.get("passed") and m["day"] in CURRICULUM_DAYS
    ]
    day_titles = {d: CURRICULUM_DAYS[d]["title"] for d in completed_days}

    covered_str = ", ".join(str(d) for d in sorted(days_covered)) or "none yet"

    return f"""You are a technical interviewer conducting a live conversational
interview for {member['name']}, a {member['jobRole']} with {member['yearsExperience']}
years of experience, evaluating their AI Cohort curriculum work.

Candidate completed these curriculum days: {sorted(completed_days)}
Day titles: {json.dumps(day_titles)}

Days already covered by questions so far this interview: {covered_str}

You have the full conversation transcript above. Before writing your next
message, evaluate the candidate's most recent answer:
- Was it specific and technically sound, vague, or off-target?
- Did they reference concrete details (tools, tradeoffs, numbers, code) or
  just restate the question in general terms?

Use that evaluation to decide what to do next — do NOT just advance through
a fixed list of topics:
- If the answer was shallow, vague, or dodged the question: ask a sharper
  follow-up on the SAME topic that pushes for specifics (e.g. "can you
  walk me through exactly what happened when X failed?").
- If the answer was strong: probe a related edge case, a tradeoff they
  didn't mention, or ask them to compare it to an alternative approach.
- Only move on to a new curriculum day (prioritizing days NOT already in
  the "covered" list above) once the current topic feels sufficiently
  explored.
- Keep questions conversational and natural, referencing what they just
  said when relevant, not robotic or scripted.

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"reply": "<your question or conversational response>", "day_covered": <int or null>}}

Set "day_covered" to the curriculum day number your reply is questioning
them about. Use null if your reply is a same-topic follow-up that isn't
tied to a new day, or is not about a specific day."""


def build_feedback_prompt(
    candidate: Dict[str, Any],
    history: List[Dict[str, str]],
    days_covered: Set[int],
) -> str:
    member = candidate["member"]
    transcript = "\n".join(
        f"{'Interviewer' if h['role'] == 'assistant' else 'Candidate'}: {h['content']}"
        for h in history
    )

    return f"""You are evaluating {member['name']}'s technical interview transcript
below. Curriculum days covered: {sorted(days_covered)}.

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
# Fallbacks (used if the LLM call/parse fails, so the route never crashes)
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
        candidate_dict = req.candidate.model_dump()
        await create_session(session_id, candidate_dict)

        first_reply = (
            f"Welcome {req.candidate.member.name}. Let's begin your technical "
            f"interview on the AI Cohort curriculum."
        )
        await save_message(session_id, "assistant", first_reply)
        return InterviewResponse(reply=first_reply, done=False)

    # 2. Ongoing Conversation Turn
    session = await get_session(session_id)
    if session is None:
        raise HTTPException(status_code=400, detail="Invalid or uninitialized sessionId")

    candidate = session["candidate"]
    days_covered: Set[int] = session["days_covered"]
    turn_count: int = session["turn_count"] + 1
    user_message = req.message or ""

    # Persist the user's turn, then pull the FULL history back out of
    # SQLite so the model sees exactly what's on record — this is the
    # single source of truth for conversation memory.
    await save_message(session_id, "user", user_message)
    full_history = await get_history(session_id)

    # ---- Termination check: >= 8 turns AND >= 4 distinct curriculum days ----
    if turn_count >= REQUIRED_TURNS and len(days_covered) >= REQUIRED_DAYS:
        feedback_prompt = build_feedback_prompt(candidate, full_history, days_covered)
        raw = await call_groq(
            system="You are a precise technical evaluator. Output only valid JSON.",
            messages=[{"role": "user", "content": feedback_prompt}],
            max_tokens=800,
        )
        parsed = parse_llm_json(raw)

        if parsed:
            try:
                feedback = FeedbackSchema(**parsed)
            except Exception:
                logger.exception("Feedback JSON did not match schema: %s", parsed)
                feedback = fallback_feedback()
        else:
            feedback = fallback_feedback()

        final_reply = "Thank you for completing the interview. Here is your structured feedback."

        await save_message(session_id, "assistant", final_reply)
        await update_session(session_id, days_covered, turn_count)

        return InterviewResponse(reply=final_reply, done=True, feedback=feedback)

    # ---- Standard conversational turn ----
    system_prompt = build_system_prompt(candidate, days_covered)
    raw = await call_groq(system=system_prompt, messages=full_history)
    parsed = parse_llm_json(raw)

    if parsed and "reply" in parsed:
        reply_text = parsed["reply"]
        day_covered = parsed.get("day_covered")
    else:
        fb = fallback_reply(days_covered)
        reply_text = fb["reply"]
        day_covered = fb["day_covered"]

    if isinstance(day_covered, int) and day_covered in CURRICULUM_DAYS:
        days_covered.add(day_covered)

    await save_message(session_id, "assistant", reply_text)
    await update_session(session_id, days_covered, turn_count)

    return InterviewResponse(reply=reply_text, done=False)