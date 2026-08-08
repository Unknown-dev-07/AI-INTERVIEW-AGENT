"""
test_interview.py — simulates a full interview for CAND-001 against the
FastAPI app in main.py, without hitting real Claude/Breeth APIs.

Run:
    pip install fastapi httpx requests pytest
    pytest test_interview.py -v

What this verifies:
  1. Session init returns the welcome reply, done=False.
  2. After 8 turns where Claude reports >=4 distinct days_covered, the
     route returns done=True with a well-formed FeedbackSchema.
  3. If turn_count hits 8 but fewer than 4 distinct days were covered,
     the interview does NOT terminate yet (spec: BOTH conditions required).
  4. If the Claude call fails entirely (simulated network error), the
     route still returns a 200 with a fallback reply instead of crashing.
  5. If the Breeth calls fail entirely, the interview still proceeds
     normally (Breeth is best-effort, not a hard dependency).
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
import main  # noqa: E402

client = TestClient(main.app)

CANDIDATES_PATH = Path(__file__).parent / "candidates.json"


def load_cand_001():
    with open(CANDIDATES_PATH) as f:
        data = json.load(f)
    return next(c for c in data["candidates"] if c["member"]["id"] == "CAND-001")


# ---------------------------------------------------------------------------
# Fakes for external calls
# ---------------------------------------------------------------------------

def fake_claude_cycling_days(days_cycle):
    """
    Returns an async fake for call_claude that cycles through a fixed list
    of days, so we can deterministically control days_covered growth.
    """
    state = {"i": 0}

    async def _fake(system, messages, max_tokens=600):
        day = days_cycle[state["i"] % len(days_cycle)]
        state["i"] += 1
        return json.dumps({
            "reply": f"Tell me about curriculum day {day}.",
            "day_covered": day,
        })

    return _fake


async def fake_claude_feedback(system, messages, max_tokens=800):
    return json.dumps({
        "summary": "Sarah demonstrated strong command of embeddings, vector "
                    "databases, and agent orchestration.",
        "strengths": ["Clear grasp of embeddings (Day 7)", "Solid MCP knowledge (Day 23)"],
        "gaps": ["Limited depth on deployment tooling"],
        "next": ["Review container orchestration before production rollout"],
    })


async def fake_claude_always_fails(system, messages, max_tokens=600):
    return None  # simulates a network failure / bad response


async def fake_breeth_add_episode(user_content, session_id, assistant_content=""):
    return True


async def fake_breeth_search(query, session_id, limit=5):
    return ["Candidate previously discussed embeddings with confidence."]


async def fake_breeth_always_fails(*args, **kwargs):
    return False if "add_episode" in "" else []  # unused, see targeted patches below


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_session_start_returns_welcome():
    candidate = load_cand_001()
    resp = client.post("/api/interview", json={
        "sessionId": "test-session-1",
        "candidate": candidate,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is False
    assert "Sarah Johnson" in body["reply"]


def test_full_interview_terminates_with_feedback():
    """
    8 turns, cycling through 4 distinct days (7, 8, 10, 12) so both
    termination conditions (turn_count>=8, len(days_covered)>=4) are met
    on the final turn.
    """
    candidate = load_cand_001()
    session_id = "test-session-terminate"

    client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})

    days_cycle = [7, 8, 10, 12, 7, 8, 10, 12]  # 4 distinct days across 8 turns
    fake_claude = fake_claude_cycling_days(days_cycle)

    with patch.object(main, "call_claude", side_effect=[
        None  # placeholder, replaced turn-by-turn below
    ]):
        pass  # see loop below; patching per-call is cleaner than side_effect list

    responses = []
    with patch.object(main, "breeth_add_episode", fake_breeth_add_episode), \
         patch.object(main, "breeth_search", fake_breeth_search):

        for turn in range(8):
            is_last = (turn == 7)
            claude_fn = fake_claude_feedback if is_last else fake_claude
            with patch.object(main, "call_claude", claude_fn):
                resp = client.post("/api/interview", json={
                    "sessionId": session_id,
                    "message": f"My answer for turn {turn + 1}.",
                })
            responses.append(resp)

    final = responses[-1]
    assert final.status_code == 200
    body = final.json()

    assert body["done"] is True, "Interview should terminate on turn 8 with 4+ days covered"
    assert body["feedback"] is not None
    for field in ("summary", "strengths", "gaps", "next"):
        assert field in body["feedback"]
    assert isinstance(body["feedback"]["strengths"], list)

    session = main.sessions[session_id]
    assert session["turn_count"] == 8
    assert len(session["days_covered"]) >= 4


def test_does_not_terminate_if_days_covered_insufficient():
    """
    8 turns but Claude only ever reports day_covered=7 (same day every
    time) -> only 1 distinct day. Interview must NOT terminate even
    though turn_count reaches 8.
    """
    candidate = load_cand_001()
    session_id = "test-session-insufficient-days"

    client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})

    fake_claude_same_day = fake_claude_cycling_days([7])  # only ever day 7

    with patch.object(main, "breeth_add_episode", fake_breeth_add_episode), \
         patch.object(main, "breeth_search", fake_breeth_search), \
         patch.object(main, "call_claude", fake_claude_same_day):

        last_resp = None
        for turn in range(8):
            last_resp = client.post("/api/interview", json={
                "sessionId": session_id,
                "message": f"My answer for turn {turn + 1}.",
            })

    body = last_resp.json()
    assert body["done"] is False, "Must not terminate with only 1 distinct day covered"
    assert main.sessions[session_id]["turn_count"] == 8
    assert len(main.sessions[session_id]["days_covered"]) == 1


def test_claude_failure_falls_back_gracefully():
    """
    If call_claude returns None every time (simulated outage), the route
    must still return 200 with a usable fallback reply, never a 500.
    """
    candidate = load_cand_001()
    session_id = "test-session-claude-down"

    client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})

    with patch.object(main, "breeth_add_episode", fake_breeth_add_episode), \
         patch.object(main, "breeth_search", fake_breeth_search), \
         patch.object(main, "call_claude", fake_claude_always_fails):

        resp = client.post("/api/interview", json={
            "sessionId": session_id,
            "message": "Some answer.",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is False
    assert body["reply"], "Fallback reply must not be empty"


def test_breeth_failure_does_not_break_interview():
    """
    If Breeth add_episode/search both fail (e.g. network error / bad
    API key), the interview must proceed normally using Claude alone.
    """
    candidate = load_cand_001()
    session_id = "test-session-breeth-down"

    client.post("/api/interview", json={"sessionId": session_id, "candidate": candidate})

    async def breeth_add_fails(*args, **kwargs):
        return False

    async def breeth_search_fails(*args, **kwargs):
        return []

    fake_claude = fake_claude_cycling_days([7, 8])

    with patch.object(main, "breeth_add_episode", breeth_add_fails), \
         patch.object(main, "breeth_search", breeth_search_fails), \
         patch.object(main, "call_claude", fake_claude):

        resp = client.post("/api/interview", json={
            "sessionId": session_id,
            "message": "Some answer despite Breeth being down.",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is False
    assert "day" in body["reply"].lower() or body["reply"]


def test_unknown_session_id_returns_400():
    resp = client.post("/api/interview", json={
        "sessionId": "never-initialized",
        "message": "hello",
    })
    assert resp.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
