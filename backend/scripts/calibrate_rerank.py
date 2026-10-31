"""Calibrate RERANK_MIN_SCORE against real retrieval + rerank scores.

Developer tool, read-only: it runs the app's own retrieval path (same
vector store, same in-query user_id filter) and the same Jina reranker as
production for one user, then prints per-question scores and a suggested
cutoff. It never calls Groq or any other LLM, and it never writes to the
database.

Usage (from backend/):

    python -m scripts.calibrate_rerank --email you@example.com
    python -m scripts.calibrate_rerank --email you@example.com --questions qa.json
"""

import argparse
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Document, DocumentStatus, User
from app.rag.retrieve import retrieve_scored_candidates

DEFAULT_ANSWERABLE = [
    "What are the standard working hours?",
    "How many casual leave days do employees get?",
    "When is salary paid?",
    "What is the probation policy?",
]
DEFAULT_UNANSWERABLE = [
    "Who won the cricket world cup?",
    "What is the capital of France?",
    "How do I install Python?",
]

# Extra breathing room on top of the client's own throttling so the batch
# of calibration calls stays far away from the per-minute limit.
_SLEEP_BETWEEN_QUESTIONS_SECONDS = 1.0

_CHUNK_PREVIEW_CHARS = 80
_QUESTION_PREVIEW_CHARS = 45


@dataclass
class QuestionResult:
    group: str
    question: str
    scores: list[float]  # relevance scores, highest first; empty means no candidates
    top_text: str


def look_up_user_id(email: str) -> uuid.UUID | None:
    """The id of the user with that (lowercased) email, or None."""
    statement = select(User.id).where(User.email == email).limit(1)
    with SessionLocal() as db:
        return db.scalar(statement)


def user_has_ready_documents(user_id: uuid.UUID) -> bool:
    """True when the user has at least one ingested, READY document."""
    statement = (
        select(Document.id)
        .where(Document.user_id == user_id)
        .where(Document.status == DocumentStatus.READY)
        .where(Document.chunk_count > 0)
        .limit(1)
    )
    with SessionLocal() as db:
        return db.scalar(statement) is not None


def load_questions(path: Path | None) -> dict[str, list[str]]:
    """Built-in defaults, optionally overridden by a --questions JSON file."""
    answerable = list(DEFAULT_ANSWERABLE)
    unanswerable = list(DEFAULT_UNANSWERABLE)
    if path is None:
        return {"answerable": answerable, "unanswerable": unanswerable}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    for key, fallback in (("answerable", answerable), ("unanswerable", unanswerable)):
        value = data.get(key, fallback)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"{path}: '{key}' must be a non-empty list of strings")
    return {
        "answerable": [q for q in data["answerable"]],
        "unanswerable": [q for q in data["unanswerable"]],
    }


def compute_summary(
    answerable_tops: list[float],
    unanswerable_tops: list[float],
) -> dict[str, float | bool]:
    """Compare the worst answerable top score with the best unanswerable one."""
    lowest = min(answerable_tops)
    highest = max(unanswerable_tops)
    has_gap = lowest > highest
    return {
        "lowest_answerable": lowest,
        "highest_unanswerable": highest,
        "has_gap": has_gap,
        "midpoint": (lowest + highest) / 2,
        "gap_width": lowest - highest,
    }


def format_summary(summary: dict[str, float | bool]) -> str:
    lowest = summary["lowest_answerable"]
    highest = summary["highest_unanswerable"]
    lines = [
        f"Lowest top score among answerable questions: {lowest:.4f}",
        f"Highest top score among unanswerable questions: {highest:.4f}",
    ]
    if summary["has_gap"]:
        lines.append(
            f"Clean gap. Suggested RERANK_MIN_SCORE = {summary['midpoint']:.4f} "
            f"(gap width {summary['gap_width']:.4f})"
        )
    else:
        lines.append(
            "Groups overlap. Choose a cutoff below the lowest answerable score "
            f"({lowest:.4f}) and accept some false positives (highest unanswerable "
            f"score {highest:.4f})."
        )
    return "\n".join(lines)


def format_table(rows: list[QuestionResult]) -> str:
    header = (
        f"{'GROUP':<14}{'QUESTION':<{_QUESTION_PREVIEW_CHARS}}{'TOP':>9}{'NEXT1':>9}"
        f"{'NEXT2':>9}  TOP CHUNK (first {_CHUNK_PREVIEW_CHARS} chars)"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        question = row.question[:_QUESTION_PREVIEW_CHARS].replace("\n", " ")
        group = row.group[:13]
        if row.scores:
            top = f"{row.scores[0]:.4f}"
            next1 = f"{row.scores[1]:.4f}" if len(row.scores) > 1 else "-"
            next2 = f"{row.scores[2]:.4f}" if len(row.scores) > 2 else "-"
            chunk = row.top_text[:_CHUNK_PREVIEW_CHARS].replace("\n", " ")
        else:
            top = next1 = next2 = "(none)"
            chunk = ""
        lines.append(
            f"{group:<14}{question:<{_QUESTION_PREVIEW_CHARS}}{top:>9}{next1:>9}{next2:>9}  {chunk}"
        )
    return "\n".join(lines)


def run_groups(
    questions: dict[str, list[str]],
    user_id: uuid.UUID,
) -> list[QuestionResult]:
    """Run every question through the production retrieval + rerank path."""
    rows: list[QuestionResult] = []
    first = True
    for group in ("answerable", "unanswerable"):
        for question in questions[group]:
            if not first:
                time.sleep(_SLEEP_BETWEEN_QUESTIONS_SECONDS)
            first = False
            scored, _ = retrieve_scored_candidates(question, user_id)
            ordered = sorted(
                scored,
                key=lambda pair: pair[1] if pair[1] is not None else float("-inf"),
                reverse=True,
            )
            rows.append(
                QuestionResult(
                    group=group,
                    question=question,
                    scores=[score for _, score in ordered],
                    top_text=ordered[0][0].text if ordered else "",
                )
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print Jina rerank scores for test questions to calibrate RERANK_MIN_SCORE."
    )
    parser.add_argument("--email", required=True, help="Email of an existing user (no password needed)")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help='Optional JSON file: {"answerable": [...], "unanswerable": [...]}',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    email = args.email.strip().lower()
    try:
        user_id = look_up_user_id(email)
    except Exception as error:  # noqa: BLE001 - operator-facing tool
        print(f"Database error while looking up the user: {type(error).__name__}", file=sys.stderr)
        return 1
    if user_id is None:
        print(f"No user found for email '{email}'.", file=sys.stderr)
        return 1

    try:
        has_ready = user_has_ready_documents(user_id)
    except Exception as error:  # noqa: BLE001
        print(f"Database error while checking documents: {type(error).__name__}", file=sys.stderr)
        return 1
    if not has_ready:
        print(
            f"User '{email}' has no ready documents. Upload and wait for "
            "ingestion before calibrating.",
            file=sys.stderr,
        )
        return 1

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not load questions file: {error}", file=sys.stderr)
        return 1

    try:
        rows = run_groups(questions, user_id)
    except Exception as error:  # noqa: BLE001
        print(f"Retrieval error: {type(error).__name__}", file=sys.stderr)
        return 1

    if any(score is None for row in rows for score in row.scores):
        print(
            "\nJina rerank scores are unavailable — RERANK_ENABLED is off or the "
            "rerank call failed (see the warning above). Calibration needs real scores.",
            file=sys.stderr,
        )
        return 1

    answerable_tops = [row.scores[0] for row in rows if row.group == "answerable" and row.scores]
    unanswerable_tops = [row.scores[0] for row in rows if row.group == "unanswerable" and row.scores]
    if not answerable_tops or not unanswerable_tops:
        print(
            "Not enough scored rows: at least one answerable and one unanswerable "
            "question need candidates.",
            file=sys.stderr,
        )
        return 1

    print(format_table(rows))
    print()
    print(format_summary(compute_summary(answerable_tops, unanswerable_tops)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
