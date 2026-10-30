"""Ephemeral conversation state and exact answer caching."""

from __future__ import annotations

from dataclasses import dataclass, field
import secrets
import time
from typing import Any, Callable


def _tokens(text: str) -> int:
    # A character ceiling is conservative for the bundled WordPiece reference tokenizer.
    return len(text)


@dataclass
class Exchange:
    question: str
    answer: str

    def text(self) -> str:
        return f"User: {self.question}\nAssistant: {self.answer}"


@dataclass
class Session:
    id: str
    expires_at: float
    recent: list[Exchange] = field(default_factory=list)
    summary: str = ""


class SessionStore:
    """In-process, session-only history; process restart clears all sessions."""

    def __init__(self, *, idle_seconds: int = 1800, recent_exchanges: int = 6,
                 history_tokens: int = 4000, summary_tokens: int = 1000,
                 max_sessions: int = 256,
                 token_counter: Callable[[str], int] = _tokens,
                 summarizer: Callable[[str, list[Exchange]], str] | None = None,
                 clock: Callable[[], float] = time.time):
        self.idle_seconds = idle_seconds
        self.recent_exchanges = recent_exchanges
        self.history_tokens = history_tokens
        self.summary_tokens = summary_tokens
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self.token_counter = token_counter
        self.summarizer = summarizer
        self.clock = clock
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        now = self.clock()
        self._prune(now)
        if len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda sid: self._sessions[sid].expires_at)
            del self._sessions[oldest]
        session = Session(secrets.token_urlsafe(32), now + self.idle_seconds)
        self._sessions[session.id] = session
        return session

    def _prune(self, now: float) -> None:
        for session_id in [sid for sid, session in self._sessions.items() if session.expires_at <= now]:
            del self._sessions[session_id]

    def get(self, session_id: str) -> Session | None:
        now = self.clock()
        self._prune(now)
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.expires_at = now + self.idle_seconds
        return session

    def end(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def add_exchange(self, session_id: str, question: str, answer: str, *, commit_if=None) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError("session expired or unknown")
        # Compact a copy: a failed/cancelled summary must not corrupt session history.
        candidate = Session(session.id, session.expires_at, list(session.recent), session.summary)
        candidate.recent.append(Exchange(question, answer))
        while len(candidate.recent) > self.recent_exchanges:
            self._compact_oldest(candidate)
        while (candidate.recent and
               self.token_counter("\n".join(x.text() for x in candidate.recent)) > self.history_tokens):
            self._compact_oldest(candidate)
        if self._sessions.get(session_id) is not session or (commit_if is not None and not commit_if()):
            return session
        session.recent, session.summary = candidate.recent, candidate.summary
        return session

    def _compact_oldest(self, session: Session) -> None:
        if self.summarizer is None:
            raise RuntimeError("history summarizer is required when history exceeds its budget")
        older = session.recent[0]
        summary = self.summarizer(session.summary, [older])
        if not isinstance(summary, str) or self.token_counter(summary) > self.summary_tokens:
            raise ValueError("history summarizer returned an invalid or over-budget summary")
        session.summary = summary
        session.recent.pop(0)

    def context(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if session is None:
            raise KeyError("session expired or unknown")
        return {"summary": session.summary, "exchanges": list(session.recent)}


class AnswerCache:
    """Exact, session-scoped cache of completed answers, held only in memory."""

    def __init__(self, *, ttl_seconds: int = 86400, max_entries: int = 512,
                 max_bytes: int = 1024**3, clock: Callable[[], float] = time.time):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.clock = clock
        self._entries: dict[tuple[str, str], tuple[float, Any, int]] = {}
        self._bytes = 0

    @staticmethod
    def _key(identity: Any) -> str:
        # JSON preserves exact strings and makes mapping key order immaterial.
        import json
        return json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def get(self, session_id: str, identity: Any, *, session_expires_at: float) -> Any | None:
        key = (session_id, self._key(identity))
        now = self.clock()
        self._prune(now)
        item = self._entries.get(key)
        if item is None:
            return None
        if item[0] <= now or session_expires_at <= now:
            self._remove(key)
            return None
        self._entries.pop(key)
        self._entries[key] = item
        return item[1]

    def put(self, session_id: str, identity: Any, answer: Any, *, session_expires_at: float) -> None:
        now = self.clock()
        self._prune(now)
        expiry = min(now + self.ttl_seconds, session_expires_at)
        if expiry > self.clock():
            key_text = self._key(identity)
            key = (session_id, key_text)
            size = len(session_id.encode()) + len(key_text.encode()) + len(
                self._json(answer).encode())
            if size > self.max_bytes or self.max_entries <= 0:
                return
            self._remove(key)
            self._entries[key] = (expiry, answer, size)
            self._bytes += size
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                self._remove(next(iter(self._entries)))

    @staticmethod
    def _json(value: Any) -> str:
        import json
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _remove(self, key: tuple[str, str]) -> None:
        item = self._entries.pop(key, None)
        if item is not None:
            self._bytes -= item[2]

    def _prune(self, now: float) -> None:
        for key in [key for key, item in self._entries.items() if item[0] <= now]:
            self._remove(key)

    def clear_session(self, session_id: str) -> None:
        for key in [k for k in self._entries if k[0] == session_id]:
            self._remove(key)
