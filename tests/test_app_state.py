import unittest

from rag_app.state import AnswerCache, SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_session_is_opaque_idle_expires_and_end_clears(self):
        now = [0]
        store = SessionStore(clock=lambda: now[0], idle_seconds=30)
        session = store.create()
        self.assertGreaterEqual(len(session.id), 40)
        now[0] = 29
        self.assertIsNotNone(store.get(session.id))
        now[0] = 60
        self.assertIsNone(store.get(session.id))
        session = store.create()
        store.end(session.id)
        self.assertIsNone(store.get(session.id))

    def test_history_keeps_six_recent_and_summarizes_older(self):
        seen = []
        store = SessionStore(recent_exchanges=6, summary_tokens=3,
                             token_counter=lambda s: len(s.split()),
                             summarizer=lambda old, exchanges: seen.append(exchanges[0].question) or "one two three")
        session = store.create()
        for n in range(7):
            store.add_exchange(session.id, f"q{n}", f"a{n}")
        context = store.context(session.id)
        self.assertEqual([e.question for e in context["exchanges"]], [f"q{i}" for i in range(1, 7)])
        self.assertEqual(seen, ["q0"])
        self.assertEqual(context["summary"], "one two three")

    def test_history_token_budget_moves_old_exchanges_to_summary(self):
        store = SessionStore(recent_exchanges=6, history_tokens=4, token_counter=lambda s: len(s.split()),
                             summarizer=lambda old, exchanges: old + " one" if old else "one")
        session = store.create()
        store.add_exchange(session.id, "one", "a")
        store.add_exchange(session.id, "two", "b")
        context = store.context(session.id)
        self.assertEqual(len(context["exchanges"]), 1)
        self.assertIn("one", context["summary"])

    def test_oversized_summary_does_not_drop_exchange(self):
        store = SessionStore(recent_exchanges=1, summary_tokens=1, token_counter=lambda s: len(s.split()),
                             summarizer=lambda old, exchanges: "too many words")
        session = store.create()
        store.add_exchange(session.id, "first", "answer")
        with self.assertRaises(ValueError):
            store.add_exchange(session.id, "second", "answer")
        self.assertEqual(session.recent[0].question, "first")

    def test_expired_sessions_pruned_and_active_lru_evicted_at_bound(self):
        now = [0]
        store = SessionStore(clock=lambda: now[0], idle_seconds=10, max_sessions=2)
        old = store.create()
        now[0] = 1
        recent = store.create()
        now[0] = 2
        self.assertIsNotNone(store.get(old.id))
        now[0] = 3
        newest = store.create()
        self.assertIsNone(store.get(recent.id))
        self.assertIsNotNone(store.get(old.id))
        self.assertIsNotNone(store.get(newest.id))

        now[0] = 100
        fresh = store.create()
        self.assertEqual(list(store._sessions), [fresh.id])


class AnswerCacheTests(unittest.TestCase):
    def test_exact_identity_session_scope_and_lifetime_cap(self):
        now = [100]
        cache = AnswerCache(ttl_seconds=86400, clock=lambda: now[0])
        cache.put("s1", {"question": "Same?", "version": 1}, "ok", session_expires_at=130)
        self.assertEqual(cache.get("s1", {"version": 1, "question": "Same?"}, session_expires_at=130), "ok")
        self.assertIsNone(cache.get("s2", {"question": "Same?", "version": 1}, session_expires_at=130))
        self.assertIsNone(cache.get("s1", {"question": "same?", "version": 1}, session_expires_at=130))
        now[0] = 130
        self.assertIsNone(cache.get("s1", {"question": "Same?", "version": 1}, session_expires_at=160))

    def test_clear_session(self):
        cache = AnswerCache()
        cache.put("s", "q", "answer", session_expires_at=9999999999)
        cache.clear_session("s")
        self.assertIsNone(cache.get("s", "q", session_expires_at=9999999999))

    def test_expired_entries_pruned_and_oldest_evicted_at_entry_bound(self):
        now = [100]
        cache = AnswerCache(ttl_seconds=20, max_entries=2, clock=lambda: now[0])
        cache.put("s", "a", "A", session_expires_at=1000)
        cache.put("s", "b", "B", session_expires_at=1000)
        self.assertEqual(cache.get("s", "a", session_expires_at=1000), "A")
        cache.put("s", "c", "C", session_expires_at=1000)
        self.assertIsNone(cache.get("s", "b", session_expires_at=1000))
        self.assertEqual(cache.get("s", "a", session_expires_at=1000), "A")
        self.assertEqual(cache.get("s", "c", session_expires_at=1000), "C")
        now[0] = 120
        cache.put("s", "d", "D", session_expires_at=1000)
        self.assertEqual(len(cache._entries), 1)
        self.assertEqual(cache.get("s", "d", session_expires_at=1000), "D")

    def test_oversized_entry_is_not_cached_and_byte_usage_stays_bounded(self):
        cache = AnswerCache(max_bytes=80)
        cache.put("s", "question", {"answer": "x" * 200}, session_expires_at=9999999999)
        self.assertEqual(cache._entries, {})
        self.assertEqual(cache._bytes, 0)


if __name__ == "__main__":
    unittest.main()


def test_summary_failure_does_not_change_history():
    from rag_app.state import SessionStore
    def fail(*args):raise RuntimeError('summary failed')
    store=SessionStore(recent_exchanges=1,summarizer=fail)
    session=store.create();store.add_exchange(session.id,'first','answer')
    import pytest
    with pytest.raises(RuntimeError):store.add_exchange(session.id,'second','answer')
    assert [x.question for x in store.context(session.id)['exchanges']]==['first']


def test_cancelled_history_update_is_not_committed():
    from rag_app.state import SessionStore
    store=SessionStore();session=store.create()
    store.add_exchange(session.id,'cancelled','answer',commit_if=lambda:False)
    assert store.context(session.id)['exchanges']==[]
