import unittest
import json
from unittest.mock import patch
import httpx

from rag_app.generation import (AnswerGenerator, ConversationResolver, GenerationError,
                                ANSWER_SCHEMA, GroqREST, MODEL, _groq_schema, history_summarizer)


class AnswerGeneratorTests(unittest.TestCase):
    evidence = [{"id": "S1", "text": "Use command X for version 2."}]
    answer = {"status": "answered", "answer": "Use X [S1].", "citations": ["S1"]}
    complete = {"verdict": "complete", "supported": True, "complete": True,
                "missing_steps": [], "version_conflicts": [], "findings": [], "unsupported_claims": []}

    def run_with(self, results):
        calls = []
        def fake(prompt, schema):
            calls.append((prompt, schema))
            item = results.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return AnswerGenerator(fake).generate("How?", self.evidence), calls

    def test_writer_verifier_pass_returns_answer(self):
        result, calls = self.run_with([self.answer.copy(), self.complete.copy()])
        self.assertEqual(result, self.answer)
        self.assertEqual(len(calls), 2)

    def test_verifier_claim_findings_override_optimistic_booleans(self):
        verdict = {**self.complete, "unsupported_claims": ["Download label does not prove version 2 is excluded."]}
        result, calls = self.run_with([self.answer.copy(), verdict,
            {"status":"insufficient_evidence","answer":"","citations":[]}])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(calls), 3)

    def test_specific_unsupported_claim_gets_one_repair_and_must_be_reverified(self):
        rejected = {**self.complete, "verdict":"evidence_gap", "supported":False,
                    "unsupported_claims":["Unjustified version exclusion"]}
        revised = {**self.answer, "answer":"Use the explicitly supported version 2 procedure [S1]."}
        result,calls = self.run_with([self.answer.copy(),rejected,revised,self.complete.copy()])
        self.assertEqual(result,revised)
        self.assertEqual(len(calls),4)
        result,calls = self.run_with([self.answer.copy(),rejected,revised,rejected])
        self.assertEqual(result["status"],"insufficient_evidence")
        self.assertEqual(len(calls),4)

    def test_gap_withholds_without_repair(self):
        gap = {**self.complete, "verdict": "evidence_gap", "supported": False, "complete": False}
        result, calls = self.run_with([self.answer.copy(), gap])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["citations"], [])
        self.assertEqual(len(calls), 2)

    def test_unsupported_verdict_and_version_conflict_never_enter_repair(self):
        unsupported = {**self.complete, "supported": False}
        result, calls = self.run_with([self.answer.copy(), unsupported])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(calls), 2)
        conflict = {**self.complete, "complete": False, "version_conflicts": ["version mismatch"]}
        result, calls = self.run_with([self.answer.copy(), conflict])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(len(calls), 2)

    def test_clarification_is_not_answered(self):
        clarify = {**self.complete, "verdict": "clarify", "supported": False, "complete": False}
        result, calls = self.run_with([self.answer.copy(), clarify])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(len(calls), 2)

    def test_incomplete_draft_gets_one_repair_then_second_verification(self):
        incomplete = {**self.complete, "complete": False, "findings": ["step missing"]}
        revised = {**self.answer, "answer": "Do X then Y [S1]."}
        result, calls = self.run_with([self.answer.copy(), incomplete, revised, self.complete.copy()])
        self.assertEqual(result["answer"], revised["answer"])
        self.assertEqual(len(calls), 4)
        self.assertIn("Repair the rejected draft", calls[2][0])

    def test_bad_citation_and_provider_failure_are_errors(self):
        invalid = {**self.answer, "citations": ["S9"]}
        calls = []
        def invalid_twice(prompt, schema):
            calls.append((prompt, schema))
            return invalid
        with self.assertRaises(GenerationError):
            AnswerGenerator(invalid_twice).generate("How?", self.evidence)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(GenerationError):
            self.run_with([GenerationError("offline")])

    def test_bad_citation_repair_uses_actual_source_ids_and_then_verifies(self):
        evidence = [{"id": "S5", "text": "Use command X."}]
        invalid = {"status": "answered", "answer": "Use X [S4].", "citations": ["S4"]}
        valid = {"status": "answered", "answer": "Use X [S5].", "citations": ["S5"]}
        results = [invalid, valid, self.complete.copy()]
        calls = []
        def fake(prompt, schema):
            calls.append((prompt, schema))
            return results.pop(0)
        original_schema = json.dumps(ANSWER_SCHEMA)
        result = AnswerGenerator(fake).generate("How?", evidence)
        self.assertEqual(result, valid)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][1]["properties"]["citations"]["items"]["enum"], ["S5"])
        self.assertIn("Use only source IDs", calls[1][0])
        self.assertEqual(json.dumps(ANSWER_SCHEMA), original_schema)

    def test_format_repair_consumes_only_repair_when_semantic_check_fails(self):
        invalid = {"status": "answered", "answer": "Use X [S9].", "citations": ["S9"]}
        corrected = {"status": "answered", "answer": "Use X [S1].", "citations": ["S1"]}
        incomplete = {**self.complete, "complete": False, "missing_steps": ["step Y"]}
        result, calls = self.run_with([invalid, corrected, incomplete])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(calls), 3)

    def test_empty_evidence_makes_no_provider_call_and_nonanswers_are_sanitized(self):
        generator = AnswerGenerator(lambda *args: self.fail("provider should not be called"))
        result = generator.generate("How?", [])
        self.assertEqual(result["status"], "insufficient_evidence")
        private = {"status": "needs_clarification", "answer": "partial SECRET fix", "citations": []}
        result, _ = self.run_with([private])
        self.assertNotIn("SECRET", result["answer"])

    def test_missing_key_is_bounded_error(self):
        with self.assertRaisesRegex(GenerationError, "not configured"):
            GroqREST(api_key="")

    def test_empty_nonanswers_are_safe_but_empty_answers_are_errors(self):
        for status in ("insufficient_evidence", "needs_clarification"):
            result, calls = self.run_with([{"status": status, "answer": "", "citations": []}])
            self.assertEqual(result["status"], status)
            self.assertTrue(result["answer"])
            self.assertEqual(len(calls), 1)
        with self.assertRaises(GenerationError) as error:
            self.run_with([{"status": "answered", "answer": "", "citations": []}])
        self.assertEqual(error.exception.category, "invalid_answer_shape")

    def test_private_trace_records_verifier_rejection_without_changing_output(self):
        replies = [self.answer.copy(), {**self.complete, "verdict": "evidence_gap", "complete": False}]
        generator = AnswerGenerator(lambda *args: replies.pop(0))
        result = generator.generate("How?", self.evidence)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual([x["stage"] for x in generator.trace], ["write", "verify"])
        self.assertEqual(generator.trace[-1]["result"]["verdict"], "evidence_gap")
        self.assertNotIn("trace", result)

    def test_quota_headers_pace_next_call_and_respect_deadline(self):
        clock = [0.0]
        calls = []
        def sleep(seconds): clock[0] += seconds
        def post(url, **kwargs):
            calls.append(clock[0])
            return httpx.Response(200, request=httpx.Request("POST", url),
                headers={"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "1m2s"},
                json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
        with patch("rag_app.generation.time.monotonic", lambda: clock[0]), patch("rag_app.generation.time.sleep", sleep):
            client = GroqREST("secret", post=post)
            client.json_call("test", {})
            client.json_call("test", {})
            self.assertGreaterEqual(calls[1]-calls[0], 64)
            client.deadline = clock[0] + 5
            with self.assertRaises(GenerationError) as error: client.json_call("test", {})
            self.assertEqual(error.exception.category, "quota_deadline")
            self.assertEqual(len(calls), 2)

    def test_two_rate_limits_get_buffered_bounded_retries(self):
        clock, calls = [0.0], []
        def post(url, **kwargs):
            calls.append(clock[0])
            return httpx.Response(429 if len(calls) < 3 else 200,
                request=httpx.Request("POST", url), headers={"retry-after": "nan"},
                json={"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]})
        def sleep(seconds): clock[0] += seconds
        with patch("rag_app.generation.time.monotonic", lambda: clock[0]), patch("rag_app.generation.time.sleep", sleep):
            client = GroqREST("secret", post=post)
            self.assertEqual(client.json_call("test", {}), {})
        self.assertEqual(len(calls), 3)
        self.assertGreaterEqual(calls[1]-calls[0], 5)
        self.assertGreaterEqual(calls[2]-calls[1], 5)

    def test_unreferenced_citation_array_does_not_pass(self):
        uncited = {**self.answer, "answer": "Use X without an inline citation."}
        calls = []
        def fake(prompt, schema):
            calls.append((prompt, schema))
            return uncited
        with self.assertRaises(GenerationError):
            AnswerGenerator(fake).generate("How?", self.evidence)
        self.assertEqual(len(calls), 2)

    def test_groq_retries_one_transient_and_accumulates_usage(self):
        attempts = []
        url = "https://api.groq.com/openai/v1/chat/completions"
        def response(payload, status=200, headers=None):
            return httpx.Response(status, json=payload, headers=headers, request=httpx.Request("POST", url))
        def post(request_url, **kwargs):
            attempts.append((request_url, kwargs))
            if len(attempts) == 1:
                return response({"error": {"message": "proxy response"}}, 503, {"Retry-After": "0"})
            payload = {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"ok": true}', "reasoning": "secret internal reasoning"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6,
                          "completion_tokens_details": {"reasoning_tokens": 1},
                          "prompt_tokens_details": {"cached_tokens": 3}}}
            return response(payload)
        with patch("rag_app.generation.time.sleep", lambda _: None):
            client = GroqREST("secret", post=post, timeout=60)
            result = client.json_call("prompt", {"type": "OBJECT"})
        self.assertEqual(result, {"ok": True})
        self.assertNotIn("secret internal reasoning", json.dumps(result))
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all(kwargs["timeout"] <= 60 for _, kwargs in attempts))
        self.assertNotIn("secret", attempts[0][0])
        self.assertEqual(attempts[0][1]["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(client.usage[0]["prompt_tokens"], 4)
        self.assertIn("latency_seconds", client.usage[0])
        request_body = attempts[0][1]["json"]
        self.assertEqual(request_body["model"], MODEL)
        self.assertEqual(request_body["reasoning_effort"], "low")
        self.assertEqual(request_body["max_completion_tokens"], 1500)
        self.assertFalse(request_body["stream"])
        self.assertEqual(request_body["response_format"]["type"], "json_schema")
        self.assertTrue(request_body["response_format"]["json_schema"]["strict"])
        self.assertEqual(request_body["messages"][0]["role"], "system")
        self.assertEqual(request_body["messages"][1]["role"], "user")

    def test_groq_key_aliases_and_route(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "standard", "GROK_API_KEY": "alias"}):
            self.assertEqual(GroqREST().api_key, "standard")
        with patch.dict("os.environ", {"GROQ_API_KEY": "", "GROK_API_KEY": "alias"}):
            self.assertEqual(GroqREST().api_key, "alias")
        self.assertEqual(MODEL, "openai/gpt-oss-120b")

    def test_nested_groq_schema_is_strict_and_lowercase(self):
        converted = _groq_schema({"type": "OBJECT", "properties": {
            "items": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "name": {"type": "STRING"}, "enabled": {"type": "BOOLEAN"}}}}}})
        self.assertEqual(converted["type"], "object")
        self.assertFalse(converted["additionalProperties"])
        self.assertEqual(converted["required"], ["items"])
        nested = converted["properties"]["items"]["items"]
        self.assertEqual(nested["required"], ["name", "enabled"])
        self.assertFalse(nested["additionalProperties"])
        self.assertEqual(nested["properties"]["name"]["type"], "string")

    def test_billing_error_is_not_retried_and_model_is_explicit(self):
        attempts = []
        url = "https://api.groq.com/openai/v1/chat/completions"
        def unavailable(request_url, **kwargs):
            attempts.append((request_url, kwargs))
            return httpx.Response(402, json={"error": {"message": "error code:1010"}}, request=httpx.Request("POST", url))
        client = GroqREST("secret", post=unavailable)
        with self.assertRaises(GenerationError) as error:
            client.json_call("test", {"type": "OBJECT"})
        self.assertEqual(error.exception.category, "http_402")
        self.assertEqual(len(attempts), 1)
        self.assertIn("/chat/completions", attempts[0][0])
        self.assertNotIn("1010", str(error.exception))

    def test_truncated_and_refusal_responses_are_bounded_errors(self):
        payloads = [
            {"choices": [{"finish_reason": "length", "message": {"content": '{"answer":"partial'}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "do not surface this"}}]},
        ]
        for payload in payloads:
            url = "https://api.groq.com/openai/v1/chat/completions"
            client = GroqREST("secret", post=lambda *args, **kwargs: httpx.Response(
                200, json=payload, request=httpx.Request("POST", url)))
            with self.assertRaises(GenerationError) as error:
                client.json_call("test", {"type": "OBJECT"})
            self.assertNotIn("partial", str(error.exception))
            self.assertNotIn("surface", str(error.exception))

    def test_httpx_transport_error_is_bounded(self):
        url = "https://api.groq.com/openai/v1/chat/completions"
        def broken(*args, **kwargs):
            raise httpx.ConnectError("secret proxy response body", request=httpx.Request("POST", url))
        with self.assertRaises(GenerationError) as error:
            GroqREST("secret", post=broken).json_call("prompt", {"type": "OBJECT"})
        self.assertIn("ConnectError", str(error.exception))
        self.assertNotIn("secret", str(error.exception))

    def test_followup_resolver_uses_history_only_when_dependent(self):
        calls = []
        independent = ConversationResolver(lambda prompt, schema: calls.append(prompt) or {
            "dependent": False, "question": "modified", "needs_clarification": False})
        self.assertEqual(independent.resolve("Exact standalone?", {"summary": "old", "exchanges": []}),
                         {"dependent": False, "question": "Exact standalone?", "needs_clarification": False})
        self.assertIn("old", calls[0])
        dependent = ConversationResolver(lambda prompt, schema: {
            "dependent": True, "question": "How about Product X version 11?", "needs_clarification": False})
        self.assertEqual(dependent.resolve("What about version 11?", {"summary": "Product X v10", "exchanges": []})["question"],
                         "How about Product X version 11?")
        ambiguous = ConversationResolver(lambda prompt, schema: {
            "dependent": True, "question": "guess", "needs_clarification": True})
        self.assertTrue(ambiguous.resolve("What about that?", {})["needs_clarification"])

    def test_history_summarizer_preservation_instruction_and_validation(self):
        prompts = []
        summarize = history_summarizer(lambda prompt, schema: prompts.append(prompt) or {"summary": "Product X version 7 error E42; correction: version 8."})
        from rag_app.state import Exchange
        result = summarize("old", [Exchange("error?", "v8")])
        self.assertIn("version 7", result)
        self.assertIn("Preserve exact product names", prompts[0])
        with self.assertRaises(GenerationError):
            history_summarizer(lambda prompt, schema: {})("", [])


if __name__ == "__main__":
    unittest.main()
