"""Shared local evaluation budget, valued at published standard-route prices."""
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from rag_app.indexes import atomic_json


class BudgetLimitError(RuntimeError):
    pass


class EvaluationBudget:
    def __init__(self, path, limit=0.50):
        self.path = Path(path)
        self.limit = limit

    def _change(self, update):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {
                'limit_usd': self.limit, 'input_usd_per_million': 0.30,
                'output_usd_per_million': 1.20, 'cache_read_usd_per_million': 0.0015,
                'basis': 'standard-route token value; selected free route bills zero',
                'requests': {}}
            result = update(data)
            atomic_json(self.path, data)
            return result

    def begin(self, body):
        if body.get('model', 'deepseek-v4.1-flash:free') not in {'deepseek-v4.1-flash', 'deepseek-v4.1-flash:free'}:
            raise BudgetLimitError('No configured price for this judge model')
        # UTF-8 byte count bounds text tokens conservatively before a request.
        size = len(json.dumps(body, ensure_ascii=False).encode()) + 256
        output = body.get('max_tokens', body.get('max_completion_tokens', 3000))
        reserve = (size * 0.30 + output * 1.20) / 1_000_000

        def reserve_request(data):
            if sum(r['value_usd'] for r in data['requests'].values()) + reserve > data['limit_usd']:
                raise BudgetLimitError('Evaluation token-value budget reached')
            ticket = str(uuid4())
            data['requests'][ticket] = {'state': 'reserved', 'value_usd': reserve}
            return ticket

        return self._change(reserve_request)

    def finish(self, ticket, usage, status):
        def settle(data):
            if status == 200 and not usage:
                data['requests'][ticket].update(state='usage_missing', status=status)
                return
            prompt = usage.get('prompt_tokens') or 0
            completion = usage.get('completion_tokens') or 0
            cached = min(prompt, usage.get('prompt_cache_hit_tokens') or
                         (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
            value = ((prompt - cached) * 0.30 + cached * 0.0015 + completion * 1.20) / 1_000_000
            data['requests'][ticket] = {'state': 'reported', 'status': status,
                'input_tokens': prompt, 'output_tokens': completion, 'cached_input_tokens': cached,
                'value_usd': value}

        self._change(settle)
