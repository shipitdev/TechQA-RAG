"""Generate OpenUI through the same bounded Groq transport as answer generation."""
import os
import time
import httpx
from rag_app.generation import GenerationError, build_messages

UI_SCHEMA = {'type': 'OBJECT', 'properties': {'openui': {'type': 'STRING'}}, 'required': ['openui']}


def generate_presentation(result, question, generator):
    """Return approved UI or fall back to the already verified canonical answer."""
    started = time.monotonic()
    metrics = result.setdefault('metrics', {})
    try:
        ui = os.getenv('UI_INTERNAL_URL', 'http://127.0.0.1:3000')
        spec = httpx.get(ui + '/api/ui-spec', timeout=5).json()
        prompt = build_messages(
            spec['system_prompt'] + '\nPut only the OpenUI Lang program in the openui JSON field. Present this verified answer without adding facts. Use the source IDs in the approved answer.',
            {'approved_answer': result['answer']})
        lang = generator.json_call(prompt, UI_SCHEMA)['openui']
        checked = httpx.post(ui + '/api/ui-validate', json={'openui': lang, 'sources': result['sources']}, timeout=5).json()
        if not checked.get('valid'):
            metrics['presentation_error'] = 'invalid_layout'
            return None
        ids = set(checked.get('citation_source_ids', []))
        if not ids or not ids <= set(result['citations']):
            metrics['presentation_error'] = 'invalid_citations'
            return None
        cited_sources = [source for source in result['sources'] if source['id'] in ids]
        verdict = generator._verify(question, cited_sources, {
            'answer': checked['visible_text'], 'citations': sorted(ids),
            'approved_answer': result['answer']})
        if verdict['verdict'] == 'complete' and verdict['supported'] and verdict['complete'] and not verdict['missing_steps'] and not verdict['version_conflicts']:
            return lang
        metrics['presentation_error'] = 'unverified_layout'
    except GenerationError as exc:
        metrics['presentation_error'] = exc.category
    except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError):
        metrics['presentation_error'] = 'presentation_dependency_failed'
    finally:
        metrics['presentation_ms'] = (time.monotonic() - started) * 1000
    return None
