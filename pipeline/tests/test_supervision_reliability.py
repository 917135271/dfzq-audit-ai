import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from test_supervision_extraction import Extractor, make_ir, request_data

from common.supervision import SupervisionExtractRequest
from pipeline.llm_client import LLMError, make_llm_client
from pipeline.supervision.extract import prepare_extraction
from pipeline.supervision.service import SupervisionConfig


@pytest.mark.parametrize('failure', ['json', 'schema'])
def test_invalid_output_is_regenerated_with_same_evidence(failure):
    class Flaky(Extractor):
        calls = 0

        def chat_json(self, system, user):
            self.calls += 1
            if self.calls == 1:
                self.original = json.loads(user)
                if failure == 'json':
                    raise LLMError('invalid JSON')
                return {'facts': [{'ruleId': 'REG-1'}]}
            assert json.loads(user) == self.original
            assert '不得省略 value' in system
            return super().chat_json(system, user)

    client = Flaky()
    output = prepare_extraction(SupervisionExtractRequest.model_validate(request_data()),
                                make_ir(), client=client, max_evidence_chars=1000)
    assert client.calls == 2
    assert len(output['facts']) == 1
    assert any(c.get('code') == 'MODEL_OUTPUT_RETRY' for c in output['metadataChecks'])


def test_exhausted_schema_retries_fail_without_partial_result():
    class Broken:
        calls = 0

        def chat_json(self, system, user):
            self.calls += 1
            return {'facts': [{'ruleId': 'REG-1'}]}

    client = Broken()
    with pytest.raises(ValidationError):
        prepare_extraction(SupervisionExtractRequest.model_validate(request_data()), make_ir(),
                           client=client, max_evidence_chars=1000, model_attempts=2)
    assert client.calls == 2


def test_untrusted_quotes_are_not_relaxed_by_retries():
    client = Extractor(lambda f: f['values']['problem'].update(quote='不存在的原文'))
    with pytest.raises(ValueError, match='non-verbatim'):
        prepare_extraction(SupervisionExtractRequest.model_validate(request_data()), make_ir(),
                           client=client, max_evidence_chars=1000)
    assert len(client.inputs) == 1


def test_supervision_timeout_configuration_does_not_change_query_defaults(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    config = SupervisionConfig(backend='gateway', max_evidence_chars=16000)
    with patch('pipeline.llm_client.LLMClient') as constructor:
        make_llm_client(timeout=config.model_timeout_seconds, retries=1)
        assert constructor.call_args.kwargs['timeout'] == 300
        assert constructor.call_args.kwargs['retries'] == 1
        make_llm_client()
        assert constructor.call_args.kwargs['timeout'] == 60
        assert constructor.call_args.kwargs['retries'] == 3


@pytest.mark.parametrize('field,value', [('model_attempts', 0), ('model_attempts', 4),
                                         ('model_timeout_seconds', 0)])
def test_invalid_retry_configuration_rejected(field, value):
    with pytest.raises(ValidationError):
        SupervisionConfig(backend='gateway', max_evidence_chars=16000, **{field: value})
