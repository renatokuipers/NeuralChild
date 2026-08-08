# llm_module.py

Thin HTTP client for an OpenAI-compatible chat-completions endpoint and an embeddings endpoint, both addressed via URLs read from the project `config` singleton. Adds retry-with-jitter, optional JSON extraction from fenced code blocks, and a hardcoded offline stub response generator. No provider SDK is used — plain `requests.post` with a JSON body.

## Public surface

| Name | Kind | Contract |
|---|---|---|
| `LLMError` | class (Exception) | Declared at llm_module.py:21; never raised or caught anywhere in this file. |
| `chat_completion` | function | POST system+user prompts to `config.server.llm_server_url`; returns response text, or a parsed dict when `structured_output=True`, or `None` after all retries fail. |
| `get_embeddings` | function | POST a list of strings to `config.server.embedding_server_url`; returns a list of embedding vectors or `None`. |
| `simulate_llm_response` | function | Returns one of three hardcoded values (a dict or one of two strings) selected by the `structured_output` flag plus a keyword match on the system prompt. Never called within this file. |
| `logger` | module constant | `logging.getLogger(__name__)`; module import also calls `logging.basicConfig(level=INFO)` (llm_module.py:18). |

## Key behaviour

- `chat_completion` defaults: `temperature` falls back to `config.model.temperature`; `max_tokens` falls back to `config.model.max_tokens`, and is set to `None` when that config value is `<= 0` (llm_module.py:53-54). `max_tokens` is only added to the payload if truthy (llm_module.py:79-80), so an explicit `0` is silently dropped.
- Payload is `{model, messages:[system,user], temperature}` — `config.model.llm_model` names the model (llm_module.py:69-76). Headers carry only `Content-Type: application/json` (llm_module.py:57-59, llm_module.py:174-176); there is no `Authorization` header, so any endpoint requiring auth would reject every request.
- Request timeout is hardcoded at 30 s in both functions (llm_module.py:90, llm_module.py:193). Default retries: 3, base delay 2.0 s, multiplied by `random.uniform(0.5, 1.5)` jitter (llm_module.py:149, llm_module.py:214).
- Response parsing expects `response_data["choices"][0]["message"]["content"]` (llm_module.py:97-98). Embeddings expect `response_data["data"][i]["embedding"]` (llm_module.py:199-201).
- JSON extraction for `structured_output` handles three shapes: a ```` ```json ```` fence (offset +7), a bare ```` ``` ```` fence (offset +3), or raw content assumed to be JSON. Missing closing fence falls back to end-of-string (llm_module.py:104-120).

```
chat_completion(structured_output=True)
  |
  v
POST --> HTTP error / RequestException --> log, sleep(delay*jitter), retry
  |
  +--> "choices" missing --> log, sleep(delay*jitter), retry
  |
  +--> content extracted
         |
         +-- JSONDecodeError, attempts left --> sleep(delay) [no jitter], continue
         +-- JSONDecodeError, last attempt  --> return raw str  (type escapes dict contract)
         +-- parsed                          --> return dict
  |
  v
all retries exhausted --> log --> return None
```

- `get_embeddings` is a structural copy of the same retry loop with no structured-output branch; both loops duplicate the log/sleep/jitter logic.
- `simulate_llm_response` branches on `"mother" in system_prompt.lower() or "nurturing" in system_prompt.lower()` (llm_module.py:244); the structured branch returns a fixed dict with keys `understanding` / `response` / `action` (llm_module.py:248-252), and the two remaining branches return fixed strings (llm_module.py:255, llm_module.py:258). `user_prompt` is accepted but never read.

## Imports

Third-party: `requests`, `pydantic` (`BaseModel`).
Standard library: `json`, `logging`, `time`, `random`, `typing` (`Dict`, `Any`, `Optional`, `List`, `Union`).
Project-internal: `config` (the name `config` imported from a module named `config`). Attributes referenced on it: `config.model.temperature`, `config.model.max_tokens`, `config.model.llm_model`, `config.model.embedding_model`, `config.server.llm_server_url`, `config.server.embedding_server_url`. Whether these exist is unverifiable from this file alone.

## Defects and gaps

- `BaseModel` imported at llm_module.py:13 and never used.
- `LLMError` defined at llm_module.py:21 and never raised, caught, or referenced in this file — failures return `None` instead.
- `simulate_llm_response` (llm_module.py:223) is defined but never invoked from this file, and `chat_completion` never falls back to it despite the docstring at llm_module.py:230-231 describing it as a fallback when the real service is unavailable.
- Hardcoded caregiver schema: when `structured_output=True`, llm_module.py:63-66 appends a fixed JSON schema mentioning "your nurturing response to the child" to *any* caller's system prompt. There is no way to request a different structured shape.
- Return-type contract violation: with `structured_output=True`, llm_module.py:134 returns the raw string on the final attempt if JSON parsing fails, so callers expecting a dict can receive `str`.
- Broad `except Exception` at llm_module.py:144 and llm_module.py:209 swallows `KeyError`/`TypeError` from malformed response shapes (llm_module.py:98, llm_module.py:201) as retryable transport-style failures; the original traceback is discarded, only `str(e)` is logged.
- `response.raise_for_status()` (llm_module.py:94, llm_module.py:196) turns every non-2xx status into a `RequestException`, so permanently-failing 4xx responses (unknown model name, malformed payload) are retried the full `retry_count` with jittered sleeps despite never being able to succeed.
- Retry-delay inconsistency: the JSON-parse retry path sleeps `retry_delay` with no jitter (llm_module.py:130) then `continue`s, bypassing the jittered sleep used by every other path.
- `retry_count=0` makes `range(retry_count)` empty — no request is ever sent and the function returns `None` while logging "All 0 attempts ... failed" (llm_module.py:153, llm_module.py:218).
- `logging.basicConfig(level=logging.INFO)` at llm_module.py:18 runs on import, mutating root-logger configuration for the whole process.
- 30 s timeout and the 0.5–1.5 jitter range are hardcoded and not exposed as parameters or config values, unlike `retry_count`/`retry_delay`.
- The `else` at llm_module.py:138-139 logs the missing-`choices` case but produces no distinguishable error to the caller — indistinguishable from a network failure.

## Notes

- Both request functions signal every failure mode identically (`None`), so callers cannot distinguish "server down" from "malformed response" from "JSON unparseable".
- The fenced-block extractor uses fixed offsets (+7 for ```` ```json ````, +3 for ```` ``` ````) and the language-tag test is case-sensitive; an uppercase or space-separated tag falls into the bare-fence branch, leaving the tag text inside the extracted slice so parsing fails and the attempt is burned as a retry.
