# llm_module.py

Source under review: the **legacy NeuralChild snapshot** at `C:/Users/renat/AppData/Local/Temp/claude/E--python-projects-NeuralChild/9361633d-db3a-49f2-8f27-fb7624f16eb7/scratchpad/neural-child/llm_module.py` — 89 lines. Every line citation below refers to that legacy file only; any same-named module elsewhere in the current working tree is out of scope and was not read.

Single-function HTTP client for an OpenAI-compatible chat-completions endpoint, defaulted to a local LM Studio server running `qwen2.5-7b-instruct`. It supports two mutually exclusive modes: token-streaming (printed to stdout as it arrives) and a non-streaming structured-output mode that constrains the reply to the `MotherResponse` JSON schema. All failures are printed and converted to `None`; nothing is raised to the caller.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `chat_completion` | function | Posts a two-message (system + user) chat completion and returns the assistant text, a `MotherResponse` dict, or `None` on any error. |

Parameters and defaults, per legacy llm_module.py:8-17: `system_prompt`, `user_prompt` (required); `model="qwen2.5-7b-instruct"`; `temperature=0.7`; `max_tokens=-1`; `stream=True`; `server_url="http://0.0.0.0:1234/v1/chat/completions"`; `structured_output=False`. No retry, no config lookup, no logging handler — everything is a literal default or a print.

## Key behaviour

- `structured_output=True` builds a `response_format` block of type `json_schema` with `name` = the `MotherResponse` class name, `strict: True`, and `schema` from `MotherResponse.model_json_schema()`, then force-overrides `stream` to `False` (legacy llm_module.py:22-29, 42-46).
- Payload carries exactly two messages, system then user — no history, no tool definitions, no assistant turns (legacy llm_module.py:31-40).
- Streaming branch: POST with `stream=True` and a 60 s timeout, `raise_for_status()`, then iterates `iter_lines(decode_unicode=True)`. Each non-empty line has its first six characters sliced off (assumed `"data: "` prefix) and is JSON-parsed; `choices[0].delta.content` is appended to a list and echoed to stdout with `end="", flush=True`. A JSON decode failure skips the line. Returns the joined string (legacy llm_module.py:51-69).
- Non-streaming branch: single POST with 60 s timeout, reads `choices[0].message.content`. With a schema it parses that string as JSON, constructs `MotherResponse`, and returns `.dict()`; without a schema it returns the raw string (legacy llm_module.py:71-83).
- Error funnel: `requests.exceptions.RequestException` and bare `Exception` are both caught, printed, and fall through to `return None` (legacy llm_module.py:85-89).

```
                       structured_output?
                        /              \
                     yes                no
                      |                  |
        stream := False, add        stream stays as passed
        response_format(json_schema)      |
                      |            +------+------+
                      |            |             |
                      v         stream=True   stream=False
              POST (no stream)      |             |
                      |        SSE loop:      POST once
              parse JSON content   strip 6 chars,   |
                      |            print delta,     |
              MotherResponse(**x)  collect          |
                      |                 |           |
                  .dict()             str         raw str
                      \                |           /
                       \--------- or None on any error
```

## Imports

Third-party: `requests`.
Standard library: `json`; `typing.Optional`, `typing.Dict`, `typing.Any`.
Project-internal (legacy tree): `MotherResponse` from `schemas`.

## Defects and gaps

- Return annotation is `Optional[Dict[str, Any]]`, but two of the three success paths return a `str` — the streaming join (legacy llm_module.py:69) and the unstructured raw content (legacy llm_module.py:83). Only the structured path returns a dict (legacy llm_module.py:79).
- Pydantic version mixing: `model_json_schema()` is the v2 API (legacy llm_module.py:27) while `.dict()` is the v1-style API kept only as a deprecated alias in v2 (legacy llm_module.py:79). Under v1 the `model_json_schema()` call fails outright.
- SSE parsing blindly slices `chunk[6:]` on every non-empty line (legacy llm_module.py:59). Any line not prefixed by exactly `"data: "` is silently corrupted and then discarded by the decode-error handler (legacy llm_module.py:65-66), so a protocol mismatch is indistinguishable from an empty response.
- Streaming failure is indistinguishable from success: if no chunk yields content the function returns `""`, not `None` (legacy llm_module.py:69).
- `data.get("choices", [{}])[0]` (legacy llm_module.py:60, 75) only supplies its `[{}]` default when the key is *absent*. A response carrying `"choices": []` raises `IndexError`, and a decoded chunk that is not a dict raises `AttributeError` on `.get`; neither is a `JSONDecodeError`, so both bypass the per-line handler and abort the whole call through the bare `except Exception`.
- The inner handler catches only `JSONDecodeError` (legacy llm_module.py:80), but `MotherResponse(**...)` sits inside the same `try`. Any construction/validation failure it raises escapes to the bare `except Exception` (legacy llm_module.py:87) and prints "Unexpected error" instead of the intended "Failed to parse structured LLM response". A top-level JSON value that is not a mapping raises `TypeError` down the same path.
- The bare `except Exception` (legacy llm_module.py:87) also masks programming errors — `NameError`, `AttributeError`, a wrong `MotherResponse` binding — as a plain `None` return.
- `server_url` defaults to `http://0.0.0.0:1234` (legacy llm_module.py:15). `0.0.0.0` is a bind-side wildcard, not a connect address; it resolves to localhost on some stacks and fails on others.
- Hardcoded magic values with no override: the 60-second timeout appears twice (legacy llm_module.py:53, 72) and the six-character SSE prefix length (legacy llm_module.py:59). `max_tokens=-1` is forwarded verbatim into the payload (legacy llm_module.py:13, 38) with no validation; negative token limits are an LM Studio "unlimited" convention rather than a portable OpenAI-spec value.
- `"strict": True` (legacy llm_module.py:26) is asserted over whatever `model_json_schema()` returns; the schema is neither inspected nor post-processed before being sent. Whether a given server accepts that raw schema under strict mode depends on the server's json-schema rules and is not verifiable from this file.
- Streaming mode unconditionally writes tokens and a trailing newline to stdout (legacy llm_module.py:64, 68); no parameter suppresses it, so the transport is fused to a stdout side effect.
- The `Any` binding from `typing` is used only inside the inaccurate return annotation.

## Notes

- The whole legacy module is one function: no client object, no session reuse, no retry loop, no backoff, no connection pooling, and no `logging` usage.
- `structured_output` is hard-wired to `MotherResponse`; there is no parameter for a different schema, so any other structured shape requires editing the file.
- The return contract cannot separate failure from emptiness: every error path yields `None` and a contentless stream yields `""`, with the only diagnostic detail going to stdout.
