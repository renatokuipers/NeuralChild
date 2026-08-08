# text_embed.py

Source: legacy snapshot at scratchpad/neural-child/text_embed.py. All `text_embed.py:N` citations below refer to that snapshot file, not to any same-named module in the working tree.

Single-function HTTP client that requests text embeddings from an OpenAI-compatible `/v1/embeddings` endpoint, defaulting to a local LM Studio server. The whole module is 25 lines of code: one function, one POST, one dictionary lookup, and two `except` clauses (one `requests`-specific, one catch-all). It performs no batching, retrying, caching, validation, or vector post-processing.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `get_embeddings` | function | Takes `str` or `list[str]` plus optional `model` and `server_url`; POSTs to the endpoint and returns the parsed `data` list, or `[]` on any failure. |

Defaults baked into the signature (text_embed.py:7-8):

| Parameter | Default |
| --- | --- |
| `model` | `text-embedding-nomic-embed-text-v1.5` |
| `server_url` | `http://0.0.0.0:1234/v1/embeddings` |
| request timeout | `30` seconds (text_embed.py:15, not a parameter) |

## Key behaviour

- Input normalisation (text_embed.py:10): a `str` is wrapped into a one-element list; anything else (including a list, tuple, or `None`) is forwarded unchanged as the JSON `input` value. No type check beyond `isinstance(..., str)`.
- Request body is exactly `{"model": ..., "input": [...]}` — no `encoding_format`, no dimensions, no auth header.
- `.json()` is chained directly onto `requests.post(...)` (text_embed.py:12-16). `raise_for_status()` is never called, so a 4xx/5xx response body is parsed as if it were a success.
- Result extraction is `response.get("data", [])` (text_embed.py:18). The returned list elements are whatever the server sent; the function never inspects, sorts, or re-indexes them, so callers must rely on server-side ordering or a per-item index field to map results back to inputs.
- Failure path is uniform: print a message to stdout, return `[]`. No logging module, no re-raise, no error object.

```
text_input ──► [str? → wrap in list] ──► POST {model, input}  (timeout=30)
                                              │
                        ┌─────────────────────┼──────────────────────┐
                        ▼                     ▼                      ▼
                  .json() ok            RequestException        any Exception
                        │              (connect/timeout/DNS)    (e.g. AttributeError)
                        ▼                     │                      │
              resp.get("data", [])         print + []            print + []
                        │
                        ▼
                 list (unvalidated)
```

All terminal states can produce `[]`; the caller cannot tell success-with-no-data from a dead server.

## Imports

- Third-party: `requests`
- Standard library: `typing` (`List`, `Dict`, `Union`, `Optional`)
- Project-internal: none

## Defects and gaps

- text_embed.py:4 — `Optional` is imported but never used anywhere in the file.
- text_embed.py:8 — default `server_url` uses `0.0.0.0`, a wildcard bind address being used as a connect target. It resolves to localhost on some stacks and fails on others; it is not portable and there is no environment-variable or config override.
- text_embed.py:8 — the model id is hardcoded in the signature; changing the model served by LM Studio silently breaks every caller that relies on the default, and no availability check is made.
- text_embed.py:15 — the 30-second timeout is a literal, not a parameter. A large batch that exceeds it raises a timeout, which is caught and turned into `[]`, so the caller sees "no embeddings" rather than "too slow".
- text_embed.py:16 — no status check before `.json()`. An HTTP error whose body is valid JSON (for example an `error` object) flows into line 18, yields no `data` key, and returns `[]` with no message printed at all — the quietest failure in the file.
- text_embed.py:18 — if the response JSON is not a mapping (a bare list or string), `.get` raises `AttributeError`, caught by the broad handler at line 23.
- text_embed.py:20-25 — both handlers swallow the exception entirely and return the same sentinel `[]`. Errors go to stdout via `print`, so they cannot be filtered, levelled, or captured by a logging configuration.
- text_embed.py:23 — bare `except Exception` at a non-boundary function masks any programming error raised inside the `try`, including the `AttributeError` above. Which of the two handlers catches a JSON decode failure depends on the installed `requests` version and is not determinable from this file alone.
- The declared return type is `List[Dict]` (text_embed.py:8), but nothing validates that the extracted `data` elements are dictionaries (text_embed.py:18); the annotation is a promise the code does not enforce.
- text_embed.py:6, 10 — the `Union[str, List[str]]` hint is never enforced at runtime; a `None`, dict, or arbitrary object is passed straight into the request body with no validation and no error message.
- `get_embeddings` has no caller inside this file and there is no `__main__` block — expected for a module-level API, but nothing in this file exercises it.

## Notes

- An empty list input is forwarded as an empty `input` array; the function does not short-circuit, so a pointless HTTP round trip is made.
- The trailing comment on text_embed.py:8 is the only documentation; there is no module or function docstring.
- Whether the returned `data` shape matches what any consumer expects is unverifiable from this file alone.
