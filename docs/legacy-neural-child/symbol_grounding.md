# symbol_grounding.py

A 29-line module holding a single class that maps text concepts to embedding vectors and back to string tokens. It stores every added symbol both in two Python dicts and in a growing CUDA tensor matrix, and resolves an arbitrary embedding to the nearest stored token by dot-product argmax. No persistence, no configuration, no logging; nothing in the file calls anything else in the file.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `SymbolGrounding` | class | Holds `concept_map`, `reverse_map`, `embedding_matrix`; no constructor arguments (symbol_grounding.py:4) |
| `SymbolGrounding.add_symbol(concept, token)` | method | Embeds `concept`, stores it under key `token` and reverse-keys it by the embedding tuple, appends the row to the matrix (symbol_grounding.py:10) |
| `SymbolGrounding.get_token(embedding)` | method | Returns the token whose stored row maximises the dot product with `embedding` (symbol_grounding.py:16) |
| `SymbolGrounding.batch_ground(concepts)` | method | Returns `{concept: {'token': '[UPPERCASE]', 'embedding': tensor}}`; stores nothing (symbol_grounding.py:21) |

## Key behaviour

- State at construction (symbol_grounding.py:6-8): `concept_map` dict, `reverse_map` dict, `embedding_matrix` an empty CUDA tensor of shape `(0, 768)`. Embedding width 768 and device `cuda` are hardcoded.
- `add_symbol` takes the first element of the `get_embeddings` return, reads its `'embedding'` key, and wraps it in a CUDA tensor (symbol_grounding.py:11). It passes a bare string; `batch_ground` passes a list (symbol_grounding.py:22) — whether `get_embeddings` accepts both shapes is unverifiable from this file alone.
- Forward map is keyed by `token`, not by `concept`; the concept string is only used to produce the vector. So `add_symbol("dog", "[ANIMAL]")` stores the dog vector under `[ANIMAL]`.
- Reverse map key is `tuple(embedding.cpu().numpy())` — one NumPy scalar per embedding dimension (symbol_grounding.py:13). Every lookup rehashes the whole vector.
- `get_token` shapes: matrix `(N, 768)` matmul vector `(768,)` gives similarities `(N,)`; `argmax` picks a row index, and that row is round-tripped to CPU/NumPy/tuple to index `reverse_map` (symbol_grounding.py:17-19). This is a raw dot product, not cosine — no normalisation anywhere in the file, so longer vectors win regardless of direction.
- Matrix growth is a full `torch.cat` reallocation per call (symbol_grounding.py:14): O(N²) copying over N insertions.
- `batch_ground` mutates no instance state: it synthesises tokens as the uppercased concept in square brackets and never touches `concept_map`, `reverse_map`, or `embedding_matrix` (symbol_grounding.py:23-28).

```
add_symbol(concept, token)
  concept ─► get_embeddings ─► [0]['embedding'] ─► tensor(cuda)
                                                    ├─► concept_map[token]        (never read in this file)
                                                    ├─► reverse_map[tuple(np)] = token
                                                    └─► cat onto embedding_matrix (N,768)

get_token(embedding(768,))
  embedding_matrix (N,768) · embedding ─► sims (N,) ─► argmax ─► row ─► tuple(np) ─► reverse_map ─► token

batch_ground(concepts)
  concepts ─► get_embeddings ─► {concept: {'[CONCEPT]', tensor}}   ── isolated, writes no state
```

## Imports

- `torch` (symbol_grounding.py:1).
- `get_embeddings` from `text_embed` (symbol_grounding.py:2) — whether `text_embed` is a project module or an installed package is unverifiable from this file alone.

## Defects and gaps

- Hard CUDA dependency with no fallback: `device='cuda'` at symbol_grounding.py:8, :11, :26. Constructing the class at all fails on a machine without CUDA.
- Hardcoded 768 embedding width (symbol_grounding.py:8). If `get_embeddings` returns any other dimension, the `torch.cat` at symbol_grounding.py:14 raises a size-mismatch error on the very first `add_symbol`.
- `get_token` on an empty instance: `argmax` over the zero-length similarity tensor from a `(0, 768)` matrix raises rather than returning a sentinel (symbol_grounding.py:17-18). No emptiness guard.
- `get_token` has no device or shape check on its argument. A CPU tensor, or anything not shaped `(768,)`, raises inside `matmul` (symbol_grounding.py:17).
- `concept_map` is written at symbol_grounding.py:12 and never read anywhere in this file — the retrieval path uses `reverse_map` exclusively. It is dead state as far as this file shows.
- Reverse-map collision: two concepts producing bitwise-identical embeddings overwrite the same `reverse_map` key, silently losing the earlier token (symbol_grounding.py:13).
- Float-tuple keying is fragile by construction (symbol_grounding.py:13, :19). It only works because the lookup key is derived from the same stored row; any recomputed or transferred embedding, or any dtype change, misses the dict and raises `KeyError` with no handling.
- `batch_ground` token scheme `[CONCEPT]` (symbol_grounding.py:25) contradicts `add_symbol`, where the token is caller-supplied. Its outputs are unregistered, so `get_token` can never return a batch-generated token.
- `batch_ground` zips `concepts` against the `get_embeddings` result with no length check (symbol_grounding.py:27). A short result silently drops the trailing concepts instead of failing.
- `batch_ground` keys its output dict by concept (symbol_grounding.py:24), so duplicate entries in the input list silently collapse to one.
- Re-using a `token` in `add_symbol` overwrites `concept_map[token]` (symbol_grounding.py:12) but still appends a row (symbol_grounding.py:14) and adds a second `reverse_map` entry (symbol_grounding.py:13). The matrix row count then permanently exceeds `len(concept_map)`, and the superseded row stays a live match candidate for `get_token`.
- Append-only: no method removes or replaces a row in `embedding_matrix` or an entry in either dict, so the matrix only grows.
- Nothing in this file invokes `get_token` or `batch_ground`; `add_symbol` is likewise uncalled here. All three are unreferenced within the file.
- No exception handling of any kind: a malformed `get_embeddings` result (missing `'embedding'` key, empty list) surfaces as a raw `KeyError`/`IndexError` at symbol_grounding.py:11.
- No docstrings on the module, class, or any method, so the only contract is the type hints on symbol_grounding.py:10, :16, :21 — and `concepts: list` gives no element type.

## Notes

- The class is entirely in-memory: no save, load, or serialisation path exists, so the grounding table is rebuilt from scratch every process start.
- Nothing stores a row-index-to-token mapping. A matrix row resolves to a token only by re-tupling its float values and hitting `reverse_map`, so if the two ever diverge the row becomes unresolvable.
- Whether `text_embed.get_embeddings` returns a list of dicts with an `'embedding'` key, and what dimension it produces, is unverifiable from this file alone — both assumptions are baked into symbol_grounding.py:11 and :26.
