# emotional_embedding.py

A 23-line module defining a single `nn.Module` that turns text into a semantic embedding plus two scalar affect values. It delegates all text encoding to an external `get_embeddings` call and adds one trainable linear head on top. There is no training loop, no loss, no persistence, and no `__main__` in this file.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `EmotionalEmbedder` | class (`nn.Module`) | Constructed with no arguments; `forward(text_input)` returns a dict of tensors. |
| `EmotionalEmbedder.__init__` | method | Creates `valence_proj = nn.Linear(768, 4)` and re-initialises its **weight** in place from N(mean=0.0, std=0.02). Bias is left at PyTorch's default. emotional_embedding.py:6-9 |
| `EmotionalEmbedder.forward` | method | Encodes `text_input`, projects, returns `{'semantic_embedding', 'valence', 'arousal'}`. emotional_embedding.py:11-23 |

## Key behaviour

- `forward` passes `text_input` unmodified to `get_embeddings` (emotional_embedding.py:12). No type check, no batching logic, no truncation — whatever the caller supplies goes straight through.
- The return value of `get_embeddings` is assumed to be an **iterable of mappings**, each carrying an `'embedding'` key; the list comprehension at emotional_embedding.py:14 is the only place that assumption is expressed.
- The stacked list is converted with `torch.tensor(...)` at emotional_embedding.py:13-17 using `device='cuda'` and `dtype=torch.float32`. This constructs a **leaf tensor with `requires_grad=False`** — no gradient can flow back past this point, so only `valence_proj` is ever trainable here.
- Shapes: input list of length B, each embedding assumed length 768 → `embeddings` is `[B, 768]`; `valence_proj(embeddings)` is `[B, 4]`; after `torch.sigmoid` every element lies in (0, 1).
- Only columns 0 and 1 of the `[B, 4]` projection are read (emotional_embedding.py:21-22). Columns 2 and 3 are computed and discarded on every call.
- Outputs: `semantic_embedding` `[B, 768]` (the raw encoder output, unmodified), `valence` `[B]`, `arousal` `[B]`, both in (0, 1) — never negative, despite "valence" conventionally being a signed axis.

```
text_input
   |
   v
get_embeddings(text_input)        # external, not read here
   |  -> [{'embedding': [...]}, ...]
   v
torch.tensor(..., device='cuda', float32)   ---> 'semantic_embedding'  [B,768]
   |
   v
nn.Linear(768, 4)  ->  sigmoid   ->  [B,4]
                                      |  col 0 -> 'valence'  [B]
                                      |  col 1 -> 'arousal'  [B]
                                      +- col 2,3 -> discarded
```

## Imports

- Third-party: `torch`, `torch.nn` (as `nn`).
- `get_embeddings` from `text_embed` (emotional_embedding.py:3) — a bare top-level module name, imported absolutely. Whether it resolves to a sibling project module or an installed package is unverifiable from this file alone; its return contract is only inferred from the usage here.

## Defects and gaps

- **Hardcoded CUDA device**, emotional_embedding.py:15. On a machine without CUDA this raises at runtime. There is no `.to(next(self.parameters()).device)` and no fallback.
- **Device mismatch is likely by construction.** `embeddings` is forced onto `cuda`, but `valence_proj` lives wherever the caller placed the module. If the module was never moved to GPU, emotional_embedding.py:18 raises a device-mismatch `RuntimeError`. The module places data, not parameters — the inverse of the normal PyTorch contract.
- **Half the projection head is dead compute.** The layer really does produce 4 dimensions as its comment says (emotional_embedding.py:8), but only columns 0 and 1 are returned (emotional_embedding.py:21-22). Columns 2 and 3 reach no output, so they can receive no gradient from any loss built on this module's return value.
- **Magic dimension 768** is hardcoded at emotional_embedding.py:8 with no assertion against the actual embedding width. Any encoder returning a different width fails inside the matmul with a shape error rather than a clear message.
- **Magic init constant.** `std=0.02` at emotional_embedding.py:9 overwrites PyTorch's default `Linear` initialisation with an unexplained fixed value; it is not derived from the 768 input width and is not configurable.
- **Sigmoid squashes valence into (0, 1)**, emotional_embedding.py:18 — the name implies a bipolar quantity, so no output can express negative valence. Nothing in the file rescales it.
- **No input validation and no error handling anywhere.** An empty `embed_result` builds a 1-D tensor of shape `[0]`, not `[0, 768]`, so emotional_embedding.py:18 fails with an opaque shape `RuntimeError` instead of returning empty results; items missing `'embedding'` raise a bare `KeyError`; ragged embedding lengths raise a bare `ValueError` from `torch.tensor`. No `try`/`except` exists in the file.
- **The batch assumption is unguarded.** `valence_arousal[:, 0]` (emotional_embedding.py:21) requires a 2-D projection, which in turn requires `get_embeddings` to return a sequence of mappings. Any single-mapping or non-sequence return degrades into a `TypeError`/`KeyError` inside the comprehension at emotional_embedding.py:14 with no diagnostic.
- **Bias not initialised** alongside the weight, emotional_embedding.py:9. Only `.weight.data` is overwritten; the bias keeps the default uniform init, so the deliberate small-std scheme is only half applied.
- **Nothing in this file instantiates or calls `EmotionalEmbedder`.** Whether it is reachable from a runtime entry point is unverifiable from this file alone.

## Notes

- `.weight.data.normal_(...)` mutates the tensor's `.data` directly rather than using `torch.no_grad()` or `nn.init.normal_`; functionally equivalent here but bypasses autograd bookkeeping conventions.
- Building a tensor from a Python list of embeddings is a per-call host-side copy; there is no caching, no pinned memory, and no reuse of a preallocated buffer.
- `valence` and `arousal` are basic-slice **views** into `valence_arousal`, so holding either keeps the whole `[B, 4]` tensor alive; `semantic_embedding` is the same object as `embeddings`, aliased to the caller rather than copied.
- The class holds no state between calls beyond its single `Linear`, and `forward` reads no globals other than the imported `get_embeddings`.
