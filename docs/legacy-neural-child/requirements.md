# requirements.txt

Flat pip requirements file for the legacy neural-child codebase. Eight lines, each an exact `==` pin of a top-level package, with no comments, no sections, no extras, no environment markers, and no options lines. It declares the intended install set only; it does not lock transitive dependencies and carries no hashes.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| torch | pin (==2.0.1) | requirements.txt:1 — tensor/autograd runtime; no index directive accompanies it |
| transformers | pin (==4.38.0) | requirements.txt:2 — HuggingFace model/tokenizer library |
| numpy | pin (==1.24.3) | requirements.txt:3 — array numerics, pinned in the 1.x line |
| python-dateutil | pin (==2.8.2) | requirements.txt:4 — date parsing/arithmetic |
| wandb | pin (==0.15.4) | requirements.txt:5 — experiment tracking client |
| accelerate | pin (==0.22.0) | requirements.txt:6 — device placement / mixed-precision launcher for transformers |
| bitsandbytes | pin (==0.41.1) | requirements.txt:7 — 8-bit/4-bit quantized optimizers and linear layers |
| gradio | pin (==5.11.0) | requirements.txt:8 — web UI server |

## Key behaviour

- Purely declarative. `pip install -r requirements.txt` resolves eight direct requirements; every one uses strict equality, so pip has zero version freedom on these names and full freedom on everything they pull in.
- No `--index-url`, `--extra-index-url`, `--find-links`, `-r`, `-c`, or `--hash` directives appear anywhere in the file, so no alternate wheel source (e.g. a CUDA-specific torch index) and no constraint file is wired in from here.
- No environment markers (`sys_platform`, `python_version`, `platform_machine`) on any line, so the same set is demanded on every OS and interpreter version.
- By role the eight names fall into four groups, with no constraint tying any group to another: model runtime (torch, transformers, accelerate, bitsandbytes); numerics (numpy); utility (python-dateutil); observability and UI (wandb, gradio).

## Imports

Not applicable — this is a requirements manifest, not Python source. It contains no import statements. The packages it names are third-party; no project-internal or editable (`-e`) entries are present.

## Defects and gaps

- requirements.txt:1 — torch is pinned with no accompanying index directive, so nothing in this file selects a particular build variant. Which variant (CPU, or a specific CUDA build) was intended is unverifiable from this file alone.
- requirements.txt:7 — bitsandbytes is demanded unconditionally with no environment marker, so the same pin is required on every platform the file is used on. The file provides no marker or fallback to vary it.
- requirements.txt:3, requirements.txt:8 — the set mixes a gradio 5.x pin with a numpy 1.x pin. Because every entry uses strict `==`, any transitive requirement for a different numpy makes the whole set unresolvable; the file provides no constraint file or range to absorb that.
- Version generations across the set are widely separated (transformers 4.38.0 and gradio 5.11.0 against accelerate 0.22.0, wandb 0.15.4 and torch 2.0.1). Nothing in the file records that this combination was ever resolved together; whether it installs cleanly is not determinable from the manifest.
- No `pip` / `setuptools` / `wheel` pin and no `--hash` entries, so the install is not reproducible even though the eight direct names are exactly pinned.
- No Python version is declared or constrained anywhere in the file.
- Whether every listed package is actually imported by the codebase — and whether packages the codebase imports are missing from this list — cannot be determined from this file alone.

## Notes

- One requirement per line with no blank-line grouping and no trailing comments, so line N maps 1:1 to package N.
- Strict `==` on all eight names means the set has no slack: adding one more requirement can make it unsolvable.
