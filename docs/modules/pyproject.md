# pyproject.toml

Packaging and tooling manifest for the `neuralchild` distribution. Declares a setuptools build, five runtime dependencies, a `dev` extras group, one console-script entry point, and formatter/import-sorter settings. It is pure declarative configuration — no executable logic, no `[tool.pytest]`, `[tool.mypy]`, or `[tool.ruff]` sections.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `[build-system]` | table | setuptools>=42 + wheel, backend `setuptools.build_meta` (pyproject.toml:1-3) |
| `project.name` | constant | Distribution name `neuralchild` (pyproject.toml:6) |
| `project.version` | constant | Static `0.1.0`; no dynamic/VCS versioning (pyproject.toml:7) |
| `project.description` | constant | "A psychological brain simulation" (pyproject.toml:8) |
| `project.readme` | constant | Points at `README.md` (pyproject.toml:9) |
| `project.license` | constant | Table form `{file = "LICENSE"}` (pyproject.toml:11) |
| `project.classifiers` | constant | Python 3, MIT License, OS Independent (pyproject.toml:12-16) |
| `project.dependencies` | constant | torch>=2.0.0, pydantic>=2.0.0, requests>=2.25.0, pyyaml>=6.0, numpy>=1.20.0 (pyproject.toml:17-23) |
| `project.requires-python` | constant | `>=3.8`, no upper bound (pyproject.toml:24) |
| `optional-dependencies.dev` | constant | pytest>=7.0.0, black>=22.0.0, isort>=5.0.0, mypy>=0.9.0 (pyproject.toml:26-32) |
| `project.scripts.neuralchild` | console script | Installs `neuralchild` → `neuralchild.cli:main` (pyproject.toml:35) |
| `[tool.setuptools].packages` | constant | Explicit list containing exactly `["neuralchild"]` (pyproject.toml:37-38) |
| `[tool.black]` | table | line-length 88, target `py38` (pyproject.toml:40-42) |
| `[tool.isort]` | table | profile `black`, line_length 88 (pyproject.toml:44-46) |

## Key behaviour

- Build path: any PEP 517 frontend reads `[build-system]`, installs setuptools>=42 and wheel, and invokes `setuptools.build_meta`. No `setup.py` or `setup.cfg` is referenced from here.
- Package discovery is **explicit, not automatic**: because `[tool.setuptools].packages` is given as a literal list, setuptools does not run auto-discovery. Only the top-level `neuralchild` package is packaged; any sibling top-level package or any subpackage of `neuralchild` is excluded from the built wheel/sdist unless listed. Whether such packages exist is unverifiable from this file alone.
- No `package-dir` mapping is declared, so a flat layout (`neuralchild/` directly beside pyproject.toml) is assumed. A `src/` layout would break the build with this configuration.
- The single entry point is the `neuralchild` console script; installation generates a wrapper that imports `neuralchild.cli` and calls `main()`. Existence of that module and callable is unverifiable from this file alone.
- Style settings are mutually consistent: black line-length 88 matches isort `line_length = 88` with `profile = "black"`, so the two tools will not fight over formatting.
- Dependency floors are all lower bounds with no ceilings and no environment markers — resolution is fully open-ended toward newer releases of torch, pydantic, numpy, requests, and pyyaml.

## Imports

Not a Python module; declares dependencies rather than importing.

- Third-party (runtime): torch, pydantic, requests, pyyaml, numpy.
- Third-party (dev extras): pytest, black, isort, mypy.
- Build-time: setuptools, wheel.
- Project-internal reference: the module path `neuralchild.cli` and its `main` attribute, named by the console script.

## Defects and gaps

- pyproject.toml:24 vs 18 — `requires-python = ">=3.8"` is paired with an open-ended `torch>=2.0.0`. Both are lower bounds only, so a 3.8 install resolves backwards to whatever early 2.x still ships 3.8 wheels rather than failing; the declared support window and the torch version actually installed are not pinned to agree.
- pyproject.toml:38 — the explicit one-element `packages` list is a common cause of "installs but imports fail": subpackages are omitted from the distribution. There is no `[tool.setuptools.packages.find]` section to compensate.
- pyproject.toml:11 and 14 — the `license = {file = ...}` table form and the `License :: OSI Approved :: MIT License` classifier are both superseded by PEP 639 (SPDX expression plus `license-files`). `[build-system]` sets no upper bound on setuptools, so the build backend that resolves is free to be one that rejects this metadata form.
- pyproject.toml:31 — `mypy>=0.9.0` names a release number mypy never published (the 0.9 series was numbered 0.900/0.910). Under PEP 440 the release segment `(0,9,0)` sorts below `(0,900)`, so the floor sits far lower than the digits suggest.
- pyproject.toml:2 — `requires = ["setuptools>=42", "wheel"]` still lists `wheel` explicitly. `setuptools.build_meta` supplies its own wheel handling, so the entry is legacy carry-over from the `setup.py bdist_wheel` era and pins nothing useful.
- No `[tool.setuptools.package-data]` and no `include-package-data`, so only `.py` files under `neuralchild` reach the built distribution. Any non-Python resource the package needs at runtime is dropped by an install; whether such resources exist is unverifiable from this file alone.
- pyproject.toml:28-31 — pytest and mypy are declared as dev dependencies but the file contains no `[tool.pytest.ini_options]` (no testpaths, no rootdir hints) and no `[tool.mypy]` section, so both run on defaults.
- pyproject.toml:9 and 11 — `README.md` and `LICENSE` are referenced as build inputs; their presence is unverifiable from this file alone, and a missing one aborts the build.
- pyproject.toml:12-16 — classifiers omit per-minor `Programming Language :: Python :: 3.x` entries, so declared Python support is not visible in package metadata beyond `requires-python`.
- No `[project.urls]`, no `keywords`, no author email, and the author is a placeholder-style "NeuralChild Team" (pyproject.toml:10).

## Notes

- `[project.scripts]` is the only executable interface named anywhere in this file; everything else is metadata. Whether the CLI actually works must be verified against the `neuralchild` package itself.
- Nothing here configures a linter, test runner, or type checker beyond formatting, so no tool-config precedence conflict exists within this file.
