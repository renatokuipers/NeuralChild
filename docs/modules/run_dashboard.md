# run_dashboard.py

A 62-line standalone launcher script. It checks that six dashboard-related packages are importable, pip-installs any that are missing into the current interpreter, then spawns a separate Python process for a dashboard script located next to itself. It has no project-internal imports and defines no classes — it is pure process orchestration around `subprocess`.

## Public surface

| Name | Kind | Contract |
| --- | --- | --- |
| `REQUIRED_PACKAGES` | constant (list[str]) | Six distribution names: dash, dash-bootstrap-components, plotly, pandas, numpy, pydantic (run_dashboard.py:12) |
| `check_and_install_packages()` | function | Probes each name with `importlib.util.find_spec`; pip-installs the missing ones; returns None (run_dashboard.py:21) |
| `main()` | function | Runs the check, locates and blocks on the dashboard subprocess; returns 0 on launch, 1 if no dashboard file found (run_dashboard.py:37) |

Module guard at run_dashboard.py:61 wraps `sys.exit(main())` at run_dashboard.py:62.

## Key behaviour

- Distribution-to-module mapping is a single `replace('-', '_')` (run_dashboard.py:26). This happens to be correct for `dash-bootstrap-components`, and is a no-op for the other five.
- Detection is import-spec based, not version based: any importable module of that name satisfies the check, regardless of version, and no version constraints are passed to pip.
- Install is a single batched `pip install` invocation via `sys.executable -m pip` (run_dashboard.py:32), so packages land in the same interpreter that will later spawn the dashboard.
- Dashboard resolution is two-candidate, in fixed order, both anchored to `os.path.dirname(os.path.abspath(__file__))`:
  1. `neural_child_dashboard.py` (underscores) — run_dashboard.py:43
  2. `neural-child-dashboard.py` (hyphens) — run_dashboard.py:50
- Launch is `subprocess.call`, i.e. blocking and inheriting stdio; the parent stays alive for the dashboard's whole lifetime.
- No CLI arguments are read or forwarded — `sys.argv` is never consulted; the child process gets only the script path.

```
main()
  |
  +-- check_and_install_packages()
  |     for pkg in REQUIRED_PACKAGES:
  |       find_spec(pkg.replace('-','_')) is None -> queue
  |     queue non-empty -> pip install <queue>   (uncaught CalledProcessError on failure)
  |
  +-- path A = <script dir>/neural_child_dashboard.py
  |     exists -> subprocess.call([python, A])  --> return 0   (child exit code discarded)
  |
  +-- path B = <script dir>/neural-child-dashboard.py
        exists -> subprocess.call([python, B])  --> return 0   (child exit code discarded)
        else   -> print error naming path B only --> return 1
```

## Imports

Third-party: none.

Standard library: `subprocess`, `sys`, `os`, `importlib.util`. All four are used.

Project-internal: none. This file imports nothing from the repository.

## Defects and gaps

- **Child exit status is discarded.** `subprocess.call` returns the dashboard's exit code at run_dashboard.py:48 and run_dashboard.py:53, but the return value is dropped and `main()` unconditionally returns 0 at run_dashboard.py:59. A dashboard that crashes immediately produces a success exit status from the runner, so CI or a shell caller cannot detect the failure.
- **Error message names the wrong path.** The failure branch at run_dashboard.py:55 prints the value of `dashboard_path`, which by then has been reassigned to the hyphenated candidate (run_dashboard.py:50). The underscored candidate that was tried first is never mentioned, so the operator sees a filename the script does not primarily look for.
- **`check_call` failure is unhandled.** If pip fails (no network, resolution conflict, read-only environment), run_dashboard.py:32 raises `CalledProcessError` and the script dies with a raw traceback; the "installed successfully" print at run_dashboard.py:33 is only reachable on success, so it does not lie, but there is no diagnostic path.
- **Unconditional install into the active interpreter.** run_dashboard.py:32 mutates whatever environment is running the script, including a system Python, with no virtualenv detection, no `--user`, and no confirmation.
- **Docstring overstates the guarantee.** run_dashboard.py:22 says the function checks packages "and install them if needed", but the check is name-presence only — a shadowing local file or package directory named e.g. `numpy` on `sys.path` satisfies `find_spec` and suppresses the install.
- **Hardcoded dependency list.** REQUIRED_PACKAGES (run_dashboard.py:12) is duplicated knowledge with no link to any packaging metadata; nothing in this file keeps it in sync with what the dashboard actually imports.
- **Child working directory is not controlled.** Neither `subprocess.call` passes `cwd` (run_dashboard.py:48, run_dashboard.py:53), so the dashboard inherits whatever directory the runner was invoked from, even though its script path was resolved against `__file__`. Whether the dashboard depends on relative paths is unverifiable from this file alone.
- **Both launch branches are duplicates.** run_dashboard.py:47-48 and run_dashboard.py:52-53 repeat the same print string and the same `subprocess.call` argument list (differing only in indentation), relying solely on which candidate `dashboard_path` currently holds.
- No unreferenced definitions: both functions are called within this file.

## Notes

- Whether either `neural_child_dashboard.py` or `neural-child-dashboard.py` exists in the repository is unverifiable from this file alone, as is whether the two-candidate fallback ever resolves.
- The hyphenated candidate at run_dashboard.py:50 is only ever handed to the interpreter as a script path, never imported, so its non-identifier filename does not matter to this file.
- `main()` returning 1 is the only non-zero exit; there is no distinct code for install failure versus missing dashboard.
