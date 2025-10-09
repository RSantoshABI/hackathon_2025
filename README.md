## Hackathon 2025 Solutions by MatchaSpeed

This repository contains solutions for the 2025 RGM Hackathon challenges.

Primary problem statement addressed:
1. Build data‑driven models to optimize portfolio pricing by understanding price elasticity and cannibalization — maximize MACO with smarter pricing.

Relevant solution components:
- `pricing/` – Pricing & elasticity modeling (statistical / ML layer)
- `optimization/` – Price optimization engine & methodology (detailed README inside that folder)
- `optimization_app/` – Lightweight web application wrapper for running the optimization interactively

---

## 🚀 Quick Start: Launch the Web App

Once dependencies are synced (see Setup below), start the app with:

```powershell
uv run optimization_app/launch.py
```

This will:
1. Use `uv` to invoke Python inside the project’s managed virtual environment
2. Load configuration from `optimization_app/config.py`
3. Start the interactive interface (CLI / Streamlit / FastAPI style depending on implementation)

If the app exposes a local URL, open it in your browser after the command starts.

---

## 🧩 Prerequisites

- Python 3.11+ (the project specifies versions in `pyproject.toml`)
- Git
- `uv` (ultra‑fast Python package & environment manager) – replaces manual `pip` + `venv`
- (Optional) Jupyter / VS Code for exploring notebooks

> Note: All dependency management is done with `uv` using the locked set in `uv.lock`.

---

## 🛠️ Setup (One-Time)

### 1. Clone the repository
```powershell
git clone <REPO_URL> hackathon_2025
cd hackathon_2025
```

### 2. Install `uv`
Choose one of the methods below (PowerShell examples shown):

```powershell
# Recommended (official installation script)
irm https://astral.sh/uv/install.ps1 | iex

# OR via pip (falls back to existing environment)
pip install uv

# Verify
uv --version
```

### 3. Sync dependencies
This creates (or reuses) a `.venv` at the project root and installs the exact versions from `uv.lock`.

```powershell
uv sync
```

If you add new packages later:
```powershell
uv add <package_name>
```
(`uv` will update `pyproject.toml` and refresh the lock file.)

### 4. (Optional) Activate the virtual environment
Most commands can be run with `uv run` without activation, but you can also:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## ▶️ Running Components

### Web App
```powershell
uv run optimization_app/launch.py
```

### Notebooks
Install Jupyter if not present (only once):
```powershell
uv add notebook
uv run jupyter notebook
```

### Tests (if desired)
```powershell
uv run pytest -q
```

---

## 📂 Project Structure (High-Level)
```
pricing/               # Pricing & elasticity modeling utilities
optimization/          # Optimization engine (detailed README inside)
optimization_app/      # App launcher & config
optimization_data/     # Input data samples / references
data/                  # Additional raw / supporting data
pyproject.toml         # Project metadata & dependencies
uv.lock                # Locked dependency set
```

---

## 📘 Optimization Engine Docs
All deep technical details (architecture, constraints, caching, methodology, input formats, troubleshooting) live in:

`optimization/README.md`

Refer there when you need to:
- Understand constraint formulations
- Extend or embed the optimizer
- Run programmatic (non‑app) optimization workflows

---

## 🔄 Updating Dependencies
```powershell
# Add a library
uv add pandas

# Remove a library
uv remove <package>

# Re-sync after pulling upstream changes
uv sync
```

---

## ❓ Troubleshooting Basics

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `uv sync` again (out-of-date env) |
| Lock / dependency drift | Delete `.venv` then `uv sync` |
| Wrong Python version | Ensure `python --version` matches required version; reinstall `uv` if needed |
| App won't start | Check `optimization_app/config.py` and any required data file paths |

---

## 🔐 Licensing / Use
Internal hackathon project – not for external distribution.

---

## 📝 At a Glance
- Use `uv sync` once to install everything.
- Launch app with `uv run optimization_app/launch.py`.
- Dive deeper via `optimization/README.md`.

Happy optimizing! 🎯
