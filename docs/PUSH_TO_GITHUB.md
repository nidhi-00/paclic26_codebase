# Safely push this revision to the old repository

The safest method preserves the old repository's `.git` directory and replaces only its tracked working files.

Assume:

- old clone: `~/paclic26_codebase`
- extracted new folder: `~/Downloads/paclic26_codebase_complete`

Adjust those paths to your machine.

## 1. Check and back up the old repository

```bash
cd ~/paclic26_codebase
git status
git remote -v
git branch --show-current
```

Commit or stash any work you still need:

```bash
cd ~/paclic26_codebase
git add -A
git commit -m "checkpoint before reproducibility overhaul"
```

If there is nothing to commit, Git will say the tree is clean.

Create a new branch:

```bash
cd ~/paclic26_codebase
git switch -c paclic-reproducibility-overhaul
```

## 2. Replace working files while preserving `.git`

```bash
cd ~/paclic26_codebase
rsync -av --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='data/' \
  --exclude='results/' \
  ~/Downloads/paclic26_codebase_complete/ ./
```

The command deletes stale tracked working files but does not touch Git history, your virtual environment, raw data or local results.

## 3. Review the replacement

```bash
cd ~/paclic26_codebase
git status --short
git diff --stat
git diff -- README.md configs/analysis.yaml src/analyse.py
```

## 4. Recreate the environment and test

```bash
cd ~/paclic26_codebase
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest
./scripts/run_debug.sh
```

Do not commit `data/`, `results/`, model weights, the virtual environment or the raw GECO workbook.

## 5. Commit and push

```bash
cd ~/paclic26_codebase
git add -A
git commit -m "rebuild PACLIC surprisal pipeline for reproducible held-out evaluation"
git push -u origin paclic-reproducibility-overhaul
```

Open a pull request from `paclic-reproducibility-overhaul` into your default branch. In the PR description, link `docs/IMPLEMENTATION_REPORT.md` and mention that real GECO numbers must be regenerated.

## Alternative: initialise a fresh repository

Use this only when you do not need the old Git history:

```bash
cd ~/Downloads/paclic26_codebase_complete
git init
git add -A
git commit -m "initial reproducible PACLIC codebase"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```
