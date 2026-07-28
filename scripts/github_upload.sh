#!/usr/bin/env bash
# Safe first-time GitHub upload. Refuses to run if data/large files would commit.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
echo "=== Repo: $REPO ==="
if [ ! -f .gitignore ]; then
  echo "!! No .gitignore found. Aborting for safety."; exit 1
fi
if [ ! -d .git ]; then git init; echo "initialized git repo"; fi
git add .
echo "=== Checking staged files for large/data files... ==="
BIG=$(git diff --cached --name-only | while read -r f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt 52428800 ]; then echo "$f ($((sz/1048576))MB)"; fi
done)
DATA=$(git diff --cached --name-only | grep -iE '\.(pth|pt|npy|mat|tar|jpe?g|png|h5)$' || true)
if [ -n "$BIG" ] || [ -n "$DATA" ]; then
  echo "!! REFUSING TO COMMIT — data/large files staged:"
  [ -n "$BIG" ]  && echo "  LARGE:" && echo "$BIG"
  [ -n "$DATA" ] && echo "  DATA:" && echo "$DATA"
  echo "  Fix .gitignore, git rm --cached <file>, re-run. Aborting."; exit 1
fi
echo "=== Safe: only code/text staged. Files: ==="
git diff --cached --name-only | head -40
echo "Total staged: $(git diff --cached --name-only | wc -l) files"
echo ""
echo "=== Next (run manually after reviewing): ==="
echo "  git commit -m 'Initial commit: EEG cross-subject decoding pipeline'"
echo "  git branch -M main"
echo "  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
echo "  git push -u origin main"
