#!/bin/bash
# One-time: init (if needed), create dev branch, initial commit, push to origin.
set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  git init
fi
git remote add origin git@github.com:Sentience-Robotics/speech-ai-pipeline.git 2>/dev/null || true
git checkout -b dev 2>/dev/null || git checkout dev
git add -A
git status
if git diff --cached --quiet; then
  echo "Nothing to commit (already committed?)."
else
  git commit -m "Initial base: config, run script, README, requirements (generic interface only)"
fi
git push -u origin dev
echo "Done: dev branch pushed to origin."
