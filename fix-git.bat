@echo off
cd /d "c:\Users\PC04\Desktop\qa-verify"

echo === Fixing git configuration ===
echo.

echo Step 1: Removing duplicate remote 'qa-verify-warisan'
git remote remove qa-verify-warisan
echo [OK] Removed qa-verify-warisan remote

echo.
echo Step 2: Ensuring 'origin' remote is correct
git remote remove origin
git remote add origin https://github.com/Plaryy/qa-verify-warisan.git
echo [OK] Configured origin remote

echo.
echo Step 3: Checking git status
git status --short

echo.
echo Step 4: Staging files
git add .

echo.
echo Step 5: Committing changes
git commit -m "chore: clean workspace and generalize prompts" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

echo.
echo Step 6: Pushing to origin
git push -u origin main

echo.
echo === FINAL STATUS ===
git status
echo.
echo === Remote configuration ===
git remote -v
