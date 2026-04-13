@echo off
cd /d "c:\Users\PC04\Desktop\qa-verify"

echo === Checking if git repo exists ===
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo === Initializing git repository ===
    git init
)

echo === Ensuring branch is main ===
git symbolic-ref refs/heads/main HEAD >nul 2>&1
if errorlevel 1 (
    git checkout -b main 2>nul
    if errorlevel 1 (
        git branch -m main 2>nul
    )
)

echo === Configuring remote origin ===
git remote remove origin >nul 2>&1
git remote add origin https://github.com/Plaryy/qa-verify-warisan.git

echo === Staging all files ===
git add .

echo === Committing with message and trailer ===
git commit -m "chore: clean workspace and generalize prompts" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
set commitStatus=%errorlevel%

echo === Pushing to origin main ===
git push -u origin main
set pushStatus=%errorlevel%

echo.
echo === FINAL STATUS ===
if %commitStatus% equ 0 if %pushStatus% equ 0 (
    echo SUCCESS: Committed and pushed successfully
) else (
    if not %commitStatus% equ 0 echo FAILED at: git commit
    if not %pushStatus% equ 0 echo FAILED at: git push -u origin main
)

echo.
echo === git status --short ===
git status --short

echo.
echo === git remote -v ===
git remote -v

echo.
echo === git rev-parse --abbrev-ref HEAD ===
git rev-parse --abbrev-ref HEAD
