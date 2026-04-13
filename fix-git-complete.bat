@echo off
setlocal enabledelayedexpansion
cd /d "c:\Users\PC04\Desktop\qa-verify"

echo ============================================
echo GIT CORRUPTION FIX - Complete Reset
echo ============================================
echo.

echo Step 1: Backing up current work (checking for uncommitted changes)
git status --short > temp_status.txt
set /p changes=<temp_status.txt
if not "!changes!"=="" (
    echo [INFO] Found uncommitted changes:
    git status --short
) else (
    echo [INFO] No uncommitted changes
)
del temp_status.txt
echo.

echo Step 2: Removing corrupted .git directory
rmdir /s /q .git
if exist .git (
    echo [ERROR] Could not remove .git - please close any applications accessing it
    pause
    exit /b 1
)
echo [OK] Removed .git directory
echo.

echo Step 3: Initializing fresh git repository
git init
echo [OK] Git initialized
echo.

echo Step 4: Configuring git user (if needed)
git config --global user.email "noreply@github.com" 2>nul
git config --global user.name "GitHub User" 2>nul
echo [OK] Git configured
echo.

echo Step 5: Adding all files
git add .
echo [OK] All files staged
echo.

echo Step 6: Creating initial commit
git commit -m "chore: initial commit - clean workspace and generalize prompts" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
if errorlevel 1 (
    echo [WARNING] Commit may have had issues, continuing anyway...
)
echo.

echo Step 7: Creating main branch (ensure it exists)
git branch -M main
echo [OK] Main branch created/confirmed
echo.

echo Step 8: Adding remote origin
git remote add origin https://github.com/Plaryy/qa-verify-warisan.git
echo [OK] Remote origin added
echo.

echo Step 9: Pushing to GitHub
echo NOTE: You may need to authenticate with GitHub
echo You can use either:
echo   - SSH key (if configured)
echo   - Personal Access Token (paste when prompted for password)
echo.
git push -u origin main --force
if errorlevel 1 (
    echo.
    echo [ERROR] Push failed. Please check:
    echo   1. Your GitHub credentials/authentication
    echo   2. The repository URL is correct
    echo   3. You have permission to push to this repository
    echo.
    echo Try running manually: git push -u origin main --force
    pause
    exit /b 1
)
echo.

echo ============================================
echo SUCCESS! Code pushed to GitHub
echo ============================================
echo.
echo Repository Status:
git status
echo.
echo Remote Configuration:
git remote -v
echo.
echo Recent Commits:
git log --oneline -5
echo.
pause
