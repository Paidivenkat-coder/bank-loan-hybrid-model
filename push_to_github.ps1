param (
    [Parameter(Mandatory=$false)]
    [string]$RepoUrl
)

# Set working directory to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Add local git shim to PATH
$env:PATH = "C:\Users\venkat\.local\bin;$env:PATH"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "   Bank Loan Hybrid Model - Push to GitHub Utility   " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

if (-not $RepoUrl) {
    $RepoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/your-username/bank-loan-hybrid-model.git)"
}

if (-not $RepoUrl) {
    Write-Host "No repository URL provided. Aborting." -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/3] Checking Git remote configuration..." -ForegroundColor Yellow
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "Updating origin remote to: $RepoUrl" -ForegroundColor Green
    git remote set-url origin $RepoUrl
} else {
    Write-Host "Adding origin remote: $RepoUrl" -ForegroundColor Green
    git remote add origin $RepoUrl
}

Write-Host "`n[2/3] Verifying branch and commit status..." -ForegroundColor Yellow
git branch -M main
git status --short

Write-Host "`n[3/3] Pushing to GitHub (origin main)..." -ForegroundColor Yellow
Write-Host "If prompted, please authenticate in the browser or terminal." -ForegroundColor DarkGray
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS! Your project has been pushed to GitHub." -ForegroundColor Green
    Write-Host "You can now deploy it on Streamlit Community Cloud:" -ForegroundColor Cyan
    Write-Host "1. Visit https://share.streamlit.io" -ForegroundColor White
    Write-Host "2. Click 'New App'" -ForegroundColor White
    Write-Host "3. Set Main file path: app/app.py" -ForegroundColor White
    Write-Host "4. Click Deploy!" -ForegroundColor White
} else {
    Write-Host "`nPush encountered an issue. Please verify your GitHub credentials and repository permissions." -ForegroundColor Red
}
