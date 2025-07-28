@echo off
echo Setting up Playwright environment variables for Windows...

REM Browser configuration
set PLAYWRIGHT_BROWSERS_PATH=%USERPROFILE%\AppData\Local\ms-playwright
set PLAYWRIGHT_HEADLESS=1
set PLAYWRIGHT_CHROMIUM_SANDBOX=0
set PLAYWRIGHT_LAUNCH_TIMEOUT=60000

REM Skip browser download if already installed
set PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
set PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1

REM Debugging (comment out for production)
REM set DEBUG=pw:*
REM set PLAYWRIGHT_DEBUG=1

REM Windows-specific fixes
set WER_RUNTIME_EXCEPTION_DEBUGGER_DISABLED=1

REM Temp directory
if not exist "C:\temp" mkdir C:\temp
set TEMP=C:\temp
set TMP=C:\temp

echo Environment variables set successfully!
echo.
echo Key settings:
echo - PLAYWRIGHT_HEADLESS=1 (no GUI)
echo - PLAYWRIGHT_CHROMIUM_SANDBOX=0 (disable sandbox)
echo - PLAYWRIGHT_LAUNCH_TIMEOUT=60000 (60 second timeout)
echo - Custom temp directory: C:\temp
echo.
echo Run your API server now: python api_server.py
