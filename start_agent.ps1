# Start Ollama (minimised)
Start-Process powershell -ArgumentList "-NoExit -Command ollama serve" -WindowStyle Minimized

Start-Sleep -Seconds 5

# Start dashboard from the current folder
Start-Process powershell -ArgumentList "-NoExit -Command cd `"$PSScriptRoot`"; py web_universal.py"

Start-Sleep -Seconds 8
Start-Process "http://127.0.0.1:5000"