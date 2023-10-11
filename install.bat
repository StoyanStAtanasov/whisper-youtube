Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

pip install git+https://github.com/openai/whisper.git
pip install yt-dlp