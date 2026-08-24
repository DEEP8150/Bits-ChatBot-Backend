To Run Backend locally : 

Frontend: npm run dev -- --host
Backend: npm run dev 
Backend/tts: first activate piper virtual environment : .\piper-env\Scripts\Activate.ps1, then run : python -m piper.http_server -m en_US-lessac-medium --port 5001 .
Run local LLM:  llama serve -hf bartowski/Qwen2.5-3B-Instruct-GGUF --port 8080

Text-To-Speech(TTS) and Speech-To-Text(STT) only works in localhost and not on any ip address.
