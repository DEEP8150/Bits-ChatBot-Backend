## Run the Project Locally

### Frontend

```bash
npm run dev -- --host
```

### Backend

```bash
npm run dev
```

### Text-to-Speech Backend

First, activate the Piper virtual environment:

```powershell
.\piper-env\Scripts\Activate.ps1
```

Then start the Piper HTTP server:

```bash
python -m piper.http_server -m en_US-lessac-medium --port 5001
```

### Run the Local LLM

Start the local Qwen model using llama.cpp:

```bash
llama serve -hf bartowski/Qwen2.5-3B-Instruct-GGUF --port 8080
```

### Important

> **Text-to-Speech (TTS) and Speech-to-Text (STT) currently work only on `localhost` and not when accessing the application through a local network IP address.**
