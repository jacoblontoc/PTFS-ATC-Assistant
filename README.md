# PTFS ATC Assistant

A real-time Air Traffic Control assistant for flight simulator roleplay. It listens to your game or microphone audio, transcribes pilot transmissions, and instantly suggests realistic ATC responses — displayed in a clean chat-style interface.

Built for [PTFS (Plane Terror Flight Simulator)](https://www.roblox.com/games/857057492) on Roblox, and compatible with MSFS, Discord, or any audio source.

---

## Features

- **Real-time audio capture** — listens to system audio (game, Discord, etc.) via WASAPI loopback, or your microphone for testing
- **Fast local transcription** — uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base.en`) for low-latency speech-to-text, fully offline
- **Local AI responses** — powered by [Ollama](https://ollama.ai) running on your machine; no internet required after setup
- **ATC controller types** — switch between All, Departure, Ground, and Clearance modes, each with a tailored prompt
- **Model selector** — choose between `llama3.2:3b` (fast), `llama3.1:8b` (balanced), or `llama3.2-vision:11b` (vision-capable)
- **Chat UI** — pilot transmissions on the left, ATC suggestions on the right; one-click copy on every suggestion
- **Mic toggle** — flip between system loopback and microphone without restarting
- **Sensitivity slider** — adjust the voice activity detection threshold live

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or newer |
| [Ollama](https://ollama.ai) | Latest |
| Windows | 10 or 11 (WASAPI loopback) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PTFS-ATC-Assistant.git
cd PTFS-ATC-Assistant
```

### 2. Run the setup script

Double-click `setup.bat` or run it in a terminal:

```bat
setup.bat
```

This will:
- Install all Python dependencies
- Pull the `llama3.2:3b` model via Ollama

### 3. Manual setup (optional)

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
```

For the larger models:

```bash
ollama pull llama3.1:8b
ollama pull llama3.2-vision:11b
```

---

## Usage

```bash
python main.py
```

1. **Select an audio source** from the dropdown
   - `[Loopback]` entries capture audio playing through that output device (your game, Discord, etc.)
   - `[Mic]` entries use a microphone input
2. **Press ▶ Start Listening**
3. When a pilot speaks, their transmission appears as a gray bubble on the left
4. The suggested ATC response appears as a blue bubble on the right
5. Click **⊘ copy** under any ATC suggestion to copy it to your clipboard

### Mic toggle

Click **🎤 Mic** to quickly switch to your default microphone (useful for solo testing).

### Settings

Click **⚙️ Settings** in the top-right to change:

| Option | Description |
|--------|-------------|
| **All** | General controller — handles any request |
| **Departure** | Post-takeoff: climb instructions, heading assignments |
| **Ground** | Taxi routes, pushback approval, runway crossings |
| **Clearance** | Pre-departure clearances, initial altitudes, departure procedures |

### Model selector

| Model | Speed | Quality |
|-------|-------|---------|
| `llama3.2:3b` | ⚡ Fastest | Good |
| `llama3.1:8b` | Moderate | Better |
| `llama3.2-vision:11b` | Slowest | Best |

---

## Project Structure

```
PTFS-ATC-Assistant/
├── main.py          # Tkinter chat UI and app orchestration
├── audio.py         # Audio capture (WASAPI loopback + mic) with VAD
├── stt.py           # Speech-to-text via faster-whisper
├── llm.py           # ATC response generation via Ollama
├── requirements.txt # Python dependencies
├── setup.bat        # First-time setup script (Windows)
└── charts/          # Airport charts reference
```

---

## How It Works

```
Game/Discord audio
       │
       ▼
  WASAPI loopback       ← or microphone
       │
       ▼
  Energy-based VAD      ← detects speech segments
       │
       ▼
  faster-whisper        ← transcribes to text (offline, int8 CPU)
       │
       ▼
  Ollama (llama3.2:3b)  ← generates ATC suggestion (offline)
       │
       ▼
    Chat UI             ← displays pilot text + ATC suggestion
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

[MIT](https://choosealicense.com/licenses/mit/)
