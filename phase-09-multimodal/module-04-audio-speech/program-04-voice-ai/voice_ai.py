import subprocess
import requests
import whisper


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Jabra microphone
MICROPHONE = "hw:1,0"

# Audio files
INPUT_AUDIO = "user_audio.wav"
OUTPUT_AUDIO = "response.wav"

# Record for 5 seconds
RECORD_SECONDS = 5

# Whisper model
WHISPER_MODEL = "tiny"

# Ollama model
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# Step 1: Record user's voice
# ---------------------------------------------------------

print("🎤 Speak now...")

record_command = [
    "arecord",
    "-D", MICROPHONE,
    "-f", "S16_LE",
    "-c", "1",
    "-r", "16000",
    "-d", str(RECORD_SECONDS),
    INPUT_AUDIO
]

subprocess.run(record_command, check=True)

print("Recording completed.")


# ---------------------------------------------------------
# Step 2: Convert speech to text using Whisper
# ---------------------------------------------------------

print("\nLoading Whisper...")

whisper_model = whisper.load_model(WHISPER_MODEL)

print("Transcribing...")

result = whisper_model.transcribe(INPUT_AUDIO)

user_text = result["text"].strip()

print("\nYou said:")
print(user_text)


# ---------------------------------------------------------
# Step 3: Send text to Ollama
# ---------------------------------------------------------

print("\nSending text to Ollama...")

prompt = f"""
You are a helpful voice assistant.

The user said:
{user_text}

Give a short and natural response suitable for a voice conversation.
"""

response = requests.post(
    OLLAMA_URL,
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    },
    timeout=120
)

response.raise_for_status()

assistant_text = response.json()["response"].strip()

print("\nAssistant:")
print(assistant_text)


# ---------------------------------------------------------
# Step 4: Convert LLM response to speech using Piper
# ---------------------------------------------------------

print("\nGenerating voice response...")

piper_command = [
    "python",
    "-m",
    "piper",
    "--model",
    "en_US-lessac-medium",
    "--output_file",
    OUTPUT_AUDIO
]

subprocess.run(
    piper_command,
    input=assistant_text,
    text=True,
    check=True
)

print(f"Voice response saved to: {OUTPUT_AUDIO}")


# ---------------------------------------------------------
# Step 5: Play the response
# ---------------------------------------------------------

print("\n🔊 Playing response...")

subprocess.run(
    ["aplay", OUTPUT_AUDIO],
    check=True
)

print("\nVoice AI interaction completed.")