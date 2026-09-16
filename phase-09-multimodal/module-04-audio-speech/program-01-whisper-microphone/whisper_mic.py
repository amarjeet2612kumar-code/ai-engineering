import subprocess
import whisper


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Jabra microphone identified earlier as hw:1,0
MICROPHONE = "hw:1,0"

# Audio file where the microphone recording will be saved
AUDIO_FILE = "recording.wav"

# Record for 5 seconds
RECORD_SECONDS = 5

# Use the smallest Whisper model because our laptop has
# limited CPU resources.
MODEL_NAME = "tiny"


# ---------------------------------------------------------
# Step 1: Record audio from the microphone
# ---------------------------------------------------------

print("Recording started...")
print("Please speak into your microphone.")

command = [
    "arecord",
    "-D", MICROPHONE,
    "-f", "S16_LE",
    "-c", "1",
    "-r", "16000",
    "-d", str(RECORD_SECONDS),
    AUDIO_FILE
]

# Run the Linux arecord command from Python
subprocess.run(command, check=True)

print(f"Recording saved to: {AUDIO_FILE}")


# ---------------------------------------------------------
# Step 2: Load Whisper
# ---------------------------------------------------------

print("\nLoading Whisper model...")

model = whisper.load_model(MODEL_NAME)

print("Whisper model loaded.")


# ---------------------------------------------------------
# Step 3: Transcribe the recorded audio
# ---------------------------------------------------------

print("\nTranscribing audio...")

result = model.transcribe(AUDIO_FILE)

transcript = result["text"]


# ---------------------------------------------------------
# Step 4: Display the transcription
# ---------------------------------------------------------

print("\n------------------------------")
print("TRANSCRIPTION")
print("------------------------------")
print(transcript)
print("------------------------------")