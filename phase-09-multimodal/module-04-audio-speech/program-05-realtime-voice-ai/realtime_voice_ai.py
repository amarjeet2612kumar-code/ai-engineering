# =========================================================
# IMPORTS
# =========================================================

# subprocess:
# Used to run Linux commands/programs from Python.
#
# We use it to:
# 1. Start "arecord" for microphone audio
# 2. Start "piper" for text-to-speech
# 3. Start "aplay" to play the generated audio
import subprocess


# time:
# Used to measure how much time each stage takes.
#
# Example:
# STT took 1.5 seconds
# LLM took 8 seconds
# TTS took 3 seconds
import time


# wave:
# Python's built-in module for reading/writing WAV audio files.
#
# We use it to save the raw microphone audio
# collected from the continuous audio stream.
import wave


# requests:
# Used to send HTTP requests.
#
# We use it to call the local Ollama API:
# Python → HTTP request → Ollama → LLM response
import requests


# whisper:
# OpenAI Whisper library.
#
# We use Whisper for:
# Speech-to-Text (STT)
#
# Audio → Whisper → Text
import whisper


# webrtcvad:
# Voice Activity Detection library.
#
# It determines whether a small audio frame contains:
# - human speech
# - silence / non-speech
#
# We use it to detect when the user starts and stops speaking.
import webrtcvad


# =========================================================
# CONFIGURATION
# =========================================================

# Your Jabra microphone.
MICROPHONE = "hw:1,0"


# File where the user's complete speech turn is saved.
AUDIO_FILE = "user_audio.wav"


# File containing the assistant's generated speech.
RESPONSE_FILE = "response.wav"


# Audio sample rate.
#
# 16000 Hz means:
# 16,000 audio samples are captured every second.
#
# This is commonly used for speech processing.
SAMPLE_RATE = 16000


# ---------------------------------------------------------
# Audio frame size
# ---------------------------------------------------------
#
# VAD doesn't process an entire recording at once.
# It checks small audio frames.
#
# We use 20 ms frames.
#
# Calculation:
#
# 16,000 samples/sec × 0.020 sec = 320 samples
#
# 16-bit audio = 2 bytes/sample
#
# 320 × 2 = 640 bytes
#
FRAME_SIZE = 640


# Whisper model.
#
# "tiny" is used because your laptop has limited CPU resources.
WHISPER_MODEL = "tiny"


# Local Ollama model.
OLLAMA_MODEL = "llama3.2:3b"


# Ollama local REST API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# VAD sensitivity
# ---------------------------------------------------------
#
# 0 = least aggressive
# 1 = slightly aggressive
# 2 = more aggressive
# 3 = most aggressive
#
VAD_MODE = 2


# ---------------------------------------------------------
# Silence threshold
# ---------------------------------------------------------
#
# VAD checks audio in 20 ms frames.
#
# 25 frames × 20 ms = 500 ms
#
# So we consider approximately 0.5 seconds of continuous
# silence as the end of the user's speech turn.
#
SILENCE_FRAMES = 25


# ---------------------------------------------------------
# Pre-buffer
# ---------------------------------------------------------
#
# Sometimes VAD detects speech a little late.
#
# We therefore keep a small amount of audio before
# speech detection.
#
# 10 frames × 20 ms = 200 ms
#
PRE_BUFFER_FRAMES = 10


# ---------------------------------------------------------
# Minimum speech duration
# ---------------------------------------------------------
#
# Very short microphone noise can sometimes be detected
# as speech.
#
# We require at least 0.5 seconds of speech before
# sending the audio to Whisper.
#
MIN_SPEECH_SECONDS = 0.5


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading Whisper...")

# Load the Whisper model into memory.
whisper_model = whisper.load_model(WHISPER_MODEL)

print("Whisper loaded.")


# Create the Voice Activity Detector.
vad = webrtcvad.Vad(VAD_MODE)

print("VAD loaded.")


# =========================================================
# FUNCTION: PROCESS ONE VOICE TURN
# =========================================================

def process_voice_turn(speech_audio, speech_duration):
    """
    Process one complete user speech turn.

    Flow:

    Audio
      ↓
    WAV file
      ↓
    Whisper
      ↓
    Text
      ↓
    Ollama
      ↓
    Text response
      ↓
    Piper
      ↓
    Audio response
    """

    # -----------------------------------------------------
    # Save the collected audio as WAV
    # -----------------------------------------------------

    print("\nSaving captured speech...")

    with wave.open(AUDIO_FILE, "wb") as wav_file:

        # Mono audio.
        wav_file.setnchannels(1)

        # 16-bit audio = 2 bytes/sample.
        wav_file.setsampwidth(2)

        # 16 kHz sample rate.
        wav_file.setframerate(SAMPLE_RATE)

        # Write all collected audio frames.
        wav_file.writeframes(speech_audio)

    print(f"Audio saved to: {AUDIO_FILE}")


    # =====================================================
    # SPEECH-TO-TEXT
    # =====================================================

    print("\nTranscribing...")

    stt_start = time.time()

    # Whisper converts the recorded speech into text.
    result = whisper_model.transcribe(AUDIO_FILE)

    # Extract the actual transcript.
    user_text = result["text"].strip()

    stt_time = time.time() - stt_start


    print("\n--------------------------------")
    print("USER")
    print("--------------------------------")
    print(user_text)

    print(f"\nSpeech duration : {speech_duration:.2f} sec")
    print(f"STT latency     : {stt_time:.2f} sec")


    # -----------------------------------------------------
    # If Whisper could not understand anything,
    # don't call the LLM.
    # -----------------------------------------------------

    if not user_text:

        print("\nNo speech was recognized.")

        return True


    # =====================================================
    # STOP COMMAND
    # =====================================================
    #
    # The user can end the conversation by saying:
    #
    # stop
    # exit
    # quit
    # goodbye
    #

    stop_commands = {
        "stop",
        "exit",
        "quit",
        "goodbye",
        "good bye"
    }


    # Convert text to lowercase for easy comparison.
    normalized_text = user_text.lower().strip()


    if normalized_text in stop_commands:

        print("\nStopping Voice AI...")

        return False


    # =====================================================
    # OLLAMA / LLM
    # =====================================================

    print("\nThinking...")

    llm_start = time.time()


    # Prompt for the local LLM.
    #
    # Because the response will be spoken aloud,
    # we don't want a very long response.
    prompt = f"""
You are a voice assistant.

User said:
{user_text}

Give a concise spoken response.

For simple questions, answer in 1–2 sentences.

If the question needs explanation, provide enough
information to answer clearly without unnecessary detail.

Do not use bullet points or markdown because your
response will be converted to speech.
"""


    # Send the prompt to Ollama.
    response = requests.post(
        OLLAMA_URL,

        json={
            "model": OLLAMA_MODEL,

            "prompt": prompt,

            # Wait for the complete LLM response.
            "stream": False,

            "options": {
                # Maximum number of tokens generated.
                "num_predict": 150
            }
        },

        # Prevent Python from waiting forever.
        timeout=120
    )


    # Raise an error if Ollama returned an HTTP error.
    response.raise_for_status()


    # Extract the generated response.
    assistant_text = response.json()["response"].strip()

    llm_time = time.time() - llm_start


    print("\n--------------------------------")
    print("ASSISTANT")
    print("--------------------------------")
    print(assistant_text)

    print(f"\nLLM latency     : {llm_time:.2f} sec")


    # =====================================================
    # TEXT-TO-SPEECH
    # =====================================================

    print("\nGenerating voice response...")

    tts_start = time.time()


    # Piper converts:
    #
    # Text → Speech audio
    #
    piper_command = [
        "python",
        "-m",
        "piper",

        # Piper voice model.
        "--model",
        "en_US-lessac-medium",

        # Output WAV file.
        "--output_file",
        RESPONSE_FILE
    ]


    # Send the LLM response to Piper.
    subprocess.run(
        piper_command,

        # Text sent to Piper through standard input.
        input=assistant_text,

        text=True,

        check=True
    )


    tts_time = time.time() - tts_start


    print(f"TTS latency     : {tts_time:.2f} sec")


    # =====================================================
    # PLAY AUDIO
    # =====================================================

    print("\n🔊 Assistant speaking...")

    # aplay is Linux's command-line audio player.
    subprocess.run(
        ["aplay", RESPONSE_FILE],
        check=True
    )


    # =====================================================
    # METRICS
    # =====================================================

    processing_time = (
        stt_time +
        llm_time +
        tts_time
    )


    print("\n--------------------------------")
    print("VOICE AI METRICS")
    print("--------------------------------")

    print(
        f"Speech duration : "
        f"{speech_duration:.2f} sec"
    )

    print(
        f"STT             : "
        f"{stt_time:.2f} sec"
    )

    print(
        f"LLM             : "
        f"{llm_time:.2f} sec"
    )

    print(
        f"TTS             : "
        f"{tts_time:.2f} sec"
    )

    print(
        f"Processing      : "
        f"{processing_time:.2f} sec"
    )

    print("--------------------------------")


    # True means:
    # Continue listening for another user turn.
    return True


# =========================================================
# MAIN CONVERSATION LOOP
# =========================================================

while True:

    print("\n================================")
    print("🎤 WAITING FOR SPEECH")
    print("================================")
    print("Speak now...")
    print("Say 'stop' to exit.")


    # -----------------------------------------------------
    # Start microphone.
    #
    # IMPORTANT:
    # We open the microphone once for each conversation turn.
    # We don't use a fixed recording duration.
    # -----------------------------------------------------

    record_command = [
        "arecord",
        "-D", MICROPHONE,
        "-f", "S16_LE",
        "-c", "1",
        "-r", str(SAMPLE_RATE),
        "-t", "raw",
    ]


    process = subprocess.Popen(
        record_command,

        # Read raw audio from stdout.
        stdout=subprocess.PIPE,

        # Hide arecord messages.
        stderr=subprocess.DEVNULL
    )


    # -----------------------------------------------------
    # Buffers
    # -----------------------------------------------------

    speech_audio = bytearray()

    pre_buffer = []

    silent_frames = 0

    speech_started = False

    speech_start_time = None


    try:

        while True:

            # ---------------------------------------------
            # Read one 20 ms audio frame.
            # ---------------------------------------------

            frame = process.stdout.read(FRAME_SIZE)


            if len(frame) != FRAME_SIZE:

                continue


            # ---------------------------------------------
            # Ask VAD whether this frame contains speech.
            # ---------------------------------------------

            is_speech = vad.is_speech(
                frame,
                SAMPLE_RATE
            )


            # =============================================
            # SPEECH DETECTED
            # =============================================

            if is_speech:

                # -----------------------------------------
                # First speech frame.
                # -----------------------------------------

                if not speech_started:

                    speech_started = True

                    speech_start_time = time.time()

                    print("\n🎤 Speech detected!")


                    # Add audio from the pre-buffer.
                    for buffered_frame in pre_buffer:

                        speech_audio.extend(
                            buffered_frame
                        )

                    pre_buffer.clear()


                # Add current frame to speech buffer.
                speech_audio.extend(frame)

                # Reset silence counter.
                silent_frames = 0


            # =============================================
            # SILENCE DETECTED
            # =============================================

            else:

                # -----------------------------------------
                # Speech hasn't started yet.
                #
                # Maintain the rolling pre-buffer.
                # -----------------------------------------

                if not speech_started:

                    pre_buffer.append(frame)


                    # Keep only the latest few frames.
                    if len(pre_buffer) > PRE_BUFFER_FRAMES:

                        pre_buffer.pop(0)


                # -----------------------------------------
                # Speech already started.
                # -----------------------------------------

                else:

                    # Keep the silence frame.
                    #
                    # This prevents cutting off the final
                    # word immediately.
                    speech_audio.extend(frame)

                    silent_frames += 1


            # =============================================
            # END OF SPEECH
            # =============================================

            if speech_started:

                if silent_frames >= SILENCE_FRAMES:

                    print("\n🛑 Speech ended.")

                    speech_duration = (
                        time.time()
                        - speech_start_time
                    )

                    break


    except KeyboardInterrupt:

        print("\nKeyboard interrupt received.")

        process.terminate()
        process.wait()

        break


    finally:

        # Stop the microphone process.
        process.terminate()

        process.wait()


    # =====================================================
    # IGNORE VERY SHORT AUDIO
    # =====================================================

    if speech_duration < MIN_SPEECH_SECONDS:

        print(
            f"\nSpeech was too short "
            f"({speech_duration:.2f} sec)."
        )

        print("Ignoring it.")

        continue


    # =====================================================
    # PROCESS USER SPEECH
    # =====================================================

    should_continue = process_voice_turn(
        speech_audio,
        speech_duration
    )


    # -----------------------------------------------------
    # False means the user said:
    # stop / exit / quit / goodbye
    # -----------------------------------------------------

    if not should_continue:

        break


# =========================================================
# PROGRAM END
# =========================================================

print("\n================================")
print("Voice AI stopped.")
print("================================")