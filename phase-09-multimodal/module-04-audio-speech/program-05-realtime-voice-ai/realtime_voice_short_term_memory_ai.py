import json
import os
import re
import subprocess
import time
import wave

import requests
import whisper
import webrtcvad


# ============================================================
# CONFIGURATION
# ============================================================

# Jabra microphone detected on the system.
MICROPHONE = "hw:1,0"

# Audio sample rate.
SAMPLE_RATE = 16000

# 20 ms audio frame.
#
# 16000 samples/sec × 0.020 sec = 320 samples
# 16-bit audio = 2 bytes/sample
# 320 × 2 = 640 bytes
FRAME_SIZE = 640

# Small Whisper model for CPU-friendly execution.
WHISPER_MODEL = "tiny"

# Local Ollama model.
OLLAMA_MODEL = "llama3.2:3b"

# Ollama REST API endpoint.
OLLAMA_URL = "http://localhost:11434/api/generate"

# Piper voice.
PIPER_VOICE = "en_US-lessac-medium"

# Temporary user audio file.
AUDIO_FILE = "user_audio.wav"

# Generated assistant audio file.
RESPONSE_FILE = "response.wav"

# Persistent conversation memory.
MEMORY_FILE = "conversation_memory.json"


# ============================================================
# MEMORY CONFIGURATION
# ============================================================

# Maximum number of messages sent to Ollama.
#
# 20 messages means approximately 10 user/assistant turns.
#
# Keeping this limited helps reduce CPU processing time.
MAX_MEMORY_MESSAGES = 20


# ============================================================
# VAD CONFIGURATION
# ============================================================

# WebRTC VAD aggressiveness:
#
# 0 = least aggressive
# 1 = more aggressive
# 2 = aggressive
# 3 = most aggressive
#
# We use 3 to reduce background-noise detection.
VAD_MODE = 3

# Number of consecutive speech frames required
# before speech is considered started.
#
# 5 × 20 ms = 100 ms
MIN_SPEECH_FRAMES = 5

# Number of consecutive silent frames required
# before speech is considered finished.
#
# 40 × 20 ms = 800 ms
SILENCE_FRAMES = 40

# Keep the last 200 ms before speech detection.
#
# 10 × 20 ms = 200 ms
PRE_BUFFER_FRAMES = 10

# Ignore very short speech.
MIN_SPEECH_SECONDS = 0.5


# ============================================================
# CREATE VAD
# ============================================================

vad = webrtcvad.Vad(VAD_MODE)


# ============================================================
# LOAD WHISPER
# ============================================================

print()
print("=" * 60)
print("Loading Whisper model...")
print("=" * 60)

whisper_model = whisper.load_model(WHISPER_MODEL)

print("Whisper model loaded.")


# ============================================================
# LOAD MEMORY
# ============================================================

def load_memory():
    """
    Load previous conversation history from JSON.

    If the file does not exist, start with empty memory.
    """

    if not os.path.exists(MEMORY_FILE):

        print()
        print("No previous conversation memory found.")
        print("Starting with fresh memory.")

        return []

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            memory = json.load(file)

        # Make sure memory is a list.
        if not isinstance(memory, list):

            print()
            print("Memory file has an invalid format.")
            print("Starting with fresh memory.")

            return []

        print()
        print(
            f"Loaded {len(memory)} messages "
            f"from previous memory."
        )

        return memory

    except Exception as error:

        print()
        print(
            f"Could not load memory: {error}"
        )

        print("Starting with fresh memory.")

        return []


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(memory):
    """
    Save conversation history to JSON.

    This makes the conversation available
    after restarting the program.
    """

    try:

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                memory,
                file,
                indent=2,
                ensure_ascii=False
            )

    except Exception as error:

        print(
            f"Warning: could not save memory: {error}"
        )


# ============================================================
# GLOBAL CONVERSATION MEMORY
# ============================================================

# This list is loaded once when the program starts.
conversation_history = load_memory()


# ============================================================
# SAVE AUDIO AS WAV
# ============================================================

def save_audio(audio_data, filename):
    """
    Convert raw PCM audio into a WAV file.
    """

    with wave.open(filename, "wb") as wav_file:

        # Mono microphone.
        wav_file.setnchannels(1)

        # 16-bit audio = 2 bytes per sample.
        wav_file.setsampwidth(2)

        # 16 kHz sample rate.
        wav_file.setframerate(SAMPLE_RATE)

        # Write raw PCM audio.
        wav_file.writeframes(audio_data)


# ============================================================
# STOP COMMAND DETECTION
# ============================================================

def is_stop_command(text):
    """
    Detect whether the user wants to stop.

    Examples:

        stop
        exit
        quit
        goodbye
        stop today
        no, stop today
        please stop
        stop now
        stop the conversation
        I want to stop
    """

    # Convert to lowercase.
    text = text.lower().strip()

    # Remove punctuation.
    #
    # Example:
    #
    # "No, stop today."
    #
    # becomes:
    #
    # "no stop today"
    cleaned_text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Remove duplicate spaces.
    cleaned_text = re.sub(
        r"\s+",
        " ",
        cleaned_text
    ).strip()

    # --------------------------------------------------------
    # Exact commands
    # --------------------------------------------------------

    exact_commands = {
        "stop",
        "exit",
        "quit",
        "goodbye",
        "good bye",
        "end",
        "finish",
    }

    if cleaned_text in exact_commands:

        return True

    # --------------------------------------------------------
    # Natural stop phrases
    # --------------------------------------------------------

    stop_phrases = [

        "please stop",

        "stop today",

        "stop now",

        "stop the conversation",

        "stop conversation",

        "stop the program",

        "stop program",

        "exit the program",

        "exit program",

        "quit the program",

        "quit program",

        "end conversation",

        "end the conversation",

        "end the program",

        "i want to stop",

        "i want to end",

        "lets stop",

        "let us stop",

        "we can stop",

        "we should stop",

        "no stop",

        "no stop today",
    ]

    for phrase in stop_phrases:

        if phrase in cleaned_text:

            return True

    return False


# ============================================================
# RECORD ONE SPEECH TURN
# ============================================================

def record_speech_turn():
    """
    Listen continuously until the user finishes speaking.

    There is no fixed recording duration.

    VAD determines when speech starts and stops.
    """

    print()
    print("Listening...")
    print(
        "Speak naturally. "
        "Say 'stop' or 'exit' to finish."
    )

    # --------------------------------------------------------
    # Start arecord once for this speech turn.
    #
    # We don't repeatedly open/close the microphone.
    # --------------------------------------------------------

    command = [
        "arecord",

        "-D",
        MICROPHONE,

        "-f",
        "S16_LE",

        "-c",
        "1",

        "-r",
        str(SAMPLE_RATE),

        # Output raw PCM data.
        "-t",
        "raw",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # Frames captured before speech begins.
    pre_buffer = []

    # Frames belonging to this speech turn.
    speech_audio = []

    # Consecutive speech frames.
    speech_frames = 0

    # Consecutive silence frames.
    silence_frames = 0

    # Has the user started speaking?
    speech_started = False

    # Timestamp when speech starts.
    speech_start_time = None

    try:

        while True:

            # ------------------------------------------------
            # Read one 20 ms frame.
            # ------------------------------------------------

            frame = process.stdout.read(FRAME_SIZE)

            if len(frame) != FRAME_SIZE:

                continue

            # ------------------------------------------------
            # Ask VAD whether this frame contains speech.
            # ------------------------------------------------

            is_speech = vad.is_speech(
                frame,
                SAMPLE_RATE
            )

            # =================================================
            # BEFORE SPEECH STARTS
            # =================================================

            if not speech_started:

                # Keep recent frames.
                pre_buffer.append(frame)

                # Keep only the last 200 ms.
                if len(pre_buffer) > PRE_BUFFER_FRAMES:

                    pre_buffer.pop(0)

                # Count consecutive speech frames.
                if is_speech:

                    speech_frames += 1

                else:

                    # Background noise interrupted detection.
                    speech_frames = 0

                # ------------------------------------------------
                # Speech has officially started.
                # ------------------------------------------------

                if speech_frames >= MIN_SPEECH_FRAMES:

                    speech_started = True

                    speech_start_time = time.time()

                    # Add pre-buffer.
                    speech_audio.extend(
                        pre_buffer
                    )

                    silence_frames = 0

                    print(
                        "Speech detected..."
                    )

                continue

            # =================================================
            # AFTER SPEECH STARTS
            # =================================================

            # Store current audio frame.
            speech_audio.append(frame)

            if is_speech:

                # User is speaking.
                silence_frames = 0

            else:

                # User is silent.
                silence_frames += 1

            # =================================================
            # END OF SPEECH
            # =================================================

            if silence_frames >= SILENCE_FRAMES:

                speech_duration = (
                    time.time()
                    - speech_start_time
                )

                break

        return (
            b"".join(speech_audio),
            speech_duration
        )

    finally:

        # Stop microphone process.
        process.terminate()

        try:

            process.wait(timeout=1)

        except subprocess.TimeoutExpired:

            process.kill()


# ============================================================
# BUILD CONVERSATION CONTEXT
# ============================================================

def build_conversation_context():
    """
    Convert recent conversation memory into text
    for the Ollama prompt.
    """

    conversation_text = ""

    # Use only the latest messages.
    recent_messages = conversation_history[
        -MAX_MEMORY_MESSAGES:
    ]

    for message in recent_messages:

        role = message["role"]

        content = message["content"]

        if role == "user":

            conversation_text += (
                f"User: {content}\n"
            )

        elif role == "assistant":

            conversation_text += (
                f"Assistant: {content}\n"
            )

    return conversation_text


# ============================================================
# ASK OLLAMA
# ============================================================

def ask_llm(user_text):
    """
    Send the current user message and recent memory
    to the local Ollama model.
    """

    # Build previous conversation context.
    conversation_text = (
        build_conversation_context()
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful voice AI assistant.

Give concise and natural answers because your response
will be converted into speech.

Use previous conversation context when it is relevant.

Do not claim that you permanently learn from this
conversation.

The application stores previous messages and provides
them to you as context.

Previous conversation:
{conversation_text}

Current user message:
User: {user_text}

Answer naturally and directly.
"""

    # --------------------------------------------------------
    # Ollama request
    # --------------------------------------------------------

    payload = {

        "model": OLLAMA_MODEL,

        "prompt": prompt,

        # Return one complete response.
        "stream": False,

        "options": {

            # Limit response length.
            "num_predict": 150
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    # Raise an error if Ollama returned an HTTP error.
    response.raise_for_status()

    # Convert response JSON to Python dictionary.
    result = response.json()

    return result["response"].strip()


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_response(text):
    """
    Convert assistant text into speech using Piper
    and play the generated audio.
    """

    # --------------------------------------------------------
    # Piper command
    # --------------------------------------------------------

    piper_command = [

        "python",

        "-m",

        "piper",

        "-m",

        PIPER_VOICE,

        "-f",

        RESPONSE_FILE
    ]

    # Generate WAV file.
    subprocess.run(
        piper_command,
        input=text,
        text=True,
        check=True
    )

    # --------------------------------------------------------
    # Play WAV file.
    # --------------------------------------------------------

    subprocess.run(
        [
            "aplay",
            RESPONSE_FILE
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("REAL-TIME VOICE AI")
    print("=" * 60)

    print(
        f"Microphone : {MICROPHONE}"
    )

    print(
        f"Whisper    : {WHISPER_MODEL}"
    )

    print(
        f"Ollama     : {OLLAMA_MODEL}"
    )

    print(
        f"Memory     : {MEMORY_FILE}"
    )

    print(
        f"Max memory : {MAX_MEMORY_MESSAGES} messages"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # MEMORY STATUS
    # --------------------------------------------------------

    if conversation_history:

        print()
        print(
            f"Previous memory loaded: "
            f"{len(conversation_history)} messages."
        )

    else:

        print()
        print(
            "No previous conversation memory."
        )

        print(
            "Starting with a new conversation."
        )

    print()
    print("Start speaking.")

    print(
        "Say 'stop', 'exit', 'quit', "
        "or 'goodbye' to finish."
    )

    # ========================================================
    # CONTINUOUS CONVERSATION LOOP
    # ========================================================

    while True:

        try:

            # =================================================
            # STEP 1 — LISTEN
            # =================================================

            audio_data, speech_duration = (
                record_speech_turn()
            )

            # -------------------------------------------------
            # Ignore very short speech.
            # -------------------------------------------------

            if speech_duration < MIN_SPEECH_SECONDS:

                print(
                    "Very short audio detected. "
                    "Ignoring."
                )

                continue

            # -------------------------------------------------
            # Save user audio.
            # -------------------------------------------------

            save_audio(
                audio_data,
                AUDIO_FILE
            )

            # =================================================
            # STEP 2 — SPEECH TO TEXT
            # =================================================

            print()
            print("Transcribing...")

            whisper_start = time.time()

            result = whisper_model.transcribe(
                AUDIO_FILE,

                # CPU doesn't use FP16.
                fp16=False
            )

            transcript = result[
                "text"
            ].strip()

            whisper_time = (
                time.time()
                - whisper_start
            )

            # -------------------------------------------------
            # Ignore empty transcription.
            # -------------------------------------------------

            if not transcript:

                print(
                    "No speech recognized."
                )

                continue

            print()
            print("------------------------------")
            print("USER")
            print("------------------------------")
            print(transcript)

            # =================================================
            # STEP 3 — CHECK STOP COMMAND
            # =================================================

            # IMPORTANT:
            #
            # Stop detection happens BEFORE Ollama.
            #
            # Therefore:
            #
            # "No, stop today."
            #
            # will not be sent to the LLM.
            if is_stop_command(transcript):

                print()
                print(
                    "Stop command detected."
                )

                goodbye_message = (
                    "Goodbye. Ending the voice conversation."
                )

                print()
                print("------------------------------")
                print("ASSISTANT")
                print("------------------------------")
                print(goodbye_message)

                # Speak goodbye.
                speak_response(
                    goodbye_message
                )

                # Exit the loop.
                break

            # =================================================
            # STEP 4 — ADD USER MESSAGE
            # =================================================

            conversation_history.append(
                {
                    "role": "user",
                    "content": transcript
                }
            )

            # =================================================
            # STEP 5 — OLLAMA
            # =================================================

            print()
            print("Thinking...")

            llm_start = time.time()

            answer = ask_llm(
                transcript
            )

            llm_time = (
                time.time()
                - llm_start
            )

            print()
            print("------------------------------")
            print("ASSISTANT")
            print("------------------------------")
            print(answer)

            # =================================================
            # STEP 6 — ADD ASSISTANT MESSAGE
            # =================================================

            conversation_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # -------------------------------------------------
            # IMPORTANT FIX
            # -------------------------------------------------
            #
            # Do NOT do this:
            #
            # conversation_history = conversation_history[-20:]
            #
            # That creates a local-variable scoping problem
            # inside main().
            #
            # Instead, modify the existing list in-place.
            # -------------------------------------------------

            if len(conversation_history) > MAX_MEMORY_MESSAGES:

                del conversation_history[
                    :-MAX_MEMORY_MESSAGES
                ]

            # -------------------------------------------------
            # Save memory.
            # -------------------------------------------------

            save_memory(
                conversation_history
            )

            print()
            print(
                f"Memory stored: "
                f"{len(conversation_history)} messages."
            )

            # =================================================
            # STEP 7 — TEXT TO SPEECH
            # =================================================

            print()
            print("Speaking...")

            tts_start = time.time()

            speak_response(
                answer
            )

            tts_time = (
                time.time()
                - tts_start
            )

            # =================================================
            # STEP 8 — METRICS
            # =================================================

            print()
            print("------------------------------")
            print("TURN METRICS")
            print("------------------------------")

            print(
                f"Speech duration : "
                f"{speech_duration:.2f} sec"
            )

            print(
                f"Whisper time   : "
                f"{whisper_time:.2f} sec"
            )

            print(
                f"LLM time       : "
                f"{llm_time:.2f} sec"
            )

            print(
                f"TTS time       : "
                f"{tts_time:.2f} sec"
            )

            print("------------------------------")

        # =====================================================
        # CTRL+C
        # =====================================================

        except KeyboardInterrupt:

            print()
            print(
                "Ctrl+C detected."
            )

            break

        # =====================================================
        # OTHER ERRORS
        # =====================================================

        except Exception as error:

            print()
            print(
                "Error occurred:"
            )

            print(error)

            print()
            print(
                "The assistant will continue listening."
            )

    # ========================================================
    # PROGRAM END
    # ========================================================

    print()
    print("=" * 60)
    print("VOICE AI SESSION ENDED")
    print("=" * 60)

    print(
        f"Messages in memory: "
        f"{len(conversation_history)}"
    )

    print(
        f"Memory file: "
        f"{MEMORY_FILE}"
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()