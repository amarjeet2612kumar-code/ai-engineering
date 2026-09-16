import subprocess


TEXT = "Hello Amar. Welcome to the world of multimodal AI."

VOICE_MODEL = "en_US-lessac-medium"


# Different values for speech speed
SPEEDS = {
    "fast": 0.8,
    "normal": 1.0,
    "slow": 1.2,
}


for name, length_scale in SPEEDS.items():

    output_file = f"speech_{name}.wav"

    print(f"Generating {name} speech...")

    command = [
        "python",
        "-m",
        "piper",
        "--model",
        VOICE_MODEL,
        "--length_scale",
        str(length_scale),
        "--output_file",
        output_file,
    ]

    subprocess.run(
        command,
        input=TEXT,
        text=True,
        check=True,
    )

    print(f"Created: {output_file}")