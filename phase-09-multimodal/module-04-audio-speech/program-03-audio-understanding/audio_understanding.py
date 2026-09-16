from transformers import pipeline


# Audio classification model.
# This model is trained on AudioSet and can recognize
# different types of sounds.
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Audio file that we want to understand
AUDIO_FILE = "recording.wav"


print("Loading audio understanding model...")

classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
)

print("Model loaded.")
print("\nAnalyzing audio...")


# Ask the model for the top 5 predicted sound classes
results = classifier(
    AUDIO_FILE,
    top_k=5,
)


print("\n------------------------------")
print("AUDIO UNDERSTANDING")
print("------------------------------")

for result in results:
    print(
        f"{result['label']:30s} "
        f"{result['score']:.4f}"
    )

print("------------------------------")