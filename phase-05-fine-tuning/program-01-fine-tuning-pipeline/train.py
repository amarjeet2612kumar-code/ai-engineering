# Import json so that we can read our JSONL training file.
import json

# Import os so that we can create/check the output directory.
import os

# Import PyTorch for tensors, automatic differentiation, and training.
import torch

# Import Dataset and DataLoader for handling training examples in batches.
from torch.utils.data import Dataset, DataLoader

# Import the Hugging Face tokenizer and pretrained classification model.
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import AdamW, which will update the model parameters using gradients.
from torch.optim import AdamW


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Location of our training dataset.
DATA_FILE = "data/train.jsonl"

# Small pretrained BERT model suitable for CPU-based learning.
MODEL_NAME = "prajjwal1/bert-tiny"

# Number of classification categories.
NUM_LABELS = 4

# Number of examples processed together in one mini-batch.
BATCH_SIZE = 4

# Number of times we train over the complete dataset.
EPOCHS = 3

# Learning rate controls how large each parameter update is.
LEARNING_RATE = 2e-5

# Maximum number of tokens allowed for one input sentence.
MAX_LENGTH = 64

# Directory where the fine-tuned model will be saved.
OUTPUT_DIRECTORY = "output_model"


# ============================================================
# 2. LABEL DEFINITIONS
# ============================================================

# Map numeric labels to human-readable DataOps technologies.
LABEL_NAMES = {
    0: "Spark",
    1: "Kafka",
    2: "Hive",
    3: "Airflow",
}


# ============================================================
# 3. CUSTOM DATASET CLASS
# ============================================================

# Create a PyTorch Dataset that reads our JSONL file.
class DataOpsDataset(Dataset):

    # Initialize the dataset.
    def __init__(self, file_path, tokenizer):

        # Store all training examples in this list.
        self.examples = []

        # Store the tokenizer so that text can be converted into tokens.
        self.tokenizer = tokenizer

        # Open the JSONL training file.
        with open(file_path, "r", encoding="utf-8") as file:

            # Read the file one line at a time.
            for line in file:

                # Remove unnecessary whitespace from the line.
                line = line.strip()

                # Skip empty lines if they exist.
                if not line:
                    continue

                # Convert the JSON string into a Python dictionary.
                record = json.loads(line)

                # Store only the text and label that we need.
                self.examples.append(
                    {
                        "text": record["text"],
                        "label": record["label"],
                    }
                )

    # Return the total number of examples in the dataset.
    def __len__(self):

        # The length is simply the number of stored examples.
        return len(self.examples)

    # Return one training example when PyTorch asks for an index.
    def __getitem__(self, index):

        # Get the example at the requested index.
        example = self.examples[index]

        # Convert the text into token IDs and attention masks.
        encoded = self.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Remove the extra batch dimension created by return_tensors="pt".
        input_ids = encoded["input_ids"].squeeze(0)

        # Remove the extra batch dimension from the attention mask.
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Convert the numeric label into a PyTorch tensor.
        label = torch.tensor(
            example["label"],
            dtype=torch.long,
        )

        # Return the tensors required by the model.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

# Tell the user that tokenizer loading has started.
print("Loading tokenizer...")

# Load the tokenizer associated with our pretrained BERT model.
#
# use_fast=False tells Transformers to use the Python/slow tokenizer.
# This avoids the tokenizer backend problem we encountered earlier.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)


# ============================================================
# 5. LOAD PRETRAINED MODEL
# ============================================================

# Tell the user that the pretrained model is being loaded.
print("Loading pretrained model...")

# Load BERT with a classification head containing four output classes.
#
# The BERT body provides pretrained language representations.
# The classification head will learn our Spark/Kafka/Hive/Airflow task.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)


# ============================================================
# 6. CREATE DATASET
# ============================================================

# Create our custom PyTorch dataset.
dataset = DataOpsDataset(
    DATA_FILE,
    tokenizer,
)


# ============================================================
# 7. CREATE DATALOADER
# ============================================================

# DataLoader divides the dataset into mini-batches.
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# Print the number of training examples.
print(f"Number of training examples: {len(dataset)}")

# Print how many mini-batches exist in each epoch.
print(f"Number of batches per epoch: {len(dataloader)}")


# ============================================================
# 8. SELECT TRAINING DEVICE
# ============================================================

# Use GPU if PyTorch can detect one; otherwise use the CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Move the model to the selected device.
model.to(device)

# Display the device being used.
print(f"Training device: {device}")


# ============================================================
# 9. CREATE OPTIMIZER
# ============================================================

# AdamW updates the model parameters using the gradients.
#
# The optimizer receives all model parameters because this program
# demonstrates FULL FINE-TUNING.
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# 10. TRAINING LOOP
# ============================================================

# Put the model into training mode.
model.train()

# Loop through the complete dataset multiple times.
for epoch in range(EPOCHS):

    # Store the total loss for the current epoch.
    total_loss = 0.0

    # Display which epoch has started.
    print()
    print(f"Starting Epoch {epoch + 1}/{EPOCHS}")

    # Loop through every mini-batch in the DataLoader.
    for step, batch in enumerate(dataloader):

        # Move input token IDs to the selected device.
        input_ids = batch["input_ids"].to(device)

        # Move attention masks to the selected device.
        attention_mask = batch["attention_mask"].to(device)

        # Move labels to the selected device.
        labels = batch["labels"].to(device)

        # Remove gradients from the previous training step.
        #
        # This is necessary because PyTorch accumulates gradients
        # by default.
        optimizer.zero_grad()

        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        # Send the input batch through the model.
        #
        # The model produces logits and calculates the loss because
        # we provide the correct labels.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Extract the calculated classification loss.
        loss = outputs.loss

        # ----------------------------------------------------
        # BACKWARD PASS
        # ----------------------------------------------------

        # Calculate gradients for all trainable parameters.
        loss.backward()

        # ----------------------------------------------------
        # PARAMETER UPDATE
        # ----------------------------------------------------

        # Use the gradients to update the model parameters.
        optimizer.step()

        # Add the current batch loss to the epoch loss.
        total_loss += loss.item()

        # Display the loss for the current training step.
        print(
            f"Epoch {epoch + 1} | "
            f"Step {step + 1}/{len(dataloader)} | "
            f"Loss: {loss.item():.4f}"
        )

    # Calculate the average loss for the entire epoch.
    average_loss = total_loss / len(dataloader)

    # Display the average epoch loss.
    print(
        f"Average Epoch Loss: {average_loss:.4f}"
    )


# ============================================================
# 11. SAVE FINE-TUNED MODEL
# ============================================================

# Create the output directory if it does not already exist.
os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True,
)

# Save the fine-tuned model weights and configuration.
model.save_pretrained(
    OUTPUT_DIRECTORY,
)

# Save the tokenizer files alongside the model.
tokenizer.save_pretrained(
    OUTPUT_DIRECTORY,
)

# Display where the model was saved.
print()
print("Training completed successfully.")
print(f"Model saved to: {OUTPUT_DIRECTORY}")


# ============================================================
# 12. TEST THE FINE-TUNED MODEL
# ============================================================

# Put the model into evaluation mode.
model.eval()

# Create examples that the model has not seen during training.
test_texts = [
    "The Spark executor was killed due to memory usage.",
    "The Kafka consumer group has very high lag.",
    "Hive cannot find the required table partition.",
    "The Airflow task failed during DAG execution.",
]


# Display that testing has started.
print()
print("Testing the fine-tuned model...")
print()


# Disable gradient calculation because we are only doing inference.
#
# This saves memory and computation during testing.
with torch.no_grad():

    # Test every sentence one by one.
    for text in test_texts:

        # Convert the test sentence into model inputs.
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        # Move the input token IDs to the selected device.
        input_ids = encoded["input_ids"].to(device)

        # Move the attention mask to the selected device.
        attention_mask = encoded["attention_mask"].to(device)

        # Run the test input through the fine-tuned model.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get the raw classification scores.
        logits = outputs.logits

        # Select the class with the highest score.
        predicted_label = torch.argmax(
            logits,
            dim=1,
        ).item()

        # Convert the numeric class into a technology name.
        predicted_class = LABEL_NAMES[predicted_label]

        # Display the original input.
        print(f"Text: {text}")

        # Display the predicted technology.
        print(f"Prediction: {predicted_class}")

        # Print a blank line for readability.
        print()


# ============================================================
# 13. PROGRAM COMPLETION
# ============================================================

# Display a final completion message.
print("Program 1 completed.")