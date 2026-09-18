# Import OpenCV.
# We use OpenCV to open the video and extract frames.
import cv2

# Import Base64.
# Images will be converted into Base64 strings so that
# they can be included in the API request.
import base64

# Import os.
# We use it to read the API key from the environment.
import os

# Import Path.
# Path makes file and directory handling easier.
from pathlib import Path

# Import dotenv.
# This allows us to load variables from our .env file.
from dotenv import load_dotenv

# Import the OpenAI client.
from openai import OpenAI


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Path to our input video.
VIDEO_PATH = "data/sample_video.mp4"

# Number of frames we want to sample from the video.
#
# We deliberately keep this small.
# Sending fewer frames reduces processing and API usage.
NUM_SAMPLES = 12

# Question that we want to ask about the video.
QUESTION = (
    "Describe what is happening in this video. "
    "Mention the main objects or people you can see "
    "and describe any obvious action or change over time."
)


# ---------------------------------------------------------
# 2. LOAD .ENV FILE
# ---------------------------------------------------------

# Find the project root.
#
# This file is located approximately at:
#
# phase-09-multimodal/
#     module-05-video-ai/
#         program-05-video-qa/
#             video_qa.py
#
# parents[3] takes us back to:
#
# phase-09-multimodal/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Build the path to the .env file.
ENV_PATH = PROJECT_ROOT / ".env"

# Load variables from the .env file.
load_dotenv(ENV_PATH)


# ---------------------------------------------------------
# 3. GET OPENAI API KEY
# ---------------------------------------------------------

# Read OPENAI_API_KEY from the environment.
api_key = os.getenv("OPENAI_API_KEY")

# Check whether the API key exists.
if not api_key:

    # Stop the program if the key wasn't found.
    raise RuntimeError(
        "OPENAI_API_KEY was not found in the .env file."
    )


# ---------------------------------------------------------
# 4. CREATE OPENAI CLIENT
# ---------------------------------------------------------

# Create the OpenAI API client.
#
# The client will use the API key loaded from .env.
client = OpenAI(api_key=api_key)


# ---------------------------------------------------------
# 5. OPEN THE VIDEO
# ---------------------------------------------------------

# Open the video using OpenCV.
video = cv2.VideoCapture(VIDEO_PATH)

# Check whether OpenCV successfully opened the video.
if not video.isOpened():

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ---------------------------------------------------------
# 6. READ VIDEO INFORMATION
# ---------------------------------------------------------

# Get the total number of frames.
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Get the video FPS.
fps = video.get(cv2.CAP_PROP_FPS)

# Calculate approximate duration.
duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print("\n------------------------------")
print("VIDEO INFORMATION")
print("------------------------------")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Duration     : {duration:.2f} seconds")
print("------------------------------")


# ---------------------------------------------------------
# 7. CALCULATE FRAME POSITIONS
# ---------------------------------------------------------

# Create evenly distributed frame positions.
#
# Example:
#
# 354 frames
# 12 samples
#
# We select approximately 12 frames spread
# across the entire video.
sample_positions = [
    int(i * (total_frames - 1) / (NUM_SAMPLES - 1))
    for i in range(NUM_SAMPLES)
]


print("\n------------------------------")
print("FRAME SAMPLING")
print("------------------------------")
print(f"Number of samples : {NUM_SAMPLES}")
print(f"Frame positions   : {sample_positions}")
print("------------------------------")


# ---------------------------------------------------------
# 8. EXTRACT SAMPLED FRAMES
# ---------------------------------------------------------

# This list will contain Base64 encoded images.
encoded_images = []

# Loop through each selected frame position.
for position in sample_positions:

    # Move the video reader to the selected frame.
    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        position
    )

    # Read the frame.
    success, frame = video.read()

    # Skip the frame if it could not be read.
    if not success:
        print(
            f"Could not read frame {position}"
        )
        continue

    # Resize the frame.
    #
    # We don't need a huge image for this demonstration.
    # Smaller images reduce the amount of data sent.
    frame = cv2.resize(
        frame,
        (640, 480)
    )

    # Encode the OpenCV frame as JPEG.
    #
    # cv2.imencode() converts the image from a NumPy array
    # into JPEG binary data.
    success, buffer = cv2.imencode(
        ".jpg",
        frame
    )

    # Check whether JPEG encoding worked.
    if not success:
        continue

    # Convert JPEG binary data into Base64 text.
    #
    # Base64 allows binary image data to be represented
    # as text inside the API request.
    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    # Add the encoded image to our list.
    encoded_images.append(
        image_base64
    )

    print(
        f"Prepared frame {position}"
    )


# ---------------------------------------------------------
# 9. RELEASE VIDEO
# ---------------------------------------------------------

# We are finished reading the video.
video.release()


# ---------------------------------------------------------
# 10. BUILD MULTIMODAL INPUT
# ---------------------------------------------------------

# Start the content list with our text question.
content = [
    {
        "type": "input_text",
        "text": QUESTION
    }
]


# Add every sampled frame as an image.
for image_base64 in encoded_images:

    content.append(
        {
            "type": "input_image",
            "image_url": (
                f"data:image/jpeg;base64,"
                f"{image_base64}"
            )
        }
    )


# ---------------------------------------------------------
# 11. SEND REQUEST TO MULTIMODAL MODEL
# ---------------------------------------------------------

print("\nSending video frames to the multimodal model...")

# Send the question and sampled images to the model.
response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "user",
            "content": content
        }
    ]
)


# ---------------------------------------------------------
# 12. GET THE ANSWER
# ---------------------------------------------------------

# Extract the generated text from the response.
answer = response.output_text


# ---------------------------------------------------------
# 13. DISPLAY ANSWER
# ---------------------------------------------------------

print("\n------------------------------")
print("VIDEO Q&A ANSWER")
print("------------------------------")
print(answer)
print("------------------------------")