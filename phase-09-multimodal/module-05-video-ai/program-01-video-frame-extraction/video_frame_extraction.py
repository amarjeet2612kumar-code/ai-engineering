# Import OpenCV.
# OpenCV is a computer vision library that allows us to work with
# images, videos, cameras, and video frames.
import cv2

# Import Path from pathlib.
# Path makes it easier and safer to work with file and directory paths.
from pathlib import Path


# ---------------------------------------------------------
# 1. DEFINE INPUT AND OUTPUT PATHS
# ---------------------------------------------------------

# Path of the input video.
# Our video is stored inside the "data" directory.
VIDEO_PATH = "data/sample_video.mp4"

# Define the directory where extracted frames will be saved.
OUTPUT_DIR = Path("output")

# Create the "output" directory if it does not already exist.
#
# exist_ok=True means:
# - If the directory does not exist → create it.
# - If it already exists → don't throw an error.
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# 2. OPEN THE VIDEO
# ---------------------------------------------------------

# Open the video using OpenCV.
#
# cv2.VideoCapture() creates a VideoCapture object.
# This object gives us access to the video's frames and metadata.
#
# The video is NOT loaded completely into RAM here.
# OpenCV opens the video so that we can read it frame-by-frame.
video = cv2.VideoCapture(VIDEO_PATH)

# Check whether OpenCV successfully opened the video.
#
# video.isOpened() returns:
# True  → video was opened successfully.
# False → video could not be opened.
if not video.isOpened():

    # Stop the program and show an error message.
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# ---------------------------------------------------------
# 3. READ VIDEO METADATA
# ---------------------------------------------------------

# Get the video's Frames Per Second (FPS).
#
# FPS tells us how many frames are displayed in one second.
#
# Example:
# FPS = 30
# means the video contains approximately 30 frames per second.
fps = video.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video.
#
# CAP_PROP_FRAME_COUNT asks OpenCV for the total frame count.
#
# int() converts the returned value into an integer.
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Get the width of the video in pixels.
#
# Example:
# width = 1280
# means every frame is 1280 pixels wide.
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

# Get the height of the video in pixels.
#
# Example:
# height = 720
# means every frame is 720 pixels high.
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the approximate duration of the video.
#
# Formula:
#
#     duration = total frames / FPS
#
# Example:
#
#     300 frames / 30 FPS = 10 seconds
#
# We check fps > 0 to avoid division by zero.
duration = total_frames / fps if fps > 0 else 0


# ---------------------------------------------------------
# 4. DISPLAY VIDEO INFORMATION
# ---------------------------------------------------------

# Print a separator to make the terminal output easier to read.
print("\n------------------------------")

# Print the section title.
print("VIDEO INFORMATION")

# Print another separator.
print("------------------------------")

# Print FPS.
#
# :.2f means display the number with 2 digits after the decimal.
#
# Example:
# 30 → 30.00
print(f"FPS           : {fps:.2f}")

# Print total number of frames.
print(f"Total Frames  : {total_frames}")

# Print video resolution.
#
# Example:
# 1280 x 720
print(f"Resolution    : {width} x {height}")

# Print video duration.
#
# :.2f means display two digits after the decimal.
print(f"Duration      : {duration:.2f} seconds")

# Print the closing separator.
print("------------------------------")


# ---------------------------------------------------------
# 5. READ AND SAVE VIDEO FRAMES
# ---------------------------------------------------------

# Keep track of which frame we are currently reading.
#
# We start at 0 because we haven't read any frames yet.
frame_number = 0

# Keep track of how many frames we actually save.
#
# This is different from frame_number because we won't save
# every frame.
saved_frames = 0

# Define how frequently we want to save a frame.
#
# FRAME_INTERVAL = 30 means:
#
# Read:
# Frame 1
# Frame 2
# ...
# Frame 30 → save
# Frame 31
# ...
# Frame 60 → save
#
# If the video is 30 FPS, this is approximately
# one saved frame per second.
FRAME_INTERVAL = 30


# ---------------------------------------------------------
# 6. LOOP THROUGH THE VIDEO
# ---------------------------------------------------------

# Start an infinite loop.
#
# We will keep reading frames until the video ends.
while True:

    # Read the next frame from the video.
    #
    # video.read() returns two values:
    #
    # success → tells us whether a frame was successfully read.
    # frame   → contains the actual image/frame.
    success, frame = video.read()

    # Check whether reading the frame failed.
    #
    # When the video reaches the end,
    # video.read() returns success = False.
    if not success:

        # Stop the while loop because there are no more frames.
        break

    # Increase the frame counter by 1.
    #
    # First frame:
    # 0 + 1 = 1
    #
    # Second frame:
    # 1 + 1 = 2
    #
    # And so on.
    frame_number += 1

    # Check whether the current frame is a frame
    # that we want to save.
    #
    # % is the modulo operator.
    #
    # Example:
    #
    # 30 % 30 = 0  → save
    # 60 % 30 = 0  → save
    # 90 % 30 = 0  → save
    #
    # But:
    #
    # 31 % 30 = 1  → don't save
    if frame_number % FRAME_INTERVAL == 0:

        # Create the output filename.
        #
        # Example:
        # frame_number = 30
        #
        # f"frame_{frame_number:05d}.jpg"
        #
        # becomes:
        #
        # frame_00030.jpg
        #
        # :05d means:
        # - d = integer
        # - 5 = use 5 digits
        # - 0 = fill missing digits with zero
        output_path = OUTPUT_DIR / f"frame_{frame_number:05d}.jpg"

        # Save the current frame as a JPEG image.
        #
        # cv2.imwrite() takes:
        # 1. The file path
        # 2. The image/frame
        #
        # The frame is stored as an image on disk.
        cv2.imwrite(str(output_path), frame)

        # Increase the number of saved frames by 1.
        saved_frames += 1

        # Tell us which frame was saved.
        print(f"Saved: {output_path}")


# ---------------------------------------------------------
# 7. CLOSE THE VIDEO
# ---------------------------------------------------------

# Release the VideoCapture object.
#
# This tells OpenCV that we are finished using the video.
#
# It is good practice to release resources after processing.
video.release()


# ---------------------------------------------------------
# 8. DISPLAY FINAL RESULT
# ---------------------------------------------------------

# Print a separator.
print("\n------------------------------")

# Print completion message.
print("EXTRACTION COMPLETE")

# Print the total number of frames that were read.
print(f"Frames read   : {frame_number}")

# Print how many frames we actually saved.
print(f"Frames saved  : {saved_frames}")

# Print where the extracted frames were stored.
print(f"Output folder : {OUTPUT_DIR}")

# Print final separator.
print("------------------------------")