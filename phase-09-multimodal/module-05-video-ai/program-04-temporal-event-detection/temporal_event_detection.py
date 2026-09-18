# Import OpenCV.
# We use OpenCV to open the video, read frames,
# resize frames, and convert frames to grayscale.
import cv2

# Import NumPy.
# We use NumPy to calculate pixel differences.
import numpy as np

# Import Path.
# Path makes file and directory handling easier.
from pathlib import Path


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Path to the input video.
VIDEO_PATH = "data/sample_video.mp4"

# Directory where event frames will be saved.
OUTPUT_DIR = Path("output")

# Create the output directory if it doesn't exist.
OUTPUT_DIR.mkdir(exist_ok=True)

# Number of frames we want to examine per second.
#
# Example:
# 2 means we compare approximately two frames per second.
SAMPLES_PER_SECOND = 2

# Threshold used to decide whether the visual change
# is significant enough to report as a possible event.
#
# This is NOT a probability or percentage.
# It is the average grayscale pixel difference.
CHANGE_THRESHOLD = 25.0


# ---------------------------------------------------------
# 2. OPEN VIDEO
# ---------------------------------------------------------

# Open the video using OpenCV.
video = cv2.VideoCapture(VIDEO_PATH)

# Check whether the video opened successfully.
if not video.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ---------------------------------------------------------
# 3. READ VIDEO INFORMATION
# ---------------------------------------------------------

# Get the video's frames per second.
fps = video.get(cv2.CAP_PROP_FPS)

# Get the total number of frames.
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Calculate the approximate duration.
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
# 4. CALCULATE FRAME INTERVAL
# ---------------------------------------------------------

# We want to analyze SAMPLES_PER_SECOND frames
# every second.
#
# Example:
#
# FPS = 30
# SAMPLES_PER_SECOND = 2
#
# Then:
#
# 30 / 2 = 15
#
# So we examine approximately every 15th frame.
frame_interval = max(
    1,
    int(fps / SAMPLES_PER_SECOND)
)

print(
    f"Analyzing approximately "
    f"{SAMPLES_PER_SECOND} frames per second."
)

print(
    f"Frame interval: "
    f"{frame_interval}"
)


# ---------------------------------------------------------
# 5. VARIABLES FOR TEMPORAL ANALYSIS
# ---------------------------------------------------------

# Store the previous frame.
#
# We need this because temporal analysis compares
# one point in time with another point in time.
previous_gray = None

# Keep track of the current frame number.
frame_number = 0

# Keep track of detected visual events.
events = []


# ---------------------------------------------------------
# 6. READ VIDEO FRAME-BY-FRAME
# ---------------------------------------------------------

while True:

    # Read the next frame.
    success, frame = video.read()

    # If no frame is available, the video has ended.
    if not success:
        break

    # Increase the frame counter.
    frame_number += 1

    # -----------------------------------------------------
    # 7. SAMPLE FRAMES
    # -----------------------------------------------------

    # Only analyze frames at our selected interval.
    #
    # Example:
    #
    # frame 15  → analyze
    # frame 30  → analyze
    # frame 45  → analyze
    #
    # Other frames are simply skipped.
    if frame_number % frame_interval != 0:
        continue


    # -----------------------------------------------------
    # 8. PREPROCESS THE FRAME
    # -----------------------------------------------------

    # Resize the frame.
    #
    # We don't need the original high resolution
    # for this simple difference calculation.
    small_frame = cv2.resize(
        frame,
        (320, 240)
    )

    # Convert the frame from color to grayscale.
    #
    # This reduces the image from 3 color channels
    # to 1 intensity channel.
    gray_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # 9. FIRST FRAME
    # -----------------------------------------------------

    # If this is the first sampled frame,
    # we have nothing to compare it against.
    if previous_gray is None:

        # Store the current frame as the previous frame.
        previous_gray = gray_frame

        # Continue to the next sampled frame.
        continue


    # -----------------------------------------------------
    # 10. COMPARE PREVIOUS AND CURRENT FRAME
    # -----------------------------------------------------

    # Calculate the absolute difference between
    # the previous and current grayscale frames.
    #
    # Each pixel gets:
    #
    # |previous pixel - current pixel|
    difference = cv2.absdiff(
        previous_gray,
        gray_frame
    )

    # Calculate the average difference across
    # all pixels.
    #
    # This gives us one number representing
    # the amount of visual change.
    difference_score = float(
        np.mean(difference)
    )


    # -----------------------------------------------------
    # 11. CONVERT FRAME NUMBER TO TIME
    # -----------------------------------------------------

    # Calculate the timestamp of the current frame.
    #
    # Example:
    #
    # frame 150 / 30 FPS = 5 seconds
    timestamp = frame_number / fps


    # Print the temporal comparison.
    print(
        f"{timestamp:6.2f}s | "
        f"Frame {frame_number:4d} | "
        f"Change = {difference_score:6.2f}"
    )


    # -----------------------------------------------------
    # 12. DETECT A POSSIBLE EVENT
    # -----------------------------------------------------

    # Check whether the visual change is larger
    # than our threshold.
    if difference_score > CHANGE_THRESHOLD:

        print(
            "         -> POSSIBLE VISUAL EVENT"
        )

        # Store information about this event.
        events.append(
            {
                "frame": frame_number,
                "timestamp": timestamp,
                "change_score": difference_score,
            }
        )

        # Save the current frame so we can inspect
        # what caused the detected change.
        output_path = (
            OUTPUT_DIR
            / f"event_{frame_number:05d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )


    # -----------------------------------------------------
    # 13. UPDATE PREVIOUS FRAME
    # -----------------------------------------------------

    # The current frame becomes the previous frame
    # for the next comparison.
    previous_gray = gray_frame


# ---------------------------------------------------------
# 14. RELEASE VIDEO
# ---------------------------------------------------------

# Release the video resource.
video.release()


# ---------------------------------------------------------
# 15. DISPLAY EVENT TIMELINE
# ---------------------------------------------------------

print("\n------------------------------")
print("TEMPORAL EVENT TIMELINE")
print("------------------------------")

# Check whether we detected any events.
if len(events) == 0:

    print("No possible visual events detected.")

else:

    # Print every detected event.
    for event in events:

        print(
            f"{event['timestamp']:6.2f}s | "
            f"Frame {event['frame']:4d} | "
            f"Change = "
            f"{event['change_score']:.2f}"
        )


# ---------------------------------------------------------
# 16. FINAL RESULT
# ---------------------------------------------------------

print("\n------------------------------")
print("PROCESSING COMPLETE")
print("------------------------------")

print(
    f"Possible events detected: "
    f"{len(events)}"
)

print(
    f"Event frames saved in: "
    f"{OUTPUT_DIR}"
)

print("------------------------------")