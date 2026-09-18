# Import OpenCV.
# OpenCV is used to read the video, extract frames,
# resize images, and calculate image differences.
import cv2

# Import NumPy.
# NumPy is used for numerical operations on image arrays.
import numpy as np

# Import Path.
# Path makes it easier to work with folders and file paths.
from pathlib import Path


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Path of the input video.
VIDEO_PATH = "data/sample_video.mp4"

# Directory where sampled frames will be saved.
OUTPUT_DIR = Path("output")

# Create the output directory if it does not exist.
OUTPUT_DIR.mkdir(exist_ok=True)

# Number of frames we want to sample from the entire video.
#
# Example:
# If the video contains 354 frames and NUM_SAMPLES = 10,
# we will select approximately 10 frames distributed
# across the complete video.
NUM_SAMPLES = 10

# Threshold used for basic scene-change detection.
#
# A higher value means we require a larger visual change
# before calling it a possible scene change.
SCENE_CHANGE_THRESHOLD = 25.0


# ---------------------------------------------------------
# 2. OPEN THE VIDEO
# ---------------------------------------------------------

# Open the video using OpenCV.
video = cv2.VideoCapture(VIDEO_PATH)

# Check whether the video was opened successfully.
if not video.isOpened():

    # Stop the program if OpenCV cannot open the video.
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# ---------------------------------------------------------
# 3. READ VIDEO INFORMATION
# ---------------------------------------------------------

# Get the total number of frames in the video.
total_frames = int(
    video.get(cv2.CAP_PROP_FRAME_COUNT)
)

# Get the video's FPS.
fps = video.get(cv2.CAP_PROP_FPS)

# Get the video width.
width = int(
    video.get(cv2.CAP_PROP_FRAME_WIDTH)
)

# Get the video height.
height = int(
    video.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

# Calculate approximate video duration.
duration = total_frames / fps if fps > 0 else 0


print("\n------------------------------")
print("VIDEO INFORMATION")
print("------------------------------")
print(f"Total Frames : {total_frames}")
print(f"FPS          : {fps:.2f}")
print(f"Resolution   : {width} x {height}")
print(f"Duration     : {duration:.2f} seconds")
print("------------------------------")


# ---------------------------------------------------------
# 4. CALCULATE SAMPLING POSITIONS
# ---------------------------------------------------------

# np.linspace() creates evenly distributed numbers
# between two values.
#
# We want NUM_SAMPLES positions spread across
# the entire video.
#
# Example:
#
# total_frames = 354
# NUM_SAMPLES = 10
#
# The result will contain approximately:
#
# 0, 39, 78, 118, ...
#
# These positions represent frames we want to inspect.
sample_positions = np.linspace(
    0,
    total_frames - 1,
    NUM_SAMPLES,
    dtype=int
)

# Remove duplicate positions if any occur.
#
# np.unique() ensures every selected frame number
# appears only once.
sample_positions = np.unique(sample_positions)


print("\n------------------------------")
print("FRAME SAMPLING")
print("------------------------------")
print(f"Requested Samples : {NUM_SAMPLES}")
print(f"Actual Samples    : {len(sample_positions)}")
print(f"Frame Positions   : {sample_positions}")
print("------------------------------")


# ---------------------------------------------------------
# 5. FUNCTION TO READ A SPECIFIC FRAME
# ---------------------------------------------------------

def read_frame(frame_number):
    """
    Read one specific frame from the video.

    frame_number:
        The position of the frame we want to read.

    Returns:
        The video frame as a NumPy array.
    """

    # Move the video reader to the requested frame.
    #
    # CAP_PROP_POS_FRAMES tells OpenCV which frame
    # position we want to access.
    video.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_number)
    )

    # Read the frame.
    success, frame = video.read()

    # If reading failed, return None.
    if not success:
        return None

    # Return the actual image/frame.
    return frame


# ---------------------------------------------------------
# 6. SAMPLE FRAMES
# ---------------------------------------------------------

# Store successfully read sampled frames.
sampled_frames = []

# Loop through all selected frame positions.
for position in sample_positions:

    # Read the selected frame.
    frame = read_frame(position)

    # Make sure the frame was successfully read.
    if frame is None:
        print(f"Could not read frame {position}")
        continue

    # Store the frame number and image together.
    sampled_frames.append(
        (position, frame)
    )

    # Create a filename for the sampled frame.
    output_path = (
        OUTPUT_DIR /
        f"sample_{position:05d}.jpg"
    )

    # Save the sampled frame as a JPEG image.
    cv2.imwrite(
        str(output_path),
        frame
    )

    # Print information about the saved frame.
    print(
        f"Saved sampled frame: {output_path}"
    )


# ---------------------------------------------------------
# 7. BASIC SCENE / FRAME CHANGE DETECTION
# ---------------------------------------------------------

print("\n------------------------------")
print("FRAME CHANGE ANALYSIS")
print("------------------------------")

# We need at least two frames to compare changes.
if len(sampled_frames) < 2:

    print("Not enough frames for comparison.")

else:

    # Compare each sampled frame with the previous
    # sampled frame.
    for i in range(1, len(sampled_frames)):

        # Get the previous frame number and image.
        previous_number, previous_frame = sampled_frames[i - 1]

        # Get the current frame number and image.
        current_number, current_frame = sampled_frames[i]

        # Resize both frames to a small common size.
        #
        # We don't need to compare full-resolution images.
        # Smaller images make the comparison faster.
        previous_small = cv2.resize(
            previous_frame,
            (320, 240)
        )

        current_small = cv2.resize(
            current_frame,
            (320, 240)
        )

        # Convert the previous frame to grayscale.
        #
        # Grayscale removes color information and leaves
        # brightness/intensity information.
        previous_gray = cv2.cvtColor(
            previous_small,
            cv2.COLOR_BGR2GRAY
        )

        # Convert the current frame to grayscale.
        current_gray = cv2.cvtColor(
            current_small,
            cv2.COLOR_BGR2GRAY
        )

        # Calculate the absolute pixel-by-pixel difference.
        #
        # If two images are almost identical,
        # their difference will be small.
        #
        # If they look very different,
        # their difference will be larger.
        difference = cv2.absdiff(
            previous_gray,
            current_gray
        )

        # Calculate the average difference across
        # all pixels.
        #
        # This gives us one number representing
        # how much the two frames changed.
        difference_score = np.mean(difference)

        # Print the comparison result.
        print(
            f"Frame {previous_number} → "
            f"Frame {current_number} | "
            f"Difference = {difference_score:.2f}"
        )

        # Check whether the difference is greater
        # than our scene-change threshold.
        if difference_score > SCENE_CHANGE_THRESHOLD:

            print(
                "  -> Possible scene change detected"
            )

        else:

            print(
                "  -> Similar visual content"
            )


# ---------------------------------------------------------
# 8. RELEASE VIDEO
# ---------------------------------------------------------

# Tell OpenCV that we are finished with the video.
video.release()


# ---------------------------------------------------------
# 9. FINAL RESULT
# ---------------------------------------------------------

print("\n------------------------------")
print("PROCESSING COMPLETE")
print("------------------------------")
print(
    f"Sampled frames saved: "
    f"{len(sampled_frames)}"
)
print(
    f"Output directory: "
    f"{OUTPUT_DIR}"
)
print("------------------------------")