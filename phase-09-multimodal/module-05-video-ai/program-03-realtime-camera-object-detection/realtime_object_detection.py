# Import OpenCV.
# We use OpenCV to access the laptop camera,
# read video frames, display frames, and handle
# the keyboard input.
import cv2

# Import the YOLO class from Ultralytics.
# This gives us access to pretrained YOLO models.
from ultralytics import YOLO


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

# Name of the pretrained YOLO model.
#
# "n" means nano, which is the smallest model
# in the YOLO26 detection family.
#
# We choose the smallest model because we are
# running inference on a CPU-based laptop.
MODEL_NAME = "yolo26n.pt"

# Camera index.
#
# 0 normally means the first/default camera.
CAMERA_INDEX = 0

# Minimum confidence required for a detection.
#
# For example:
#
# confidence = 0.80
# means the model is 80% confident about its prediction.
#
# Detections below this value will be filtered out.
CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------
# 2. LOAD THE PRETRAINED YOLO MODEL
# ---------------------------------------------------------

print("Loading YOLO model...")

# Load the pretrained YOLO26 nano detection model.
#
# If the model file is not already available locally,
# Ultralytics will download the pretrained weights
# the first time the model is used.
model = YOLO(MODEL_NAME)

print("YOLO model loaded.")


# ---------------------------------------------------------
# 3. OPEN THE LAPTOP CAMERA
# ---------------------------------------------------------

print("Opening camera...")

# Create a VideoCapture object.
#
# This connects OpenCV to the camera.
camera = cv2.VideoCapture(CAMERA_INDEX)

# Check whether the camera opened successfully.
if not camera.isOpened():

    # Stop the program if the camera cannot be opened.
    raise RuntimeError(
        "Could not open the camera."
    )

print("Camera opened successfully.")
print("Press 'q' to stop.")


# ---------------------------------------------------------
# 4. CONTINUOUS CAMERA LOOP
# ---------------------------------------------------------

# Keep running until the user presses 'q'.
while True:

    # Read one frame from the camera.
    #
    # success:
    #     True  → frame was captured successfully
    #     False → frame could not be captured
    #
    # frame:
    #     The actual camera image.
    success, frame = camera.read()

    # If the camera failed to provide a frame,
    # stop the loop.
    if not success:
        print("Could not read frame.")
        break


    # -----------------------------------------------------
    # 5. RUN OBJECT DETECTION
    # -----------------------------------------------------

    # Send the current camera frame to YOLO.
    #
    # conf controls the minimum detection confidence.
    #
    # YOLO analyzes the frame and returns detection results.
    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # -----------------------------------------------------
    # 6. DRAW DETECTIONS ON THE FRAME
    # -----------------------------------------------------

    # results is a list of YOLO Results objects.
    #
    # Because we supplied one frame,
    # results[0] contains the result for that frame.
    result = results[0]

    # Plot the detection results on the original frame.
    #
    # This draws:
    # - bounding boxes
    # - object names
    # - confidence scores
    annotated_frame = result.plot()


    # -----------------------------------------------------
    # 7. DISPLAY THE RESULT
    # -----------------------------------------------------

    # Display the annotated frame in a window.
    cv2.imshow(
        "Real-Time Object Detection",
        annotated_frame
    )


    # -----------------------------------------------------
    # 8. CHECK FOR EXIT
    # -----------------------------------------------------

    # Wait briefly for a keyboard event.
    #
    # & 0xFF makes the result consistent across platforms.
    #
    # ord("q") gives us the keyboard code for q.
    if cv2.waitKey(1) & 0xFF == ord("q"):

        # Exit the camera loop.
        break


# ---------------------------------------------------------
# 9. CLEAN UP
# ---------------------------------------------------------

# Release the camera.
camera.release()

# Close all OpenCV windows.
cv2.destroyAllWindows()

print("Camera released.")
print("Program stopped.")