import cv2

# Open the default camera.
# Camera index 0 usually means the first available camera.
camera = cv2.VideoCapture(0)

# Check whether the camera opened successfully.
if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

print("Camera opened successfully.")
print("Press 'q' to stop.")

while True:

    # Read one frame from the camera.
    success, frame = camera.read()

    # Stop if a frame could not be read.
    if not success:
        print("Could not read frame.")
        break

    # Display the live camera frame.
    cv2.imshow("Camera Test", frame)

    # Check whether the user pressed 'q'.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera.
camera.release()

# Close the OpenCV window.
cv2.destroyAllWindows()