import cv2
import numpy as np

# Load the pre-trained MobileNet SSD model
model_weights = "mobilenet_iter_73000.caffemodel"  # Path to the model weights
model_config = "deploy.prototxt"  # Path to the model configuration
net = cv2.dnn.readNetFromCaffe(model_config, model_weights)

# Load the COCO class labels
class_labels = []
with open("coco.names", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get the frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:  # Confidence threshold
            class_id = int(detections[0, 0, i, 1])

            # Check if the detected object is a person (class_id = 15 in COCO dataset)
            if class_id == 15:  # 15 corresponds to "person" in COCO
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Person: {confidence:.2f}%"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate the center of the detected person
                person_center_x = (startX + endX) // 2
                person_center_y = (startY + endY) // 2

                # Draw the center of the detected person
                cv2.circle(frame, (person_center_x, person_center_y), 5, (255, 0, 0), -1)

                # Calculate the center of the frame
                frame_center_x = w // 2
                frame_center_y = h // 2

                # Draw the center of the frame
                cv2.circle(frame, (frame_center_x, frame_center_y), 5, (0, 0, 255), -1)

                # Calculate the difference between the person's center and the frame's center
                diff_x = person_center_x - frame_center_x
                diff_y = person_center_y - frame_center_y

                # Display the difference
                cv2.putText(frame, f"Diff X: {diff_x}, Diff Y: {diff_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Send commands to the robot based on the difference
                if abs(diff_x) > 50:  # Athlete is not centered horizontally
                    if diff_x > 0:
                        print("Faster")
                    else:
                        print("Slow Down")
                else:
                    print("Stay centered")

    # Display the resulting frame
    cv2.imshow("Athlete Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()