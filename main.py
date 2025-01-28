import tensorflow as tf
import numpy as np
import cv2
import os

# Paths to the model files
MODEL_PATH = "ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29/saved_model"
LABELS_PATH = "coco_labels.txt"

# Load the labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().strip().split('\n')

# Load the TensorFlow model
print("Loading the TensorFlow model...")
model = tf.saved_model.load(MODEL_PATH)
detect_fn = model.signatures['serving_default']

def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    h, w, _ = frame.shape
    for i in range(int(detections['num_detections'])):
        confidence = detections['detection_scores'][0, i].numpy()
        if confidence > 0.6:  # Increase the confidence threshold
            class_id = int(detections['detection_classes'][0, i].numpy())
            label = labels[class_id]
            box = detections['detection_boxes'][0, i].numpy() * np.array([h, w, h, w])
            (y1, x1, y2, x2) = box.astype('int')

            label_text = f"{label}: {confidence:.2f}"
            print(label_text)  # Print the recognized object and confidence score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    choice = input("Do you want to use a webcam feed or provide a picture? (webcam/picture): ").strip().lower()

    if choice == 'webcam':
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = detect_objects(frame)
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif choice == 'picture':
        image_path = input("Enter the path to the image: ").strip()
        if not os.path.exists(image_path):
            print("Image file not found!")
            return

        frame = cv2.imread(image_path)
        frame = detect_objects(frame)
        cv2.imshow('Object Detection', frame)
        cv2.resizeWindow('Object Detection', 800, 600)  # Resize the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid choice. Please enter 'webcam' or 'picture'.")

if __name__ == "__main__":
    main()