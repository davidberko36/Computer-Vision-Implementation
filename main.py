# import cv2
# import tensorflow as tf
# import numpy as np

# # Paths to the model files
# MODEL_PATH = "ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
# CONFIG_PATH = "ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29/pipeline.config"
# LABELS_PATH = "coco_labels.txt"


# # load the labels
# with open(LABELS_PATH, 'r') as f:
#     labels = f.read().strip().split('\n')


# # load the model
# net = cv2.dnn.readNetFromTensorflow(MODEL_PATH)


# # Set up the video stream
# cap = cv2.VideoCapture(0)


# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break


#     # Prepare the input
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
#     net.setInput


#     # Perform object detection
#     detections = net.forward()
#     h, w = frame.shape[:2]


#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.5:
#             class_id = int(detections[0, 0, i, 1])
#             label = labels[class_id]
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x1, y1, x2, y2) = box.astype('int')


#             # Draw the bounding box and label
#             label = f"labels[class_id]: {confidence:.2f}"
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#     # Show the frame
#     cv2.imshow('Object Detection', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break


# # Cleanup
# cap.release()
# cv2.destroyAllWindows()


import tensorflow as tf
import numpy as np
import cv2

# Paths to the model files
MODEL_PATH = "ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29/saved_model"
LABELS_PATH = "coco_labels.txt"

# Load the labels
with open(LABELS_PATH, 'r') as f:
    labels = f.read().strip().split('\n')

# Load the TensorFlow model
print("Loading the TensorFlow model...")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures['serving_default']
print("Model loaded successfully!")

# Set up the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the input image
    input_tensor = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Perform object detection
    detections = infer(input_tensor)

    # Extract detection details
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(int)
    scores = detections['detection_scores'].numpy()[0]

    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i] * np.array([h, w, h, w])
            y1, x1, y2, x2 = box.astype(int)

            # Draw bounding box and label
            label = f"{labels[classes[i]]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
