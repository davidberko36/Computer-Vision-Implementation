# Computer Vision Implementation

This repository contains a Python script for performing object detection using a pre-trained TensorFlow model. The script can either use a webcam feed or an image provided by the user to detect objects and display the results.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/davidberko36/computer-vision-implementation.git
   cd computer-vision-implementation

2. Install the required packages:
```bash
pip install tensorflow opencv-python numpy
```

### Usage
Run the script:

```bash
python main.py
```

2. Choose whether to use a webcam feed or provide an image:

 - For webcam feed, type webcam and press Enter.
 - For an image, type picture and press Enter. Then, provide the path to the image file.


## Script Details
main.py: The main script that loads the TensorFlow model, captures the webcam feed or reads an image, performs object detection, and displays the results.
coco_labels.txt: A text file containing the labels for the COCO dataset.
main.py
The script performs the following steps:

1. Loads the TensorFlow model from the specified path.
2. Prompts the user to choose between using a webcam feed or providing an image.
3. If the user chooses the webcam feed:
 - Captures frames from the webcam.
 - Performs object detection on each frame.
 - Displays the frames with detected objects highlighted.
4. If the user chooses to provide an image:
 - Reads the image from the specified path.
 - Performs object detection on the image.
 - Displays the image with detected objects highlighted.
 - Prints the recognized objects and their confidence scores in the terminal.

### coco_labels.txt
This file contains the labels for the COCO dataset, which are used to identify the detected objects.


### Example
When running the script, you will see a window displaying the webcam feed or the provided image with bounding boxes around detected objects. The recognized objects and their confidence scores will be printed in the terminal.