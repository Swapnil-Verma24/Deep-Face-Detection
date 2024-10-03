# Face Detection Model with IoU Evaluation

## Overview

This project implements a face detection model using TensorFlow and Keras. The model employs transfer learning with VGG16 as the base architecture to classify and localize faces in images. The project also includes functionality to evaluate the model's performance using the Intersection over Union (IoU) metric.

## Key Features

- **Data Collection**: Captures images using a webcam and stores them for training and testing.
- **Data Augmentation**: Applies various augmentation techniques to increase the robustness of the model.
- **Model Training**: Trains a custom face detection model using augmented datasets.
- **IoU Evaluation**: Calculates the Intersection over Union (IoU) score for predicted bounding boxes against ground truth labels.

## Prerequisites

- Python 3.6 or later
- TensorFlow 2.x
- OpenCV
- Albumentations
- Matplotlib
- Other required libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Swapnil-Verma24/Face_detection_model.git
   cd Face_detection_model
2. Install required packages: It’s recommended to create a virtual environment first:
    ```bash
    # Create a virtual environment
    python -m venv env
    
    # Activate the virtual environment
    # On Windows
    env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    
    # Install the required packages
    pip install -r requirements.txt
    ```

## Model File
The trained model is saved in `facetracker.h5`, which can be loaded to make predictions.

## Usage
1. Collect images for training: The script captures images from your webcam. Make sure to allow camera access.
   
2. Run the face detection model: Open and run the [Face_detection_model.ipynb](Face_detection_model.ipynb) notebook in Jupyter Notebook or Google Colab.

## License
This project is licensed under the MIT LICENSE - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## Acknowledgments
[TensorFlow](https://www.tensorflow.org/)

[Keras](https://keras.io/)

[OpenCV](https://opencv.org/)

[Albumentations](https://albumentations.ai/)
