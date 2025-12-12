Hand Gesture Based SOS Trigger System

This project detects hand gestures in real time using a webcam and triggers an SOS alert when a specific gesture pattern is performed. It is part of the SafeSight safety system.

The model classifies three gestures:

Fist

Palm

No Gesture

When the user moves from palm → fist and holds the fist for a required duration, the system activates the SOS alert.

1. Project Structure
hand_gesture_project/
│
├── dataset_split/
│   ├── train/
│   └── val/
│
├── model/
│   └── gesture_model.h5
│
├── train_mobilenet.py
├── webcam.py
│
├── requirements.txt
└── README.md

2. Setup Instructions
Step 1: Create a virtual environment (optional)
python -m venv venv
venv\Scripts\activate     (Windows)
source venv/bin/activate  (Mac/Linux)

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Dataset format

The dataset must follow this structure:

train/
   fist/
   palm/
   no_gesture/

val/
   fist/
   palm/
   no_gesture/


Keras automatically assigns class labels based on these folder names.

3. Training the Model

Run the training script:

python train_mobilenet.py


After training, the model is saved to:

model/gesture_model.h5


The model is based on MobileNetV2 and uses transfer learning.

4. Running Real-Time Detection

To start the webcam detection:

python webcam.py


Press Q to exit the window.

The system displays color-coded predictions:

Palm → Green

No Gesture → Yellow

Fist → Red

SOS Triggered message when the gesture matches the alert condition

5. SOS Trigger Logic

The SOS is activated only when:

The system detects palm

Then it detects fist

The fist is held for the configured number of seconds

Cooldown period prevents repeated triggers

This avoids false alerts and ensures the gesture is intentional.

6. Technical Details

Model input size: 224 × 224

Model type: MobileNetV2 (transfer learning)

Framework: TensorFlow / Keras

Real-time detection: OpenCV

A smoothing mechanism (deque) is used to stabilize predictions

The system uses threshold-based timing for gesture transition

7. Requirements

List of major dependencies:

tensorflow
opencv-python
numpy
h5py
protobuf
pillow
scikit-learn


All packages are included in the requirements.txt.

8. Future Enhancements

Connecting SOS trigger to SMS / call APIs

Integrating this module with the SafeSight mobile application

Adding more gesture classes

Improving performance using ONNX or TensorRT