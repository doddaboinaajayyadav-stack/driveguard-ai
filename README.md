# DriveGuard AI 
## Driver Drowsiness Detection Using Deep Learning

DriveGuard AI is a real-time driver eye-state monitoring prototype that uses a Convolutional Neural Network (CNN), OpenCV, PyTorch, and Flask to classify eye states as **OPEN** or **CLOSED** and display a visual warning when sustained eye closure is detected.

The project combines deep learning and computer vision with a web-based dashboard to demonstrate how eye-state classification can support driver drowsiness monitoring.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Folder Structure](#-folder-structure)
5. [Prerequisites](#-prerequisites)
6. [Installation](#-installation)
7. [Dataset Setup](#-dataset-setup)
8. [Training the CNN Model](#-training-the-cnn-model)
9. [Running the Dashboard](#-running-the-dashboard)
10. [How It Works](#-how-it-works)
11. [Model Evaluation](#-model-evaluation)
12. [Testing](#-testing)
13. [Viva Reference](#-viva-reference)
14. [Troubleshooting](#-troubleshooting)
15. [Limitations and Future Improvements](#-limitations-and-future-improvements)
16. [Safety Disclaimer](#-safety-disclaimer)

---

## 📌 Project Overview

Driver drowsiness is a road-safety concern that can reduce attention and reaction time. DriveGuard AI demonstrates a computer vision approach to monitoring eye states through a webcam.

The application detects a face and eye regions, processes the detected eye regions, and passes them to a trained CNN. The model predicts whether an eye appears open or closed. A temporal frame counter is then used to display a warning when eye closure persists.

### Key Technologies

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| CNN model | Custom Convolutional Neural Network |
| Computer vision | OpenCV |
| Face and eye detection | OpenCV Haar Cascades |
| Image processing | NumPy, Pillow |
| Web framework | Flask |
| Dashboard | HTML, CSS |
| Testing | Python `unittest` |

---

## ✨ Key Features

- Real-time webcam video streaming.
- Face and eye-region detection using OpenCV Haar Cascades.
- CNN-based OPEN/CLOSED eye-state predictions.
- Prediction confidence displayed on the video feed.
- Visual warning for sustained eye closure.
- Flask-based web dashboard.
- Separate dataset preparation, model training, and evaluation scripts.
- Automated unit tests for the CNN and dataset loader.

---

## 🏗️ System Architecture

```text
                 Webcam
                    |
                    v
           OpenCV Video Capture
                    |
                    v
         Face and Eye Detection
            (Haar Cascades)
                    |
                    v
        Eye Region Preprocessing
         Grayscale, Resize 24x24
                    |
                    v
          Trained CNN Classifier
              (PyTorch)
                    |
                    v
          OPEN / CLOSED Prediction
           + Confidence Score
                    |
                    v
          Temporal Frame Counter
                    |
                    v
        Sustained Eye Closure Warning
                    |
                    v
            Flask Dashboard
       Live Video + Visual Warning
```

---

## 📁 Folder Structure

The main project files are organized as follows:

```text
Driver Drowsiness Detection/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   └── model.py
│
├── training/
│   ├── prepare_dataset.py
│   ├── dataset_loader.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── tests/
│       └── test_model_and_dataset.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── css/
│       └── style.css
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
└── models/
    └── best_model.pt
```

**Note:** The dataset and trained model checkpoint are generated locally and excluded from Git. The exact contents of the project directories may change as development continues.

---

## 💻 Prerequisites

- Python 3.11.
- A built-in or USB webcam.
- Windows 10 or Windows 11.
- Approximately 2 GB or more of available storage, depending on dataset and environment requirements.
- An internet connection to install dependencies and obtain the dataset.

The project has been tested using Python 3.11 and CPU-based PyTorch.

---

## ⚙️ Installation

### Step 1: Open the project directory

Open PowerShell or the VS Code terminal and navigate to the project folder.

```powershell
cd "Driver Drowsiness Detection"
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
```

### Step 3: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Verify the installation

```powershell
python -c "import torch, cv2, flask, numpy, PIL; print('All core imports OK'); print('PyTorch:', torch.__version__)"
```

---

## 📂 Dataset Setup

The current CNN uses two eye-state classes:

- **Open:** Images labeled as open eyes.
- **Closed:** Images labeled as closed eyes.

The source dataset used during development was extracted into:

```text
~/Downloads/archive/train/
```

Expected source structure:

```text
train/
├── Open/
├── Closed/
├── no_yawn/
└── yawn/
```

The current model uses the `Open` and `Closed` folders.

### Prepare the dataset

Run the following command from the project root:

```powershell
python -m training.prepare_dataset
```

The preparation script:

1. Reads images from the source `Open` and `Closed` directories.
2. Shuffles the image paths using a fixed random seed.
3. Creates training, validation, and test splits.
4. Copies the images into the project dataset directories.

The current prepared dataset contains:

| Split | Open | Closed | Total |
|---|---:|---:|---:|
| Training | 508 | 508 | 1,016 |
| Validation | 109 | 109 | 218 |
| Test | 109 | 109 | 218 |
| **Total** | **726** | **726** | **1,452** |

The original dataset remains unchanged.

**Important:** The dataset is not included in the Git repository. Download and prepare it locally before training a new model.

---

## 🧠 Training the CNN Model

Activate the virtual environment and run:

```powershell
python -m training.train_model
```

The training pipeline uses PyTorch to train a CNN on the prepared dataset. It evaluates the model on the validation split and saves the best checkpoint.

The checkpoint is saved to:

```text
models/best_model.pt
```

The model accepts grayscale image tensors of shape:

```text
[batch_size, 1, 24, 24]
```

The current implementation uses CPU-based training and inference.

### Evaluate the trained model

```powershell
python -m training.evaluate_model
```

The evaluation script reports test accuracy, precision, recall, F1 score, and a confusion matrix.

---

## 🌐 Running the Dashboard

### Step 1: Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 2: Start the Flask application

```powershell
python app.py
```

The application loads the trained checkpoint and starts the Flask development server.

### Step 3: Open the dashboard

Visit:

**http://127.0.0.1:5000**

Allow webcam access if requested by your system or browser.

The dashboard displays the webcam feed, eye-state predictions when eye regions are detected, and a visual warning when the sustained-eye-closure threshold is reached.

### Step 4: Stop the application

Press `Ctrl + C` in the terminal.

**Important:** A fresh clone does not include `models/best_model.pt`. Prepare the dataset and train the model locally before expecting CNN predictions.

---

## 🔍 How It Works

### 1. Face and Eye Detection

OpenCV captures frames from the webcam. Haar Cascade classifiers detect face and eye regions within the captured frame.

### 2. Image Preprocessing

Detected eye regions are converted to grayscale and resized to 24 × 24 pixels. Pixel values are normalized before being passed to the CNN.

### 3. CNN Classification

The custom CNN predicts one of two classes:

- `OPEN`
- `CLOSED`

The model also produces a confidence score for its predicted class.

The architecture includes convolutional layers, batch normalization, ReLU activations, max pooling, dropout, and fully connected layers.

```text
Input: 1 x 24 x 24
       |
       v
Convolution + BatchNorm + ReLU + MaxPool
       |
       v
Convolution + BatchNorm + ReLU + MaxPool
       |
       v
Fully Connected Layer (128)
       |
       v
Output Layer (2 classes)
       |
       v
OPEN / CLOSED
```

### 4. Temporal Warning

The application tracks consecutive frames in which a closed-eye prediction is detected. The current warning threshold is 20 consecutive frames.

When the threshold is reached, the video feed displays a sustained-eye-closure warning.

The actual time represented by 20 frames depends on the effective frame rate. The current implementation should not be assumed to represent exactly one second in every environment.

### 5. Dashboard

Flask serves the web interface and streams webcam frames to the browser. The interface presents the live camera feed and visual prediction or warning overlays.

---

## 📊 Model Evaluation

The trained CNN was evaluated on the prepared test split.

### Evaluation Results

| Metric | Result |
|---|---:|
| Test images | 218 |
| Correct predictions | 214 |
| Incorrect predictions | 4 |
| Accuracy | 98.17% |
| Closed-eye precision | 97.30% |
| Closed-eye recall | 99.08% |
| Closed-eye F1 score | 98.18% |

### Confusion Matrix

Rows represent actual classes and columns represent predicted classes. The class order is `OPEN`, followed by `CLOSED`.

```text
                 Predicted
                 OPEN  CLOSED

Actual OPEN       106      3
Actual CLOSED       1    108
```

The model correctly classified 214 out of 218 test images.

**Interpretation:** These results indicate promising performance on the current test split. They do not establish generalization to unseen drivers, new cameras, different lighting conditions, or real-world driving scenarios.

---

## 🧪 Testing

The project includes unit tests for the CNN and dataset loader.

### Run the unit tests

```powershell
python -m unittest discover -s training\tests -v
```

Expected result:

```text
Ran 4 tests
OK
```

The tests verify:

- CNN output shape.
- Finite model outputs.
- Loading open- and closed-eye images.
- Handling of an empty dataset.

### Check Python syntax

```powershell
python -m compileall -q app.py config.py src training
```

No output generally indicates that the syntax check completed successfully.

### Re-run model evaluation

```powershell
python -m training.evaluate_model
```

The current verified results are 98.17% accuracy, 97.30% closed-eye precision, 99.08% closed-eye recall, and 98.18% closed-eye F1 score.

---

## 🎓 Viva Reference

| Question | Answer |
|---|---|
| What is DriveGuard AI? | A computer vision prototype for classifying eye states and displaying a sustained-eye-closure warning. |
| Why use PyTorch? | PyTorch provides the tensor operations, neural network layers, and training utilities required to build and train the CNN. |
| What is a CNN? | A Convolutional Neural Network is a deep learning architecture that learns spatial patterns from image data. |
| Why use OpenCV? | OpenCV captures webcam frames, processes images, and provides Haar Cascade face and eye detectors. |
| What is the model input size? | A grayscale image of 24 × 24 pixels, represented as a tensor with one channel. |
| What are the output classes? | OPEN and CLOSED. |
| What is temporal analysis? | Tracking predictions across consecutive frames instead of relying on only one frame. |
| What is the warning threshold? | The current implementation uses 20 consecutive frames with a detected closed-eye prediction. |
| What accuracy did the model achieve? | 98.17% on the current test split of 218 images. |
| Can it run without a GPU? | Yes. The current implementation supports CPU-based inference. |
| Is the model production-ready? | No. It is an educational prototype that needs broader real-world validation before safety-critical use. |

---

## 🛠️ Troubleshooting

| Problem | Possible solution |
|---|---|
| `No module named 'cv2'` | Activate `.venv` and install dependencies using `python -m pip install -r requirements.txt`. |
| `No module named 'torch'` | Activate `.venv` and install the dependencies. |
| Dataset not found | Verify that the extracted dataset exists at `~/Downloads/archive/train/`. |
| Model checkpoint not found | Prepare the dataset and run the training command. |
| Webcam unavailable | Close other applications using the camera and check camera permissions. |
| Dashboard does not load | Confirm that the Flask server is running and check the terminal for errors. |
| Port 5000 is occupied | Stop the other process using the port or adjust the Flask port configuration. |
| Predictions are unstable | Check lighting, camera position, eye detection, and dataset quality. |
| Warning does not trigger reliably | Check whether the eye detector continues to return eye regions when the eyes are closed. |

---

## 🚀 Limitations and Future Improvements

### Current Limitations

- Performance depends on face and eye detection quality.
- Lighting, camera placement, glasses, and individual differences may affect predictions.
- Closed eyes alone do not necessarily mean a person is drowsy.
- The current evaluation uses a relatively small test split from the same source dataset.
- The project has not been validated as a real-world driver-safety system.

### Future Improvements

- Improve closed-eye detection when eyes are fully shut.
- Test with separate drivers and independently collected datasets.
- Add temporal smoothing to reduce prediction flicker.
- Improve alert logic and configurable thresholds.
- Add verified audio alerts and event logging.
- Evaluate performance under varied lighting and camera conditions.
- Explore yawning detection and additional drowsiness indicators.

---

## ⚠️ Safety Disclaimer

DriveGuard AI is an educational final-year academic project. It is **not a certified driver-monitoring or accident-prevention system**.

The displayed predictions and warnings may be incorrect or delayed. Do not use this prototype as a substitute for adequate rest, safe driving practices, or certified safety systems. Never test it while driving.

---

## 👨‍💻 Project Information

**Project:** DriveGuard AI — Driver Drowsiness Detection Using Deep Learning
**Project Type:** Deep Learning and Computer Vision Laboratory Project

Built to explore CNN-based image classification, real-time computer vision, and Flask application development.
