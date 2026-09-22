# CIFAR-10 Image Classification with ANN and FastAPI

This project trains a simple artificial neural network (ANN) on the CIFAR-10 dataset and exposes it through a lightweight FastAPI web application. It is designed to classify colored images into 10 categories such as airplane, automobile, dog, ship, and truck.

## Project Overview

The application does two main things:

1. Trains a Keras ANN on CIFAR-10 data.
2. Serves a small web interface and API for predicting the class of a user-uploaded image.

This project is a beginner-friendly deep learning example that also demonstrates how to turn a trained model into a usable web application.

## Tech Stack

- Python 3.10+
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Pillow
- NumPy

## Folder Structure

```text
.
├── app.py                      # FastAPI app and simple frontend
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── model/
│   └── cifar10_production_model.h5
├── src/
│   ├── features/
│   │   └── data_loader.py     # Dataset loading and preprocessing
│   └── models/
│       ├── __init__.py
│       ├── config.py          # Model settings and labels
│       ├── predict.py         # Prediction logic
│       └── train.py           # Model training pipeline
└── ipnb/
    └── CIFAR notebook files
```

## Model Architecture

The network is a simple feed-forward ANN built with Keras:

- Input: 32 x 32 x 3 image
- Flatten layer
- Dense(300, ReLU)
- Dense(150, ReLU)
- Dense(75, ReLU)
- Dense(36, ReLU)
- Dense(10, softmax)

### Why this architecture?

This is a basic ANN for CIFAR-10 classification. It is not a CNN, but it is still useful for learning the full ML pipeline: data loading, preprocessing, training, validation, and deployment.

## CIFAR-10 Classes

The model predicts one of the following 10 classes:

| Class ID | Label |
|----------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Setup Instructions

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn python-multipart pillow
```

## Training the Model

The model is trained with the script in [src/models/train.py](src/models/train.py).

Run:

```powershell
python -m src.models.train
```

This script will:

- download CIFAR-10 data using Keras
- normalize pixel values to the range [0, 1]
- build the ANN
- train it for 10 epochs
- save the final trained model in the `model/` folder

The output model file is:

```text
model/cifar10_production_model.h5
```

## Running the Web Application

Start the FastAPI server:

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open this URL in the browser:

```text
http://127.0.0.1:8000
```

You can upload an image and the app will return the predicted class.

## API Endpoint

### Predict an image

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "image=@your_image.jpg"
```

Example response:

```json
{
  "prediction": "dog"
}
```

## Data Preprocessing

The dataset loader in [src/features/data_loader.py](src/features/data_loader.py) does the following:

- loads CIFAR-10 from Keras
- converts image arrays to float32
- divides by 255 to scale the values into [0, 1]
- reshapes labels to 1D arrays for classification

This preprocessing is important because neural networks train better with small, normalized input values.

## Prediction Logic

The prediction pipeline is in [src/models/predict.py](src/models/predict.py).

It works like this:

1. load the saved model
2. preprocess the uploaded image
3. check that image dimensions match the expected shape
4. run model inference
5. return the class label with the highest probability

## Notes

- This project uses a simple ANN and is intended for learning and demonstration.
- CIFAR-10 is a standard benchmark dataset for image classification.
- For better accuracy, a CNN such as ResNet, MobileNet, or a custom conv net would be a better production model.

## Future Improvements

Possible upgrades for this project:

- add model evaluation metrics (accuracy, precision, recall)
- save training logs and plots
- add a better frontend UI
- convert the app to a production-ready API with Docker
- switch from ANN to CNN for stronger accuracy

## License

This project is intended for educational and learning purposes.
