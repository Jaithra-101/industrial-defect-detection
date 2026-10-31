# AI-Powered Industrial Defect Detection

A computer-vision quality inspection system that classifies cast metal parts (submersible pump impellers) as **OK** or **DEFECTIVE**, with Grad-CAM explanations and a Flask web app for image upload and real-time webcam inspection.

Manual visual inspection in factories is slow and inconsistent. This project automates it with a deep learning model — the perception and decision module that an automated inspection robot on a production line would use.

## Features
- **Deep learning classifier** — custom CNN baseline and MobileNetV2 transfer learning (TensorFlow/Keras)
- **Explainable AI** — Grad-CAM heatmaps show where the model sees the defect
- **Threshold tuning** — decision threshold chosen to minimise missed defects
- **Web app (Flask)** — single/batch image upload with confidence scores and heatmaps
- **Live inspection** — classifies webcam frames in real time
- **Dashboard** — defect rate, average confidence, inference time, charts and history (SQLite), CSV export
- **REST API** — `POST /api/predict` for integration with other systems (e.g. a robot controller)

## Tech stack
Python · TensorFlow / Keras · OpenCV · scikit-learn · Flask · SQLite · Bootstrap · Chart.js · Google Colab

## Dataset
[Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (Kaggle) — ~7,300 grayscale images of submersible pump impellers, labelled `ok_front` / `def_front`.
The dataset is not included in this repository; the training notebook downloads it automatically.

## Results (unseen test set)
| Model | Accuracy | Precision (defective) | Recall (defective) | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline CNN (0.42M params) | 99.30% | 98.91% | 100.00% | 99.45% | 0.9999 |
| **MobileNetV2 fine-tuned (2.26M params)** | **99.44%** | **99.56%** | **99.56%** | **99.56%** | **0.9998** |

Test set: 715 images (453 defective, 262 OK). Metrics at the default 0.5 threshold; the web app uses a tuned threshold of 0.85, which gives zero false alarms on the test set while still catching 99.3% of defects. Inference time: ~180 ms per image including Grad-CAM. Full metrics, confusion matrices, training curves and Grad-CAM examples are in the [training notebook](notebooks/Defect_Detection_Part1_Training.ipynb).

## Project structure
```
├── notebooks/
│   ├── Defect_Detection_Part1_Training.ipynb   # data, training, evaluation, Grad-CAM
│   └── Defect_Detection_Part2_WebApp.ipynb     # optional: run the web app inside Colab
└── defect_app/
    ├── app.py              # Flask routes and REST API
    ├── model_utils.py      # model loading, OpenCV preprocessing, prediction, Grad-CAM
    ├── database.py         # SQLite inspection history
    ├── templates/          # upload, results, live inspection, dashboard pages
    ├── static/style.css
    ├── models/             # defect_model.keras + model_config.json (trained model)
    └── requirements.txt
```

## Run locally
Requires Python 3.10–3.12 (TensorFlow does not support newer versions yet).

```bash
cd defect_app
pip install -r requirements.txt
python app.py
```
Open http://localhost:5000

## Retrain the model
Open `notebooks/Defect_Detection_Part1_Training.ipynb` in Google Colab with a T4 GPU and run all cells. The trained model and config are saved to Google Drive; copy them into `defect_app/models/`.
