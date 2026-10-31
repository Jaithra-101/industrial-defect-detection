# Defect Inspector — Flask web app

Web interface for the AI-powered industrial defect detection model (MobileNetV2, trained in Part 1).

## Features
- **Upload inspection** — upload one or many images and get OK / DEFECTIVE, confidence and a Grad-CAM heatmap for each
- **Live inspection** — classifies webcam frames in real time (simulates a camera on a conveyor belt)
- **Dashboard** — total inspected, defect rate, average confidence and inference time, charts, recent history
- **History** — every inspection stored in SQLite, exportable to CSV

## Project structure
```
defect_app/
├── app.py            # Flask routes
├── model_utils.py    # model loading, OpenCV preprocessing, prediction, Grad-CAM
├── database.py       # SQLite inspection history
├── templates/        # base, index (upload), result, live, dashboard
├── static/style.css
├── models/           # put defect_model.keras + model_config.json here (when running locally)
└── requirements.txt
```

## Run on your laptop
1. Copy `defect_model.keras` and `model_config.json` from `MyDrive/defect-detection/models/` into `defect_app/models/`.
2. Install the requirements (Python 3.10–3.12):
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open http://localhost:5000

Use the same TensorFlow version as Colab (check with `tf.__version__` there) if loading the model fails.

## API
`POST /api/predict` with JSON `{"image": "<base64 or data URL>", "log": false}` returns
`{label, is_defective, defect_prob, confidence, inference_ms, heatmap}`.
