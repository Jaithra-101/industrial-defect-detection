"""Flask web app for AI-powered industrial defect detection.

Environment variables:
  MODEL_DIR  folder with defect_model.keras + model_config.json   (default: ./models)
  DATA_DIR   folder for the SQLite database and saved images       (default: ./data)
  PORT       port to listen on                                     (default: 5000)
"""
import base64
import csv
import io
import os
import uuid

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, send_from_directory, url_for

from database import InspectionDB
from model_utils import DefectDetector, decode_image, overlay_heatmap

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(APP_DIR, 'models'))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(APP_DIR, 'data'))
MEDIA_DIR = os.path.join(DATA_DIR, 'media')
os.makedirs(MEDIA_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB per request

print('Loading model from', MODEL_DIR)
detector = DefectDetector(MODEL_DIR)
db = InspectionDB(os.path.join(DATA_DIR, 'inspections.db'))
print('Model ready. Decision threshold =', detector.threshold)


def save_result_images(image_bgr, result):
    """Save the original and the Grad-CAM overlay; return their file names inside MEDIA_DIR."""
    key = uuid.uuid4().hex
    image_file, heatmap_file = f'{key}.jpg', f'{key}_cam.jpg'
    cv2.imwrite(os.path.join(MEDIA_DIR, image_file), image_bgr)
    cv2.imwrite(os.path.join(MEDIA_DIR, heatmap_file), overlay_heatmap(image_bgr, result['heatmap']))
    return image_file, heatmap_file


def to_data_url(image_bgr):
    ok, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


@app.route('/')
def index():
    return render_template('index.html', threshold=detector.threshold)


@app.route('/inspect', methods=['POST'])
def inspect():
    results, errors = [], []
    for file in request.files.getlist('images'):
        if not file or not file.filename:
            continue
        if os.path.splitext(file.filename)[1].lower() not in ALLOWED_EXTENSIONS:
            errors.append(f'{file.filename}: unsupported file type')
            continue
        image = decode_image(file.read())
        if image is None:
            errors.append(f'{file.filename}: could not read image')
            continue
        result = detector.predict(image)
        image_file, heatmap_file = save_result_images(image, result)
        db.log('upload', file.filename, result, image_file, heatmap_file)
        result.update(filename=file.filename, image_file=image_file, heatmap_file=heatmap_file)
        results.append(result)
    if not results and not errors:
        return redirect(url_for('index'))
    summary = {'total': len(results), 'defective': sum(r['is_defective'] for r in results)}
    return render_template('result.html', results=results, errors=errors, summary=summary)


@app.route('/live')
def live():
    return render_template('live.html', threshold=detector.threshold)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON in: {"image": "<data URL or base64>", "log": bool}. Used by the live webcam page."""
    payload = request.get_json(silent=True) or {}
    data = payload.get('image', '')
    if ',' in data:
        data = data.split(',', 1)[1]
    try:
        image = decode_image(base64.b64decode(data))
    except (ValueError, TypeError):
        image = None
    if image is None:
        return jsonify(error='invalid image'), 400

    result = detector.predict(image)
    if payload.get('log'):
        image_file, heatmap_file = save_result_images(image, result)
        db.log('live', 'webcam frame', result, image_file, heatmap_file)
    return jsonify(
        label=result['label'],
        is_defective=result['is_defective'],
        defect_prob=round(result['defect_prob'], 4),
        confidence=round(result['confidence'], 4),
        inference_ms=round(result['inference_ms'], 1),
        heatmap=to_data_url(overlay_heatmap(image, result['heatmap'])),
    )


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', stats=db.stats(), rows=db.recent(50))


@app.route('/export.csv')
def export_csv():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['id', 'timestamp', 'source', 'filename', 'label', 'defect_prob', 'confidence', 'inference_ms'])
    for r in db.all_rows():
        writer.writerow([r['id'], r['timestamp'], r['source'], r['filename'], r['label'],
                         f"{r['defect_prob']:.4f}", f"{r['confidence']:.4f}", f"{r['inference_ms']:.1f}"])
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=inspection_history.csv'})


@app.route('/media/<path:filename>')
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.route('/health')
def health():
    return jsonify(status='ok', threshold=detector.threshold)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False, threaded=True)
