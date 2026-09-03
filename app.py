from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import os
import time
from werkzeug.utils import secure_filename
from waste_classifier import EnhancedWasteClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Ensure required storage directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize enhanced waste classifier
model_path = 'model/best.pt'
classifier = EnhancedWasteClassifier(model_path)

# Global store for latest live stream status
latest_live_status = {
    'overall': 'No waste detected',
    'detections': [],
    'timestamp': 0
}

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload image and perform detection"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            if not filename:
                filename = f"upload_{int(time.time())}.jpg"
                
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform detection using enhanced classifier
            detections = classifier.detect_and_classify(filepath)
            
            # Read the image
            img = cv2.imread(filepath)
            
            if img is not None:
                # Draw detections on image
                img = classifier.draw_detections(img, detections)
                
                # Determine overall waste classification
                overall = classifier.get_overall_classification(detections)

                # Attach disposal guide to each detection item
                for d in detections:
                    d['guide'] = classifier.get_disposal_guide(d['class_name'], d['category'])

                # Calculate Eco-Impact Carbon Savings metrics
                eco_impact = classifier.calculate_eco_impact(detections)
                
                # Use unique filename to prevent browser caching of processed image
                timestamp = int(time.time())
                processed_filename = f"processed_{timestamp}_{filename}"
                processed_path = os.path.join(app.config['STATIC_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, img)
                
                # Render result template with processed image, detection details & eco impact
                return render_template('result.html', 
                                     image_url=url_for('static', filename=processed_filename),
                                     detections=detections,
                                     overall=overall,
                                     eco_impact=eco_impact)
    
    return render_template('upload.html')

@app.route('/video')
def video():
    """Real-time video detection page"""
    return render_template('video.html')

def gen():
    """Generator function for video streaming with proper webcam resource cleanup"""
    global latest_live_status
    cap = cv2.VideoCapture(0)  # Open webcam
    
    if not cap.isOpened():
        print("Warning: Could not access webcam device 0.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Perform detection using enhanced classifier
            detections = classifier.detect_and_classify_from_frame(frame)
            
            # Overall status calculation
            overall = classifier.get_overall_classification(detections)
            eco_impact = classifier.calculate_eco_impact(detections)
            
            # Update global live status
            latest_live_status = {
                'overall': overall,
                'detections': detections,
                'eco_impact': eco_impact,
                'timestamp': time.time()
            }
            
            # Draw detections on frame
            frame = classifier.draw_detections(frame, detections)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        print("Webcam resource released successfully.")

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/live_status')
def live_status():
    """API endpoint returning real-time detection data for video stream UI & audio feedback"""
    return jsonify(latest_live_status)

if __name__ == '__main__':
    app.run(debug=True)