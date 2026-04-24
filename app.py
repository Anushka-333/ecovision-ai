from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os
from werkzeug.utils import secure_filename
from waste_classifier import EnhancedWasteClassifier

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize enhanced waste classifier
model_path = 'model/best.pt'
classifier = EnhancedWasteClassifier(model_path)

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload image and perform detection"""
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform detection using enhanced classifier
            detections = classifier.detect_and_classify(filepath)
            
            # Read the image
            img = cv2.imread(filepath)
            
            # Draw detections on image
            img = classifier.draw_detections(img, detections)
            
            # Determine overall waste classification
            overall = classifier.get_overall_classification(detections)
            
            # Save processed image to static folder
            processed_filename = 'processed_' + filename
            processed_path = os.path.join('static', processed_filename)
            cv2.imwrite(processed_path, img)
            
            # Render result template with processed image and detection details
            return render_template('result.html', 
                                 image_url=url_for('static', filename=processed_filename),
                                 detections=detections,
                                 overall=overall)
    
    return render_template('upload.html')

@app.route('/video')
def video():
    """Real-time video detection page"""
    return render_template('video.html')

def gen():
    """Generator function for video streaming"""
    cap = cv2.VideoCapture(0)  # Open webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection using enhanced classifier
        detections = classifier.detect_and_classify_from_frame(frame)
        
        # Draw detections on frame
        frame = classifier.draw_detections(frame, detections)
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)