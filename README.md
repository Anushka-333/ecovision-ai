# EcoVision AI - Garbage Detection and Classification System

This project implements an AI-powered garbage detection and classification system using YOLOv8, OpenCV, and Flask. The system detects waste objects in images and real-time video streams, automatically classifying them into Wet Waste (organic), Dry Waste (recyclable), and Hazardous waste categories.

## 🌟 New Features (V2 Update)

- **Premium UI Overhaul**: The application has been redesigned with a stunning dark-mode aesthetic, utilizing modern glassmorphism UI elements, dynamic micro-animations, and high-quality typography.
- **Voice Feedback Integration**: Added an automated text-to-speech feature using the Web Speech API. The system now verbally announces the classification results when analyzing an uploaded image!
- **Fixed Classification Logic**: Resolved bugs in the static image detection pipeline, ensuring YOLO bounding boxes and categories are successfully returned and processed.
- **Improved Heuristics**: Expanded the COCO-to-Waste mapping rules to provide more reliable out-of-the-box performance when using the base YOLOv8 model.

## Features

- **Real-time Detection**: Use your webcam for live, zero-latency waste detection.
- **Image Upload**: Upload static images (drag-and-drop supported) for deep analysis.
- **Multi-category Classification**: Intelligently classifies into Wet, Dry, and Hazardous waste.
- **Confidence Scores**: Displays AI detection confidence for each individual object.
- **Color-Coded Bounding Boxes**: Visual boxes around detected objects (Green for Wet, Blue for Dry).
- **Voice Alerts**: Audible confirmations of the overall classification.

## 🏗️ Project Structure

```
ML Project/
├── app.py                         # Main Flask application (Server)
├── waste_classifier.py            # Classification engine & logic mapping
├── train_waste_model.py           # Script to train a custom waste model
├── requirements.txt               # Python dependencies
├── training/                      # Custom model training pipeline
├── templates/                     # Premium HTML templates (index, upload, result, video)
├── static/                        # Static files (stores processed images)
├── model/                         # Directory for custom trained models (best.pt)
├── uploads/                       # Directory for uploaded raw images
└── README.md                      # This file
```

## 🚀 Installation & Usage

1. **Install Python dependencies**:
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   Start the Flask server:
   ```bash
   python app.py
   ```

3. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## 🧠 Model Information

### How it Works
The project uses the **YOLOv8n (nano)** object detection model created by Ultralytics.
By default, it utilizes a pre-trained model (trained on the general **COCO Dataset**) to identify common everyday objects. Our custom script (`waste_classifier.py`) then applies heuristic logic to categorize these detected objects into "Wet" or "Dry" waste.

### Training Your Own Custom Model
For maximum accuracy in a real-world waste management scenario, it is recommended to train the model specifically on garbage.
1. Run `python training/prepare_dataset.py` to download the TrashNet dataset.
2. Run `python train_waste_model.py` to start the YOLO training pipeline.
3. Once finished, the new `best.pt` model is automatically copied to the `model/` folder, and the application will seamlessly switch to using it!

## 🎨 Classification Categories

### Wet Waste (Organic) - Green Boxes
- Fruits: banana, apple, orange
- Vegetables: broccoli, carrot, leafy greens
- Food waste: sandwich, pizza, leftovers
- Organic materials

### Dry Waste (Recyclable) - Blue Boxes
- **Plastic**: bottles, containers, bags
- **Metal**: cans, foil, containers
- **Glass**: bottles, jars, cups
- **Paper**: newspapers, cardboard, books
- **Household**: laptops, phones, keyboards, mice, remotes

### Hazardous Waste - Red Boxes
- Batteries, chemicals, medicines, paint cans

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License
This project is open-source and available under the MIT License.

---
**Happy waste sorting! ♻️**