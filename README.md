# EcoVision AI - Garbage Detection and Classification System

This project implements an AI-powered garbage detection and classification system using YOLOv8, OpenCV, and Flask. The system detects waste objects in images and real-time video streams, automatically classifying them into Wet Waste (organic), Dry Waste (recyclable), and Hazardous waste categories.

## 🌟 New Features & Enhancements (V2.5 Update)

- **🌿 Eco-Impact Carbon Savings Dashboard**: Calculates estimated $CO_2$ emission savings (kg), energy conserved (kWh), and tree equivalents based on detected recyclable materials.
- **🇮🇳 Multi-Lingual Voice Feedback (Hindi + English)**: Switch between English and Hindi speech synthesis for audio announcements (*"Sookha Kachra (Recyclable) - Kripya Neelay Dustbin me Dalein"*).
- **🗑️ Smart Disposal & Dustbin Recommendation Guide**: Interactive guide modal for every object detailing specific bin colors (Green / Blue / Red) and preparation tips.
- **📄 One-Click PDF Report Export**: Instantly export deep waste analysis certificates as PDF reports via `html2pdf.js`.
- **📜 Recent Scan History Gallery**: Saves recent waste scans in browser `localStorage` with a popup history drawer on the homepage.
- **💎 Premium Glassmorphic UI**: Redesigned dark-mode aesthetic with modern glassmorphism UI elements, dynamic micro-animations, and high-quality typography.
- **🔧 Resolved Classification Logic & WebCam Leaks**: Fixed substring word matching bugs (`'car'` matching `'cardboard'`), unclassified fallback logic, and added webcam resource cleanup (`cap.release()`).

## Features

- **Real-time Detection**: Use your webcam for live, zero-latency waste detection.
- **Image Upload**: Upload static images (drag-and-drop supported) for deep analysis.
- **Multi-category Classification**: Intelligently classifies into Wet, Dry, and Hazardous waste.
- **Confidence Scores**: Displays AI detection confidence for each individual object.
- **Color-Coded Bounding Boxes**: Visual boxes around detected objects (Green for Wet, Blue for Dry, Red for Hazardous).
- **Multi-Lingual Voice Alerts**: Audible confirmations in Hindi and English.

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

## 🌐 Live Demo
🚀 **Try the application online here:**
- [EcoVision AI Live Demo](https://ecovision-ai-ibfn.onrender.com/)

## 📄 License
This project is open-source and available under the MIT License.

---
**Happy waste sorting! ♻️**