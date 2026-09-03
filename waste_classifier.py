import cv2
import numpy as np
import os
from ultralytics import YOLO

class EnhancedWasteClassifier:
    """
    Enhanced waste classification system with improved accuracy
    Combines object detection with rule-based and ML-based classification
    """

    def __init__(self, model_path=None):
        """Initialize the classifier with custom model if available"""
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model: {model_path}")
        else:
            self.model = YOLO('yolov8n.pt')
            print("Using pretrained YOLOv8n model")

        # Enhanced waste classification rules
        self.waste_categories = {
            'wet_waste': {
                'high_confidence': [
                    'banana', 'apple', 'orange', 'broccoli', 'carrot', 'food waste',
                    'vegetable', 'fruit', 'sandwich', 'hot dog', 'pizza', 'donut',
                    'cake', 'meat', 'chicken', 'fish', 'bread', 'rice', 'pasta',
                    'milk', 'cheese', 'egg', 'butter'
                ],
                'medium_confidence': [
                    'bowl', 'plate', 'dining table'  # Often associated with food
                ]
            },
            'dry_waste': {
                'plastic': [
                    'bottle', 'cup', 'plastic_bottle', 'plastic_bag', 'plastic_container',
                    'plastic_cup', 'plastic_toy', 'plastic_utensil', 'bowl'
                ],
                'metal': [
                    'metal_can', 'aluminum_foil', 'metal_container', 'tin_can',
                    'fork', 'knife', 'spoon'
                ],
                'glass': [
                    'wine glass', 'glass_bottle', 'glass_jar', 'glass_cup'
                ],
                'paper': [
                    'book', 'paper', 'newspaper', 'magazine', 'cardboard',
                    'cardboard_food', 'office_paper'
                ],
                'electronics': [
                    'cell phone', 'remote', 'keyboard', 'mouse', 'laptop',
                    'microwave', 'toaster', 'refrigerator'
                ]
            },
            'hazardous': [
                'battery', 'chemical_container', 'medicine', 'paint_can'
            ]
        }

        # Custom trained model classes (if using custom model)
        self.custom_classes = {
            0: "plastic_bottle",
            1: "metal_can",
            2: "paper",
            3: "cardboard",
            4: "organic_waste",
            5: "glass_bottle",
            6: "plastic_bag",
            7: "aluminum_foil",
            8: "newspaper",
            9: "food_waste"
        }

    def classify_waste_enhanced(self, class_name, confidence):
        """
        Enhanced classification with confidence-based rules & precise matching
        """
        class_lower = class_name.lower().strip()

        # Check hazardous waste first (highest priority)
        for item in self.waste_categories['hazardous']:
            if item.lower() == class_lower or item.lower() in class_lower.split('_'):
                return 'hazardous'

        # Direct mapping for COCO classes and custom classes
        coco_waste_mapping = {
            # Wet waste (Organic / Biodegradable)
            'banana': 'wet',
            'apple': 'wet',
            'orange': 'wet',
            'broccoli': 'wet',
            'carrot': 'wet',
            'sandwich': 'wet',
            'hot dog': 'wet',
            'pizza': 'wet',
            'donut': 'wet',
            'cake': 'wet',
            'potted plant': 'wet',

            # Dry waste (Recyclable / Household)
            'bottle': 'dry',
            'wine glass': 'dry',
            'cup': 'dry',
            'fork': 'dry',
            'knife': 'dry',
            'spoon': 'dry',
            'bowl': 'dry',
            'book': 'dry',
            'cell phone': 'dry',
            'remote': 'dry',
            'keyboard': 'dry',
            'mouse': 'dry',
            'laptop': 'dry',
            'microwave': 'dry',
            'toaster': 'dry',
            'sink': 'dry',
            'refrigerator': 'dry',
            'scissors': 'dry',
            'teddy bear': 'dry',
            'hair drier': 'dry',
            'toothbrush': 'dry',
            'backpack': 'dry',
            'umbrella': 'dry',
            'handbag': 'dry',
            'tie': 'dry',
            'suitcase': 'dry',
            'frisbee': 'dry',
            'skis': 'dry',
            'snowboard': 'dry',
            'sports ball': 'dry',
            'kite': 'dry',
            'baseball bat': 'dry',
            'baseball glove': 'dry',
            'skateboard': 'dry',
            'surfboard': 'dry',
            'tennis racket': 'dry',
            'chair': 'dry',
            'couch': 'dry',
            'bed': 'dry',
            'dining table': 'dry',
            'toilet': 'dry',
            'tv': 'dry',
            'clock': 'dry',
            'vase': 'dry',
            
            # Custom waste classes
            'plastic_bottle': 'dry',
            'metal_can': 'dry',
            'paper': 'dry',
            'cardboard': 'dry',
            'organic_waste': 'wet',
            'glass_bottle': 'dry',
            'plastic_bag': 'dry',
            'aluminum_foil': 'dry',
            'newspaper': 'dry',
            'food_waste': 'wet'
        }

        # Check direct mapping
        if class_lower in coco_waste_mapping:
            return coco_waste_mapping[class_lower]

        # Check wet waste keywords with exact token match
        wet_keywords = ['banana', 'apple', 'orange', 'fruit', 'food', 'vegetable', 'organic', 'compost']
        words = class_lower.replace('_', ' ').split()
        if any(w in wet_keywords for w in words):
            return 'wet' if confidence > 0.4 else 'unknown'

        # Check dry waste categories (exact word match, avoid substring match like 'car' in 'cardboard')
        for category, items in self.waste_categories['dry_waste'].items():
            for item in items:
                item_lower = item.lower()
                if class_lower == item_lower or item_lower in words or class_lower in item_lower.split('_'):
                    return 'dry' if confidence > 0.4 else 'unknown'

        # Check wet waste categories
        for item in self.waste_categories['wet_waste']['high_confidence']:
            item_lower = item.lower()
            if class_lower == item_lower or item_lower in words:
                return 'wet' if confidence > 0.4 else 'unknown'

        return 'unknown'

    def detect_and_classify(self, image_path):
        """
        Detect objects and classify waste with enhanced accuracy
        """
        # Run detection
        results = self.model(image_path)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Get class name
                if hasattr(self.model, 'names') and cls in self.model.names:
                    class_name = self.model.names[cls]
                elif cls in self.custom_classes:
                    class_name = self.custom_classes[cls]
                else:
                    class_name = f"class_{cls}"

                # Enhanced classification
                category = self.classify_waste_enhanced(class_name, conf)

                detections.append({
                    'class_name': class_name,
                    'category': category,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

        return detections

    def detect_and_classify_from_frame(self, frame):
        """
        Detect objects and classify waste from video frame
        """
        # Run detection
        results = self.model(frame)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Get class name
                if hasattr(self.model, 'names') and cls in self.model.names:
                    class_name = self.model.names[cls]
                elif cls in self.custom_classes:
                    class_name = self.custom_classes[cls]
                else:
                    class_name = f"class_{cls}"

                # Enhanced classification
                category = self.classify_waste_enhanced(class_name, conf)

                detections.append({
                    'class_name': class_name,
                    'category': category,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

        return detections

    def get_overall_classification(self, detections):
        """
        Determine overall waste classification with accurate fallback
        """
        if not detections:
            return "No waste detected in the image"

        # Filter out very low confidence detections
        high_conf_detections = [d for d in detections if d.get('confidence', 0) > 0.2]
        if not high_conf_detections:
            return "Unable to classify waste type - low confidence detections"

        # Extract categories
        categories = [d['category'] for d in high_conf_detections]

        # Filter to only known categories (excluding 'unknown')
        known_categories = [cat for cat in categories if cat != 'unknown']

        # If ALL detections are 'unknown', return clear unclassified status
        if not known_categories:
            return "Unclassified Object(s) Detected"

        # Count categories among known items
        wet_count = known_categories.count('wet')
        dry_count = known_categories.count('dry')
        hazardous_count = known_categories.count('hazardous')

        # Determine overall classification
        if hazardous_count > 0:
            return "Hazardous Waste Detected - Handle with care!"
        elif wet_count > 0 and dry_count == 0:
            return "Wet Waste (Organic)"
        elif dry_count > 0 and wet_count == 0:
            return "Dry Waste (Recyclable)"
        elif wet_count > dry_count:
            return "Mixed Waste (Mostly Wet)"
        elif dry_count > wet_count:
            return "Mixed Waste (Mostly Dry)"
        else:
            return "Mixed Waste (Wet and Dry)"

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image
        """
        if not detections:
            return image
            
        img = image.copy()

        for detection in detections:
            if 'bbox' not in detection:
                continue
                
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            conf = detection.get('confidence', 0)
            class_name = detection.get('class_name', 'unknown')
            category = detection.get('category', 'unknown')

            # Choose color based on category
            if category == 'wet':
                color = (0, 255, 0)  # Green
            elif category == 'dry':
                color = (255, 0, 0)  # Blue
            elif category == 'hazardous':
                color = (0, 0, 255)  # Red
            else:
                color = (128, 128, 128)  # Gray

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{class_name} ({category}): {conf:.2f}"

            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_height - baseline),
                         (x1 + label_width, y1), color, -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def calculate_eco_impact(self, detections):
        """
        Calculate environmental carbon savings & eco metrics from detections
        """
        co2_saved = 0.0
        energy_kwh = 0.0
        recyclable_count = 0

        for d in detections:
            cat = d.get('category', '')
            name = d.get('class_name', '').lower()

            if cat == 'dry':
                recyclable_count += 1
                if 'bottle' in name or 'plastic' in name:
                    co2_saved += 0.08
                    energy_kwh += 0.2
                elif 'can' in name or 'metal' in name or 'aluminum' in name:
                    co2_saved += 0.15
                    energy_kwh += 0.4
                elif 'glass' in name:
                    co2_saved += 0.30
                    energy_kwh += 0.5
                elif 'paper' in name or 'cardboard' in name or 'book' in name:
                    co2_saved += 0.12
                    energy_kwh += 0.3
                else:
                    co2_saved += 0.05
                    energy_kwh += 0.15
            elif cat == 'wet':
                co2_saved += 0.04
                energy_kwh += 0.05

        trees_saved = (co2_saved / 20.0) # Approx trees equivalent ratio

        return {
            'co2_saved_kg': round(co2_saved, 2),
            'energy_kwh': round(energy_kwh, 2),
            'trees_saved': round(trees_saved, 4),
            'recyclable_count': recyclable_count
        }

    def get_disposal_guide(self, class_name, category):
        """
        Return bin recommendation & disposal guide instructions for an item
        """
        name_lower = class_name.lower()
        if category == 'wet':
            return {
                'bin_name': 'Green Bin (Organic Waste)',
                'bin_color': '#10b981',
                'icon': '🥦',
                'instructions': 'Place in Green Bin for composting. Avoid mixing with plastic wrappers or metals.'
            }
        elif category == 'dry':
            if 'bottle' in name_lower or 'container' in name_lower:
                return {
                    'bin_name': 'Blue Bin (Recyclables - Plastics/Glass)',
                    'bin_color': '#3b82f6',
                    'icon': '🍾',
                    'instructions': 'Rinse out liquid residue, remove cap, and place in Blue Recycling Bin.'
                }
            elif 'can' in name_lower or 'metal' in name_lower:
                return {
                    'bin_name': 'Blue Bin (Recyclables - Metals)',
                    'bin_color': '#3b82f6',
                    'icon': '🥫',
                    'instructions': 'Empty contents completely and drop in Blue Recycling Bin for metal recycling.'
                }
            elif 'paper' in name_lower or 'cardboard' in name_lower or 'book' in name_lower:
                return {
                    'bin_name': 'Blue Bin (Paper & Cardboard)',
                    'bin_color': '#3b82f6',
                    'icon': '📦',
                    'instructions': 'Flatten cardboard boxes and keep dry before placing in Blue Bin.'
                }
            else:
                return {
                    'bin_name': 'Blue Bin (Dry Recyclables)',
                    'bin_color': '#3b82f6',
                    'icon': '♻️',
                    'instructions': 'Clean item and place in Dry Waste / Recyclables Bin.'
                }
        elif category == 'hazardous':
            return {
                'bin_name': 'Red Bin (Hazardous E-Waste)',
                'bin_color': '#ef4444',
                'icon': '⚠️',
                'instructions': 'DO NOT throw in general bins! Handle with gloves and deposit at an E-Waste/Hazardous Collection Facility.'
            }
        else:
            return {
                'bin_name': 'Gray Bin (General Trash)',
                'bin_color': '#94a3b8',
                'icon': '🗑️',
                'instructions': 'Unclassified item. Inspect material type before disposal.'
            }