
import os
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import keras
from keras.preprocessing import image
from keras.models import load_model
import cv2
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.static_folder = 'static'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

# Blood group classes (update based on your dataset)
BLOOD_GROUPS = {
    0: 'A+',
    1: 'A-', 
    2: 'B+',
    3: 'B-',
    4: 'AB+',
    5: 'AB-',
    6: 'O+',
    7: 'O-'
}

# Global model variable
model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _find_model_path():
    """Search for a model file in common locations."""
    candidate_paths = [
        'blood_group_model.h5',
        'best_blood_group_model.h5',
        os.path.join('models', 'blood_group_model.h5'),
        os.path.join('models', 'best_blood_group_model.h5')
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def load_ml_model():
    """Load the trained model if found; returns bool for success."""
    global model
    try:
        model_path = _find_model_path()
        if model_path:
            model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            logger.warning("Model file not found in expected locations")
            model = None
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        return False

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess image for prediction"""
    try:
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB format
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_blood_group(image_path):
    """Predict blood group from image and compute top-3 results"""
    global model
    
    # Lazy-load model on first use
    if model is None:
        loaded = load_ml_model()
        if not loaded:
            return None, "Model not loaded", 0, []
    
    try:
        # Preprocess image
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return None, "Error processing image", 0, []
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        probabilities = prediction[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index] * 100)

        # Build top-3 predictions list
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top3 = [
            {
                'blood_group': BLOOD_GROUPS.get(int(idx), "Unknown"),
                'confidence': float(probabilities[int(idx)] * 100)
            }
            for idx in top_indices
        ]
        
        # Get blood group label
        predicted_blood_group = BLOOD_GROUPS.get(predicted_index, "Unknown")
        
        return predicted_blood_group, "Success", confidence, top3
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return None, f"Prediction error: {str(e)}", 0, []

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            file.save(filepath)
            
            # Make prediction
            predicted_group, status, confidence, top3 = predict_blood_group(filepath)
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            if status == "Success":
                return jsonify({
                    'success': True,
                    'blood_group': predicted_group,
                    'confidence': f"{confidence:.2f}%",
                    'top3': [
                        {
                            'blood_group': item['blood_group'],
                            'confidence': f"{item['confidence']:.2f}%"
                        } for item in top3
                    ]
                })
            else:
                return jsonify({
                    'success': False,
                    'error': status
                })
        
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
            
    except Exception as e:
        logger.error(f"Prediction route error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    loaded = model is not None or load_ml_model()
    return jsonify({
        'status': 'healthy',
        'model_loaded': loaded
    })

if __name__ == '__main__':
    # Load model on startup
    load_ml_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)