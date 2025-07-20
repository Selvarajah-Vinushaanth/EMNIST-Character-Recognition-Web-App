from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import os

# Create application instance
app = Flask(__name__)

# Configure for production
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # Cache static files for 1 year
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Load model only once at startup for better performance
print("Loading model...")
model = tf.keras.models.load_model('emnist_cnn_model.keras')
print("Model loaded successfully!")

# EMNIST Balanced dataset mapping
emnist_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def preprocess_image(image):
    """Preprocess image to match EMNIST dataset format exactly as used during training"""
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array
    image_array = np.array(image)
    
    # EMNIST preprocessing: rotate 90 degrees counterclockwise and flip horizontally
    image_array = np.rot90(image_array, k=3)  # 90 degrees counterclockwise
    image_array = np.fliplr(image_array)      # flip horizontally
    
    # Invert colors (EMNIST has white text on black background)
    image_array = 255 - image_array
    
    # Normalize to [0, 1] exactly as done during training
    # Use tf.cast equivalent in numpy to ensure consistency with training
    image_array = image_array.astype('float32') / 255.0
    
    # Ensure values are in [0, 1] range
    image_array = np.clip(image_array, 0, 1)
    
    # Add channel and batch dimensions as used in the CNN model
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)   # Add batch dimension
    
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.get_json()
        image_data = data['image']

        # Decode the base64 image data
        encoded_image = image_data.split(',')[1]
        decoded_image = base64.b64decode(encoded_image)
        image = Image.open(io.BytesIO(decoded_image))

        # Preprocess the image to match EMNIST format
        image_array = preprocess_image(image)

        # Make a prediction
        prediction = model.predict(image_array)
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction[0])[-3:][::-1]
        top_predictions = [
            {
                'character': emnist_labels[idx],
                'confidence': float(prediction[0][idx])
            }
            for idx in top_indices
        ]
        
        # Main prediction is still the top one
        predicted_character = emnist_labels[top_indices[0]]
        confidence = float(prediction[0][top_indices[0]])

        return jsonify({
            'prediction': predicted_character,
            'confidence': confidence,
            'top_predictions': top_predictions
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Only used for local development
if __name__ == '__main__':
    app.run(debug=False)