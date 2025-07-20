# EMNIST Character Recognition Web App

A web application for recognizing handwritten characters and digits using a deep learning model trained on the EMNIST dataset.

![EMNIST Character Recognition App](https://via.placeholder.com/800x400?text=EMNIST+Character+Recognition)

## Features

- **Draw Recognition**: Draw characters or digits directly on a canvas
- **Image Upload**: Upload images containing handwritten characters
- **Top Predictions**: View the top 3 most likely character predictions
- **Confidence Scores**: See the confidence level for each prediction
- **History Tracking**: Review past predictions with thumbnails
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Mode**: Toggle between light and dark themes
- **Adjustable Drawing Tools**: Change pen thickness and use eraser

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras
- **Dataset**: EMNIST Balanced

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Selvarajah-Vinushaanth/EMNIST-Character-Recognition-Web-App.git
   cd Handwritten-Digits
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Model Details

The application uses a convolutional neural network (CNN) trained on the EMNIST Balanced dataset, which includes:
- 10 digits (0-9)
- 26 uppercase letters (A-Z)
- 26 lowercase letters (a-z)

The model architecture consists of:
- 2 convolutional layers with max pooling
- Fully connected hidden layer with dropout
- Output layer with softmax activation

## Development

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- Flask

### Project Structure

```
Handwritten-Digits/
├── app.py                 # Flask application
├── emnist_cnn_model.keras # Trained model file
├── templates/
│   └── index.html         # Frontend interface
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Deployment

### Local Production Server

To run the application with a production WSGI server locally:

```bash
pip install gunicorn  # On Windows, use waitress instead
gunicorn wsgi:app     # On Windows: waitress-serve --port=8000 wsgi:app
```

### Deploying to Heroku

1. Create a Heroku account and install the Heroku CLI
2. Login to Heroku CLI:
   ```
   heroku login
   ```
3. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```
4. Push your code to Heroku:
   ```
   git push heroku main
   ```
5. Set environment variables:
   ```
   heroku config:set SECRET_KEY=your-production-secret-key
   ```

### Deploying with Docker

1. Build the Docker image:
   ```
   docker build -t emnist-recognition .
   ```
2. Run the container:
   ```
   docker run -p 8000:8000 -e SECRET_KEY=your-secret-key emnist-recognition
   ```

## Future Improvements

- Add a gallery of example characters
- Implement batch prediction for multiple characters
- Add user accounts to save prediction history
- Improve mobile touch drawing experience
- Support for more languages and character sets

## License

This project is licensed under the MIT License - see the LICENSE file for details.
