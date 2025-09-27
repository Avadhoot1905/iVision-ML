import os
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify

# --- Configuration ---
# The Dockerfile exposes and runs on port 7860
HOST = '0.0.0.0'
PORT = 7860
MODEL_PATH = 'mobilenetv2.pth'

# --- Model Initialization ---
app = Flask(__name__)
model = None  # Will be set by load_model() to the MobileNetV2 instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device}")

def load_model():
    """Initializes the model, loads the checkpoint, and sets up transforms."""
    global model
    try:
        # 1. Initialize MobileNetV2 architecture
        # Your model was trained on 5 classes, so we need to adjust the classifier layer
        model = models.mobilenet_v2(weights=None)
        # Adjust the classifier layer to match your model's 5 classes
        model.classifier[1] = torch.nn.Linear(model.last_channel, 5)

        # 2. Load the state dictionary (weights)
        # We need to handle potential 'cuda' mappings if the model was saved on a GPU
        # but is being loaded on a CPU (or vice-versa).
        state_dict = torch.load(MODEL_PATH, map_location=device)

        # Handle the common issue where model weights are prefixed (e.g., 'model.features...')
        # We can detect this and remove the prefix if needed.
        if 'model' in next(iter(state_dict)):
            # Remove 'model.' prefix from keys if present (common in Hugging Face or custom savings)
            new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=True)
            app.logger.info("Successfully loaded state dict with 'model.' prefix removed.")
        else:
             # Standard loading
            model.load_state_dict(state_dict, strict=True)
            app.logger.info("Successfully loaded state dict.")


        model.to(device)
        model.eval()
        app.logger.info("MobileNetV2 model loaded successfully.")

    except FileNotFoundError:
        app.logger.error(f"Model file not found at {MODEL_PATH}. Check your Docker setup.")
        # Optionally exit or run without model if acceptable
        model = None
    except Exception as e:
        app.logger.error(f"Error loading model weights: {e}")
        model = None

# Standard ImageNet normalization parameters
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the model once when the application starts
with app.app_context():
    load_model()

# --- Utility Function for Preprocessing ---
def preprocess_image(image_bytes):
    """Converts image bytes to a PyTorch tensor for prediction."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return IMAGE_TRANSFORMS(image).unsqueeze(0).to(device)

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def health_check():
    """A simple health check to confirm the service is running."""
    if model is not None:
        return jsonify({"status": "ok", "message": "MobileNetV2 API is ready", "model_loaded": True}), 200
    else:
        return jsonify({"status": "error", "message": "Model failed to load"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and returns a prediction."""
    if model is None:
        return jsonify({"error": "Model not available"}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1. Preprocess the uploaded image
        img_bytes = file.read()
        input_tensor = preprocess_image(img_bytes)

        # 2. Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # 3. Post-process the output
        # Get the top 5 predictions and their probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_indices = torch.topk(probabilities, 5)

        # Note: You would normally load the ImageNet class names here for real labels.
        # Since I don't have your class names, I'll use indices as a placeholder.
        # In a real application, you would load a list/dictionary of class labels.

        predictions = []
        for i in range(top5_prob.size(0)):
            predictions.append({
                "label_index": top5_indices[i].item(),
                "probability": round(top5_prob[i].item() * 100, 2)
            })

        return jsonify({
            "success": True,
            "prediction_type": "Image Classification (MobileNetV2)",
            "top_predictions": predictions
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# --- Run the Flask App ---
# This check ensures the app only runs when executed directly (not when imported)
# It binds to 0.0.0.0 and the specified port (7860).
if __name__ == '__main__':
    # Increase log level for production environment
    import logging
    app.logger.setLevel(logging.INFO)
    app.run(host=HOST, port=PORT)
