from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Set the working directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the entire model
loaded_model = load_model('brain_tumor_detection_model.h5')

def names(number):
    if number == 0:
        return 'It\'s a Tumor'
    else:
        return 'No, It\'s not a tumor'

def process_image(file):
    # Extract relevant data from FileStorage object
    file_content = file.read()

    # Convert file content to a PIL Image
    image = Image.open(io.BytesIO(file_content))

    # Resize and preprocess the image
    x = np.array(image.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)

    # Make the prediction
    res = loaded_model.predict_on_batch(x)
    classification = np.argmax(res)

    # Display the image with the result text
    plt.imshow(image)
    plt.text(
        0, -10, f"Prediction: {names(classification)}",
        color='Green', fontsize=12, fontweight='bold'
    )
    plt.axis('off')  # Hide axes
    result_image_path = 'static/result_image.png'
    plt.savefig(result_image_path)
    plt.close()

    res_list = res.tolist()
    classification = int(classification)

    response = {
        'status_message': f"Prediction: {names(classification)}",
        'result_image_path': result_image_path,
        'classification': classification,
        'prediction_scores': res_list  # Include the converted list
    }
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get file from the request
    file = request.files['image']

    # Call the process_image function with the file
    response = process_image(file)
    return jsonify(response)
    #return jsonify({'result_image_path': result_image_path})

if __name__ == '__main__':
    app.run(debug=True)
