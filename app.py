import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import os

app = Flask(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Loading the model
loaded_model = load_model('brain_tumor_detection_model.h5')

def names(number):
    if number == 0:
        return 'It\'s a Tumor'
    else:
        return 'No, It\'s not a tumor'
    
result_image_path = 'static/result_image.png'  

def plotimage(image,classification):
    plt.imshow(image)
    plt.text(
        0, -10, f"Prediction: {names(classification)}",
        color='Green', fontsize=12, fontweight='bold'
    )
    plt.axis('off') 
    plt.savefig(result_image_path)
    plt.close()
    
def process_image(file):
    # Extract relevant data from FileStorage object
    file_content = file.read()

    # Convert file content to a PIL Image
    image = Image.open(io.BytesIO(file_content))

    x = np.array(image.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)

    res = loaded_model.predict_on_batch(x)
    classification = np.argmax(res)

    plotimage(image,classification)

    res_list = res.tolist()
    classification = int(classification)

    response = {
        'status_message': f"Prediction: {names(classification)}",
        'result_image_path': result_image_path,
        'classification': classification,
        'prediction_scores': res_list  
    }
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']

    response = process_image(file)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
