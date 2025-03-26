import os
import json
import time
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response, stream_with_context, jsonify
from werkzeug.utils import secure_filename
from color_by_numbers import auto_color_by_numbers, generate_test_colored_image

# Configure folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for main page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle image upload
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': f"'{filename}' uploaded successfully", 'filename': filename})
    return jsonify({'error': 'File type not allowed'}), 400

# Process endpoint that streams progress via Server-Sent Events (SSE)
# Change the /process endpoint to accept GET (for SSE)
@app.route('/process', methods=['GET'])
def process():
    # Get parameters from the query string
    detail_option = request.args.get('detail_option', 'normal')  # 'basic', 'normal', 'detailed'
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    # Read the image using OpenCV
    image = cv2.imread(filepath)
    
    def generate():
        yield "data: Starting processing...\n\n"
        time.sleep(0.5)
        yield "data: Downsampling (if needed)...\n\n"
        time.sleep(0.5)
        yield "data: Applying Gaussian blur...\n\n"
        time.sleep(0.5)
        yield "data: Converting color space...\n\n"
        time.sleep(0.5)
        yield f"data: Running k-means for '{detail_option}' detail...\n\n"
        time.sleep(0.5)
        
        # Call the processing function
        output_img, master_list, final_mask = auto_color_by_numbers(
            image,
            detail_option=detail_option,
            min_region_size=1000,
            downsample_factor=2,
            apply_morph_open=False
        )
        yield "data: Finished processing.\n\n"
        
        # Save output images
        numbered_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_numbered.png')
        colored_path = os.path.join(app.config['PROCESSED_FOLDER'], 'output_colored.png')
        cv2.imwrite(numbered_path, output_img)
        test_colored_img = generate_test_colored_image(final_mask, master_list)
        cv2.imwrite(colored_path, test_colored_img)
        
        result = {
            'numbered': numbered_path,
            'colored': colored_path,
            'master_list': master_list
        }
        yield "data: " + json.dumps(result) + "\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
