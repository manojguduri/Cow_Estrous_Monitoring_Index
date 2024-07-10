from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['LOGOS_FOLDER'] = os.path.join('static', 'logos')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLO model
model = YOLO("D:\\Projects\\Eostrous\\best.pt")

# Define class names and corresponding colors
class_names = ["Lying-Down", "Mounting", "Standing"]
class_colors = {
    "Lying-Down": (0, 255, 255),  # Purple
    "Mounting": (0, 0, 255),      # Red
    "Standing": (255, 100, 0)     # Sky Blue
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/live_feed')
def live_feed():
    return render_template('live_feed.html')

def process_frame(frame, model):
    # Run detection
    results = model.predict(source=frame, conf=0.35)

    # Get the bounding boxes and class labels
    bounding_boxes = []
    class_labels = []
    for result in results:
        for box in result.boxes:
            bounding_boxes.append(box.xywh.cpu().numpy().flatten())  # Flatten the array
            class_labels.append(int(box.cls.cpu().numpy().flatten()[0]))  # Get the class labels

    detected_cells = []

    for bbox, label in zip(bounding_boxes, class_labels):
        if bbox.size == 4:  # Ensure that the bbox has 4 elements
            x, y, w, h = bbox

            # Determine the class name and color
            class_name = class_names[label]
            color = class_colors[class_name]

            # Draw the bounding box with class name and color
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(frame, top_left, bottom_right, color, 2)
            
            # Add class name to the bounding box
            cv2.putText(frame, f'{class_name}', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Get the dimensions of the frame
    height, width, _ = frame.shape
    
    # Define grid dimensions (4 vertical partitions as an example)
    num_cols = 3
    num_rows = 3  # You can adjust the number of rows as needed
    grid_width = width // num_cols
    grid_height = height // num_rows

    # Draw the grid on the frame with reduced thickness
    grid_thickness = 1
    for col in range(num_cols + 1):
        start_x = col * grid_width
        cv2.line(frame, (start_x, 0), (start_x, height), (0, 255, 0), grid_thickness)
    for row in range(num_rows + 1):
        start_y = row * grid_height
        cv2.line(frame, (0, start_y), (width, start_y), (0, 255, 0), grid_thickness)

    # Add coordinates to the grid cells
    for row in range(num_rows):
        for col in range(num_cols):
            x = (col + 0.5) * grid_width
            y = (row + 0.5) * grid_height
            cv2.putText(frame, f'({row},{col})', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    # Calculate the grid cells covered by the bounding boxes
    for bbox in bounding_boxes:
        if bbox.size == 4:  # Ensure that the bbox has 4 elements
            x, y, w, h = bbox
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))

            top_left_cell = (top_left[0] // grid_width, top_left[1] // grid_height)
            bottom_right_cell = (bottom_right[0] // grid_width, bottom_right[1] // grid_height)
            
            for row in range(top_left_cell[1], bottom_right_cell[1] + 1):
                for col in range(top_left_cell[0], bottom_right_cell[0] + 1):
                    detected_cells.append((row, col))

    # Display detected grid cells
    for cell in detected_cells:
        print(f"Object detected in cell: {cell}")

    return frame

def gen_frames():  
    cap = cv2.VideoCapture(0)  # Use the first webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame, model)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                # Process image
                image = cv2.imread(filepath)
                processed_image = process_frame(image, model)
                result_filename = f'result_{filename}'
                result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                cv2.imwrite(result_filepath, processed_image)
            else:
                # Process video
                cap = cv2.VideoCapture(filepath)
                result_filename = f'result_{filename}'
                result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

                if cap.isOpened():
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    out = cv2.VideoWriter(result_filepath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed_frame = process_frame(frame, model)
                        out.write(processed_frame)

                    cap.release()
                    out.release()
                else:
                    print(f"Error: Could not open video file {filepath}")
                    return redirect(request.url)

            return render_template('upload_form.html', filename=filename, result_filename=result_filename)

    return render_template('upload_form.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/logos/<filename>')
def logo(filename):
    return send_from_directory(app.config['LOGOS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)