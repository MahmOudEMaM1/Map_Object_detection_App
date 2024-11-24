# app.py
from flask import Flask, request, render_template, jsonify, url_for, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
from geopy.distance import geodesic

app = Flask(__name__)
model = YOLO('Aerial-Dumping-Sites.pt')

# Store detected objects globally
detected_objects_global = []

# Detection function
def predict_and_detect_with_index(chosen_model, img, conf=0.5):
    results = chosen_model.predict(img, conf=conf)
    detected_objects = []
    index = 0

    for result in results:
        for box in result.boxes:
            index += 1
            class_name = result.names[int(box.cls[0])]

            # Draw the rectangle on the image
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)

            # Collect detected object info
            detected_objects.append({
                "index": index,
                "label": class_name,
                "coordinates": {
                    "x_min": int(box.xyxy[0][0]),
                    "y_min": int(box.xyxy[0][1]),
                    "x_max": int(box.xyxy[0][2]),
                    "y_max": int(box.xyxy[0][3])
                }
            })

    return img, detected_objects

@app.route("/process_screenshot", methods=["POST"])
def process_screenshot():
    global detected_objects_global
    data = request.get_json()
    images_data = data['images']  # Expecting multiple images
    detected_objects_global = []
    output_urls = []

    for i, image_data in enumerate(images_data):
        img_data = base64.b64decode(image_data.split(',')[1])
        img = np.array(Image.open(io.BytesIO(img_data)))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Perform detection on each image
        result_img, detected_objects = predict_and_detect_with_index(model, img)
        detected_objects_global.append(detected_objects)

        # Save the processed image
        output_path = f"static/detected_map_image_{i+1}.jpg"
        cv2.imwrite(output_path, result_img)
        output_urls.append(url_for('display_image', filename=f"detected_map_image_{i+1}.jpg"))

    return jsonify({
        "images_url": output_urls,
        "detected_objects": detected_objects_global
    })

@app.route("/apply_filter", methods=["POST"])
def apply_filter():
    global detected_objects_global
    class_filter = request.form.get('class_filter', '').lower()
    filtered_objects = [obj for objects in detected_objects_global for obj in objects if class_filter in obj['label'].lower()]
    if not filtered_objects:
        return jsonify({"error": "No objects found for the specified class filter."})
    img = cv2.imread("static/detected_map_image_1.jpg")
    for obj in filtered_objects:
        cv2.rectangle(img,
                      (obj["coordinates"]["x_min"], obj["coordinates"]["y_min"]),
                      (obj["coordinates"]["x_max"], obj["coordinates"]["y_max"]),
                      (255, 0, 0), 2)
    output_path = "static/filtered_map_image.jpg"
    cv2.imwrite(output_path, img)
    return jsonify({
        "image_url": url_for('display_image', filename="filtered_map_image.jpg"),
        "filtered_objects": filtered_objects
    })

@app.route("/select_object", methods=["POST"])
def select_object():
    global detected_objects_global
    data = request.get_json()
    object_index = int(data.get('index'))
    selected_object = None
    for objects in detected_objects_global:
        selected_object = next((obj for obj in objects if obj['index'] == object_index), None)
        if selected_object:
            break
    
    if selected_object:
        img = cv2.imread("static/detected_map_image_1.jpg")
        cv2.rectangle(img,
                      (selected_object["coordinates"]["x_min"], selected_object["coordinates"]["y_min"]),
                      (selected_object["coordinates"]["x_max"], selected_object["coordinates"]["y_max"]),
                      (0, 255, 0), 3)
        output_path = "static/selected_map_image.jpg"
        cv2.imwrite(output_path, img)
        return jsonify({
            "image_url": url_for('display_image', filename="selected_map_image.jpg"),
            "selected_object": selected_object
        })
    else:
        return jsonify({"error": "Object not found"}), 404

@app.route("/display/<filename>")
def display_image(filename):
    return send_file(f"static/{filename}", mimetype='image/jpeg')

@app.route("/", methods=["GET"])
def home():
    return render_template("map_detection.html")

if __name__ == "__main__":
    app.run(debug=True)
