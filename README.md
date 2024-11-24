# YOLO Object Detection with Satellite Maps

This project demonstrates an integration of the YOLO (You Only Look Once) object detection model with satellite imagery, providing a web interface for detecting and interacting with objects within satellite map snapshots. The application utilizes the Flask framework, OpenCV, and the `Ultralytics YOLO` model for real-time object detection, and integrates with the HERE Maps API for location-based map rendering and search.

## Features

- **Satellite Map Integration:** The application uses HERE Maps API to render satellite imagery, allowing users to search for locations and capture map screenshots for object detection.
- **Object Detection with YOLO:** The system employs a pre-trained YOLO model to detect objects within satellite images. Detected objects are highlighted with bounding boxes and their coordinates are displayed.
- **Object Filtering:** Users can filter detected objects based on class labels (e.g., "car", "building") for more focused analysis.
- **Interactive Object Selection:** After detection, users can select specific objects for further interaction, highlighting the chosen object on the map and providing detailed coordinates.
- **Dynamic Image Capture:** The map is divided into a 3x3 grid, where each section is captured for detection, allowing for comprehensive coverage of a geographical area.
- **Backend Processing:** Images are processed server-side using Flask and OpenCV, with results returned as URLs for displaying processed images and detected objects.

## Workflow

1. **Map Interaction:** Users can interact with the map by searching for locations using the search bar.
2. **Image Capture:** Upon clicking the "Capture Map and Apply Detection" button, the map is captured in a grid format, and the images are sent to the server for object detection.
3. **Detection and Filtering:** The server processes the images using the YOLO model and returns the detected objects with their coordinates. Users can apply filters to display specific object types.
4. **Result Display:** Processed images and detected objects are displayed in the web interface, allowing for further analysis or interaction.

