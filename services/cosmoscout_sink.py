import requests
import base64
import cv2
import json
import numpy as np
import time

class DisplayClient:
    #def __init__(self, host="129.247.41.99", port=9003):
    def __init__(self, host="192.168.0.103", port=9003):
        self.render_server = f"http://{host}:{port}"
    
    def send_detection(self, image, bounding_boxes, inference_speed):
        """
        Send image, bounding boxes, and inference speed
        
        Args:
            image: numpy array (OpenCV image)
            bounding_boxes: list of dicts or tuples [(x, y, w, h), ...]
            inference_speed: float (ms or fps)
        """
        # Encode image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format bounding boxes as JSON string
        bbox_json = json.dumps(bounding_boxes)
        
        # Create JavaScript function call
        js_call = f'CosmoScout.satellites.updateDetection("{image_b64}", {bbox_json}, {inference_speed})'
        print(f"DEBUG - Sending to server:")
        print(f"  Body: {js_call[:100]}... [length: {len(js_call)}]")
        
        try:
            response = requests.post(
                f"{self.render_server}/run-js",
                data=js_call
            )
            return response.ok
        except Exception as e:
            print(f"Error sending detection: {e}")
            return False
    
    def send_config(self, config_id):
        """Send configuration ID"""
        js_call = f'CosmoScout.satellites.setSatelliteConfiguration({config_id})'
        
        try:
            response = requests.post(
                f"{self.render_server}/run-js",
                data=js_call
            )
            return response.ok
        except Exception as e:
            print(f"Error sending config: {e}")
            return False


# Usage
client = DisplayClient()

# Capture from webcam
cap = cv2.VideoCapture(0)
#ret, frame = cap.read()

# TESTING
# frame = np.zeros((100, 100, 3), dtype=np.uint8)
# frame[:, :] = [0, 0, 255]

# Example bounding boxes: [(x, y, width, height), ...]
bboxes = [[100, 150, 200, 250], [400, 100, 150, 200]]
inference_time = 45.2

client.send_config(1)
for i in range(6000):
    ret, frame = cap.read()
    client.send_detection(frame, bboxes, inference_time)
    time.sleep(0.016)

cap.release()