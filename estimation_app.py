import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
DEMO_IMAGE = 'stand.jpg'
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model Parameters
width, height = 368, 368
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Streamlit App Title
st.title("Human Pose Estimation with OpenCV")
st.text("Upload a clear image where all body parts are visible for optimal detection.")

# File Uploader
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
image = np.array(Image.open(img_file_buffer)) if img_file_buffer else np.array(Image.open(DEMO_IMAGE))

# Display Original Image
st.subheader('Original Image')
st.image(image, caption="Original Image", use_container_width=True)

# Threshold Slider
thres = st.slider('Threshold for detecting the key points', min_value=0, value=20, max_value=100, step=5) / 100

# Optimized Pose Detector Function
@st.cache_data
def pose_detector(frame, threshold):
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)
        x, y = int((frame_width * point[0]) / out.shape[3]), int((frame_height * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)

    for pair in POSE_PAIRS:
        part_from, part_to = pair
        id_from, id_to = BODY_PARTS[part_from], BODY_PARTS[part_to]
        if points[id_from] and points[id_to]:
            cv2.line(frame, points[id_from], points[id_to], (0, 255, 0), 3)
            cv2.ellipse(frame, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Run Pose Detection
output = pose_detector(image, thres)

# Display Pose-Estimated Image
st.subheader('Pose Estimated Image')
st.image(output, caption="Pose Estimated Image", use_container_width=True)
