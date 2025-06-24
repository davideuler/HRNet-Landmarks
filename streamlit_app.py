import streamlit as st
import cv2
import numpy as np
import torch
import os
import time
from utils import util

# --- Configuration and Model Loading ---
st.set_page_config(page_title="Facial Landmark Detection", layout="wide")

@st.cache_resource
def load_model_and_detector():
    """Loads the facial landmark model and face detector."""
    model_path = './weights/best.pt'
    detector_path = './weights/detection.onnx'

    if not os.path.exists(model_path) or not os.path.exists(detector_path):
        st.error(f"Model or detector file not found. Please ensure models are downloaded and placed in the 'weights' directory.")
        return None, None

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_data = torch.load(model_path, map_location=device)
        model = model_data['model'].float()
        model.half()
        model.eval()

        detector = util.FaceDetector(detector_path)
        return model, detector
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, detector = load_model_and_detector()

# --- Processing Function ---
def process_video(video_path, model, detector):
    """Processes a video to detect and draw facial landmarks."""
    std = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    mean = np.array([58.395, 57.12, 57.375], 'float64').reshape(1, -1)
    scale = 1.2
    
    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{os.path.basename(video_path)}")

    stream = cv2.VideoCapture(video_path)
    if not stream.isOpened():
        st.error("Error opening video stream or file")
        return None

    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_count = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while stream.isOpened():
        success, frame = stream.read()
        if not success:
            break

        current_frame += 1
        
        boxes = detector.detect(frame, (640, 640))
        boxes = boxes.astype('int32')
        for box in boxes:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            box_w, box_h = x_max - x_min, y_max - y_min

            x_min -= int(box_w * (scale - 1) / 2)
            y_min += int(box_h * (scale - 1) / 2)
            x_max += int(box_w * (scale - 1) / 2)
            y_max += int(box_h * (scale - 1) / 2)
            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, width - 1), min(y_max, height - 1)
            box_w, box_h = x_max - x_min + 1, y_max - y_min + 1
            
            if box_h > 0 and box_w > 0:
                image = frame[y_min:y_max, x_min:x_max, :]
                image = cv2.resize(image, (256, 256))
                image = image.astype('float32')
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
                cv2.subtract(image, mean, image)
                cv2.multiply(image, 1 / std, image)
                image = image.transpose((2, 0, 1))
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).unsqueeze(0)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                image = image.to(device)
                image = image.half()

                output = model(image)
                output = output.cpu().detach().numpy()
                num_lms = output.shape[1]
                H, W = output.shape[2], output.shape[3]

                for i in range(num_lms):
                    heatmap = output[0, i, :, :]
                    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    x, y = int(x / W * box_w), int(y / H * box_h)
                    cv2.circle(frame, (x + x_min, y + y_min), 1, (0, 255, 0), 2)
        
        writer.write(frame)
        
        progress = (current_frame / frame_count)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {current_frame}/{frame_count}")

    stream.release()
    writer.release()
    status_text.text("Processing complete!")
    return output_path

# --- Streamlit UI ---
st.title("🎯 High-Resolution Facial Landmark Detection")
st.write("Upload a video to detect and visualize 68 facial landmarks. The model runs on GPU if available.")

if model is None or detector is None:
    st.stop()

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    tfile_path = os.path.join(temp_dir, uploaded_file.name)
    with open(tfile_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(tfile_path)
    
    if st.button("Process Video"):
        with st.spinner("Analyzing video and detecting landmarks... This may take a while."):
            output_video_path = process_video(tfile_path, model, detector)
        
        if output_video_path:
            st.success("Video processed successfully!")
            st.video(output_video_path)
            with open(output_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name=os.path.basename(output_video_path),
                    mime="video/mp4"
                )
        else:
            st.error("Failed to process the video.")
    
    if os.path.exists(tfile_path):
        os.remove(tfile_path)

st.markdown("---")
st.markdown("Powered by [HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection).")
