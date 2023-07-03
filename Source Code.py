import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt

# Define the class labels for COCO dataset
class_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'N/A', 'backpack', 'umbrella', 'handbag', 'tie','suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Open the video file
video_capture = cv2.VideoCapture('/home/lkm/Downloads/1410202020543050_20201014205430.mp4')

# Get the video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = 1280
frame_height =720

# Define the output video path
output_path = "/home/lkm/Desktop/FasterRcnn/output.mp4"

# Create a VideoWriter object to save the output video
output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

pre_time = time.time()
pre_timeframe = 0

while True:
    # Read the next frame
    ret, frame = video_capture.read()

    if not ret:
        break

    starting_inference_time = time.time()
    frame = cv2.resize(frame, (1280, 720))

    # Get the frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Convert the PIL Image to tensor
    image_tensor = F.to_tensor(image)

    # Perform inference
    output = model([image_tensor])

    # Process the output
    boxes = output[0]['boxes'].detach().numpy()
    labels = output[0]['labels'].detach().numpy()
    scores = output[0]['scores'].detach().numpy()

    # Draw bounding boxes and labels on the frame
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            # Convert the box coordinates to integers
            box = box.astype(int)

            # Get the class label
            class_label = class_labels[label]

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(frame, class_label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate inference time and display it on the frame
    ending_inference_time = time.time()
    inference_time = ending_inference_time - starting_inference_time
    inference_time_ms = inference_time * 1000
    cv2.putText(frame, f'Inference Time: {inference_time_ms:.3f}ms', (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 0, 255), 2)

    # Calculate the FPS
    new_timeframe = time.time()
    fps = 1 / (new_timeframe - pre_timeframe)
    pre_timeframe = new_timeframe

    # Display the FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 120), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

    # Write the frame to the output video
    output_video.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and output video objects
video_capture.release()
output_video.release()

# Close all windows
cv2.destroyAllWindows()
