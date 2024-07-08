import torch
import cv2
import pandas as pd
from datetime import datetime
import os




# Load YOLOv5 model (using a larger model for better accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Use yolov5m instead of yolov5s for better accuracy




# Initialize webcam
cap = cv2.VideoCapture(0)




# Set higher resolution for better accuracy (if your webcam supports it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)




# We Get the video writer initialized to save the output video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_video_path = 'output.avi'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))




# Log file setup
log_file = 'detection_log.csv'
log_columns = ['Timestamp', 'Object', 'Confidence', 'Frame']
log_data = []




# Directory to save frames
frames_dir = 'detected_frames'
os.makedirs(frames_dir, exist_ok=True)




if not cap.isOpened():
  print("Error: Could not open video.")
  exit()




frame_count = 0
confidence_threshold = 0.5  # Increased confidence threshold for better accuracy




while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
      break




  # Perform object detection
  results = model(frame)




  # Get detection results
  labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()




  # Annotate frame
  n = len(labels)
  for i in range(n):
      row = cords[i]
      if row[4] >= confidence_threshold:  # Apply the confidence threshold
          x1, y1, x2, y2 = int(row[0] * frame_width), int(row[1] * frame_height), int(row[2] * frame_width), int(row[3] * frame_height)
          bgr = (0, 255, 0)
          cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
          text = f"{model.names[int(labels[i])]} {row[4]:.2f}"
          cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)




          # Log detected objects
          timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
          log_data.append([timestamp, model.names[int(labels[i])], row[4], frame_count])




  # Save frame to video file
  out.write(frame)




  # Display the frame
  cv2.imshow('YOLOv5 Object Detection - The Pycodes', frame)




  # Press 'q' to quit or 's' to save the current frame
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
      break
  elif key == ord('s'):
      frame_path = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
      cv2.imwrite(frame_path, frame)
      print(f"Frame {frame_count} saved at {frame_path}")




  frame_count += 1




# Release everything if the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()




# Save log data to CSV
log_df = pd.DataFrame(log_data, columns=log_columns)
log_df.to_csv(log_file, index=False)




print(f"Detection log saved to {log_file}")
print(f"Annotated video saved to {output_video_path}")
print(f"Detected frames saved to {frames_dir}")
