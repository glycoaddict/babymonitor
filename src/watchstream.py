import cv2
import os
from dotenv import load_dotenv
from ultralytics import YOLO
from PIL import Image
import numpy as np

import threading
import queue

import time
from datetime import datetime
from zoneinfo import ZoneInfo  # Add this import

from flask import Flask, Response

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

SETTINGS = {'save_localmax_frames': False,
            'save_variable_data': False,
            'color_foreground': False,  
            'contour_area_threshold_low': 50,
            'contour_area_threshold_high': 1000,
            'bg_history': 50, # 150 works well
            'bg_varThreshold': 3,
            'bg_detectShadows': False,
            'fg_scale_factor': 0.75,
            'norm_area_threshold_for_resetting_counter': 0.0005,

            }

# Get RTSP URL, username, and password from environment variables
rtsp_url = os.getenv('RTSP_URL')
username = os.getenv('RTSP_USERNAME')
password = os.getenv('RTSP_PASSWORD')

# Construct the full RTSP URL with credentials
full_rtsp_url = f"rtsp://{username}:{password}@{rtsp_url}"

# Load the YOLO model
model = YOLO('runs/detect/train4/weights/best.pt')

# Initialize background subtractor
back_sub = cv2.createBackgroundSubtractorMOG2(history=150, 
                                              varThreshold=2,
                                              detectShadows=False,)

# Initialize a queue to hold frames
frame_queue = queue.Queue(maxsize=10)
annotated_frame_queue = queue.Queue(maxsize=10)

# Open the RTSP stream
cap = cv2.VideoCapture(full_rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        if not frame_queue.full():
            # # crop the top X pixels of the frame to exclude the datetag          
            frame = frame[200:, :]
            # resize the frame by 50%
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frame_queue.put(frame)


def process_video_loop():
    time_start = datetime.now(tz=ZoneInfo("Asia/Singapore"))
    # time_formatted = datetime.strftime(time_start, '%Y-%m-%d_%H-%M-%S.%f')
    # countour_count_log_path = f'countour_count_log_{time_formatted}.txt'
    # # recent values of variables and their frame, so that we can export images of the frame when it is a local maximum
    # recent_values_and_frames = np.zeros((10,3)) # [] of (elapsed, norm_area, frame)

    last_movement_time = time_start

    

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get(timeout=0.5)
            # print('Got a frame from the queue.')

            # get elapsed time
            current_time = datetime.now(tz=ZoneInfo("Asia/Singapore"))
            elapsed_time = (current_time - time_start).total_seconds()

            

            result = model.predict(frame, verbose=False)[0]
            annotated_frame = result.plot()
            

            # crop the foreground mask to the bbox of "person"
            xyxy = [x['box'] for x in result.summary() if x['name'] in ['person','teddy bear']]

            active_area_height = 0
            active_area_width = 0

            if xyxy:
                box = xyxy[0]
                # halve the box size, keeping a center anchor
                # Halve the box size, keeping the center the same
                SCALE_FACTOR = SETTINGS['fg_scale_factor']
                # center point
                cx = (box['x1'] + box['x2']) / 2
                cy = (box['y1'] + box['y2']) / 2
                w = (box['x2'] - box['x1'])
                h = (box['y2'] - box['y1'])

                new_w = w//2 * SCALE_FACTOR
                new_h = h//2 * SCALE_FACTOR 

                box['x1'] = int(cx - new_w)
                box['y1'] = int(cy - new_h)
                box['x2'] = int(cx + new_w)
                box['y2'] = int(cy)
                # box['y2'] = int(cy + new_h)
                

                x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
                active_area_height = y2 - y1
                active_area_width = x2 - x1
                
                # crop the frame to the bounding box of the person, making everything else black
                cropped_img = frame.copy()
                cropped_img[:, :x1] = 0
                cropped_img[:, x2:] = 0
                cropped_img[:y1, :] = 0
                cropped_img[y2:, :] = 0

                fg_mask = back_sub.apply(cropped_img)
            else:
                # Highlight moving pixels using background subtraction
                fg_mask = back_sub.apply(frame)
            

            # # # shade the foreground in red
            # if SETTINGS['color_foreground']:
            #     annotated_frame[fg_mask == 255] += np.asarray([0, 0, 150], dtype = np.uint8)
            #     annotated_frame[fg_mask == 255] = (np.clip(annotated_frame[fg_mask == 255], 0, 255)).astype(np.uint8)

            

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_indices_of_interest = []
            
            for i, contour in enumerate(contours):
                cx, cy, cw, ch = cv2.boundingRect(contour)
                if (cw >= active_area_width*0.95 or ch >= active_area_height*0.95) and not (cw >= active_area_width*0.95 and ch >= active_area_height*0.95):
                    continue
                if cv2.contourArea(contour) > SETTINGS['contour_area_threshold_low'] and cv2.contourArea(contour) < SETTINGS['contour_area_threshold_high']:
                    x, y, w, h = cv2.boundingRect(contour)
                    # cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (200, 0, 0), 1)

                    contour_indices_of_interest.append(i)
                    
                    
            # Calculate the total area of contours without overlap
            mask = np.zeros_like(fg_mask)
            contours_of_interest = [contours[i] for i in contour_indices_of_interest]
            cv2.drawContours(mask, contours_of_interest, -1, 255, thickness=cv2.FILLED)

            # Overlay the mask as green on the annotated_frame with alpha 0.5
            green_mask = np.zeros_like(annotated_frame)
            green_mask[:, :, 1] = mask  # Green channel
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, green_mask, 0.25, 0)

            total_area = cv2.countNonZero(mask)
            normalised_area = total_area / (mask.shape[0] * mask.shape[1])

            if normalised_area > SETTINGS['norm_area_threshold_for_resetting_counter']:
                # save this time
                last_movement_time = current_time
            
            

            # # format current time
            # current_time_formatted = datetime.strftime(current_time, '%Y-%m-%d %H:%M:%S.%f')

            # if SETTINGS['save_variable_data']:
            #     with open(countour_count_log_path, 'a') as f:
            #         f.write(f'{current_time_formatted}\t{elapsed_time}\t{total_area}\t{normalised_area}\n')
                
            # # fill the norm_area to recent_values_and_frames, pushing out the oldest value if the list is too long
            # recent_values_and_frames = np.roll(recent_values_and_frames, 1, axis=0)
            # recent_values_and_frames[0] = [elapsed_time, normalised_area, 0]

            # # if the norm_area is a local maximum, save the frame
            # if SETTINGS['save_localmax_frames'] and np.all(recent_values_and_frames[:,1] <= normalised_area) and normalised_area > 0:
            #     cv2.imwrite(f'frames/frame_{current_time_formatted}.png', annotated_frame)


            # calculate the seconds since the last movement
            seconds_since_last_movement = (current_time - last_movement_time).total_seconds()
            # display this as text on the frame        
            cv2.putText(annotated_frame, 
                        f'{seconds_since_last_movement:.2f}s since last movement; current norm_area={normalised_area:.4f}', 
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the annotated frame
            # cv2.imshow('RTSP Stream with Annotations', annotated_frame)

            annotated_frame_queue.put(annotated_frame)
            # print('Did put of annotated frame into queue.')
          



def generate_frames():
    print('Generate frames called.')
    
    while True:
        try:
            # Try to get a frame from the queue with a timeout of X seconds
            annotated_frame_for_view = annotated_frame_queue.get(timeout=0.5)
            ret, buffer = cv2.imencode('.jpg', annotated_frame_for_view)
            annotated_frame_for_view = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame_for_view + b'\r\n')
        except queue.Empty:
            print('annotated queue was empty')
            # If the queue is empty, return an empty response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')



# Start frame capture in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True, kwargs={'cap':cap})
capture_thread.start()

print('frame capture threading started.')

# Start the video processing loop in a separate thread
video_processing_thread = threading.Thread(target=process_video_loop)
video_processing_thread.start()

print('video_processing_threads were started.')


@app.route('/video_feed')
def video_feed():
    """Returns the video stream as a response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print('starting app')
    app.run(host='0.0.0.0', debug=True)


# Release the capture and close any OpenCV windows
video_processing_thread.join()
cap.release()
cv2.destroyAllWindows()