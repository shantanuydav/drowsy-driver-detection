import cv2
import mediapipe as mp
import numpy as np
import time
import pygame


pygame.init()
pygame.mixer.init()
pygame.mixer.music.load("buzzer_gojek.mp3")


baseoptions = mp.tasks.BaseOptions(model_asset_path=r"D:\Drowsy Driver detection\face_landmarker.task")
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing =mp.tasks.vision.drawing_utils


left_eye= [362, 365, 387, 263, 373, 380]
right_eye =[33, 160,158,133,153,144]
mouth= [61,291,39,181,0,17,269,405]
ear_threshold= 0.81
mar_threshold= 0.6
consec_frames= 3

def euclidean(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))


def eye_aspect_ratio(landmarks, eye_points):
    p= [landmarks[i] for i in eye_points]
    vertical1= euclidean(p[1], p[5],w,h)
    vertical2= euclidean(p[2], p[4],w,h)
    horizantal= euclidean(p[0], p[3],w,h)
    return (vertical1+vertical2)/(2.0*horizantal)

def mouth_aspect_ratio(landmarks, mouth_points, w, h):
    p = [landmarks[i] for i in mouth_points]
    vertical = euclidean(p[2], p[3], w, h)
    horizontal = euclidean(p[0], p[1], w, h)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

options = FaceLandmarkerOptions(
    base_options=baseoptions,
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

blink_count=0
closed_frames= 0
eye_closed_start=None
yawn_duration_threshold= 2.0
yawn_start_time= None
yawn_count =0
yawn_active= False

try:
    with FaceLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                left_ear = eye_aspect_ratio(landmarks, left_eye)
                right_ear = eye_aspect_ratio(landmarks, right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = mouth_aspect_ratio(landmarks, mouth, w, h)

                if ear < ear_threshold:
                    closed_frames += 1
                    if closed_frames >= consec_frames :
                        eyes_closed = True
                else:
                    if closed_frames >= consec_frames and eyes_closed:
                        blink_count += 1
                        eyes_closed = False
                    closed_frames = 0

                if blink_count==5:
                    cv2.putText(frame, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pygame.mixer.music.play()
                    blink_count=0

                if ear< ear_threshold:
                    if eye_closed_start is None:
                        eye_closed_start= time.time()
                        closed_duration= time.time()-eye_closed_start
                        if closed_duration>=3:
                            cv2.putText(frame, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            pygame.mixer.music.play()
                else:
                    eye_closed_start=None

                if mar> mar_threshold:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    elif time.time() - yawn_start_time >= yawn_duration_threshold and not yawn_active:
                        yawn_count += 1
                        yawn_active = True
                        if yawn_count == 3:
                            cv2.putText(frame, "DROWSY ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            pygame.mixer.music.play()
                            yawn_count=0
                else:
                    yawn_start_time = None
                    yawn_active = False

                cv2.putText(frame, f"Blinks:{blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Yawn:{yawn_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
except Exception as e:
    print(e)