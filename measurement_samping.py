import cv2
import cvzone
from ultralytics import YOLO
import numpy as np


# Open Camera
# camera = cv2.VideoCapture('rtsp://user:Tc12As3lih@192.168.113.253:554/Streaming/Channels/101')
camera = cv2.VideoCapture('Videos/dimention2.MOV')
# Model PyTorch
model = YOLO("Yolo-Weights/dimension_segmentation.pt")
classNames = ["atas", "referensi", "samping"]
# Border Line
polygon_points = np.array([[262, 208],[528, 208],[528, 534],[255, 534]], np.int32)
polygon_points = polygon_points.reshape((-1, 1, 2))
alpha_mask = 0.2

# Actual ruler length cm
real_reference_height_cm = 29.7

frame_count = 0
skip_frames = 2

reference_pixel_height = None
# Read frames
while True:
    frame_count += 1
    ret, frame = camera.read()
    if not ret or frame_count % skip_frames != 0:
        continue
    else:
        frame = cv2.resize(frame, (874, 536), interpolation=cv2.INTER_LINEAR)

        results = model(frame)[0]
        if len(results.boxes) > 0:
            # Pertama cek untuk class referensi terlebih dahulu
            for res in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = map(int, res[:6])
                w, h = x2 - x1, y2 - y1
                centroid_x = int(x1 + w / 2)
                centroid_y = int(y1 + h / 2)
                currentClass = classNames[int(class_id)]

                # Titik-titik sudut bounding box
                top_left_corner = (x1, y1)
                top_right_corner = (x2, y1)
                bottom_right_corner = (x2, y2)
                bottom_left_corner = (x1, y2)
                corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]

                if currentClass == 'referensi':
                    reference_pixel_height = h - 100
                    # Gambar area polygon
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [np.array([corners], np.int32)], (0, 0, 255))
                    cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                    frame = cv2.polylines(frame, [np.array([corners], np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)

            # Lalu cek class samping setelah reference
            for res in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = map(int, res[:6])
                w, h = x2 - x1, y2 - y1
                centroid_x = int(x1 + w / 2)
                centroid_y = int(y1 + h / 2)
                currentClass = classNames[int(class_id)]

                # Titik-titik sudut bounding box
                top_left_corner = (x1, y1)
                top_right_corner = (x2, y1)
                bottom_right_corner = (x2, y2)
                bottom_left_corner = (x1, y2)
                corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]

                if currentClass == 'samping' and reference_pixel_height is not None:
                    # Gambar area polygon
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [np.array([corners], np.int32)], (0, 255, 0))
                    cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                    frame = cv2.polylines(frame, [np.array([corners], np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

                    # Gambar Titik-titik sudut bounding box
                    cv2.circle(frame, top_left_corner, 5, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, top_right_corner, 5, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, bottom_right_corner, 5, (255, 0, 0), cv2.FILLED)
                    cv2.circle(frame, bottom_left_corner, 5, (255, 0, 0), cv2.FILLED)

                    # Gambar Text
                    cvzone.putTextRect(frame, f'A', top_left_corner,colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)
                    cvzone.putTextRect(frame, f'B', top_right_corner,colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)
                    cvzone.putTextRect(frame, f'C', bottom_right_corner,colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)
                    cvzone.putTextRect(frame, f'D', bottom_left_corner,colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)

                    # Kalkulasi rasio piksel ke cm
                    pixel_to_cm_ratio = real_reference_height_cm / reference_pixel_height

                    # Hitung panjang dan tinggi dalam satuan cm
                    object_width_cm = w * pixel_to_cm_ratio
                    object_height_cm = h * pixel_to_cm_ratio

                    cvzone.putTextRect(frame, f'Panjang: {object_width_cm:.2f} cm', (20, 50), colorR=(0, 255, 0), scale=2,
                                       thickness=2, offset=10)
                    cvzone.putTextRect(frame, f'Tinggi: {object_height_cm:.2f} cm', (20, 100), colorR=(0, 255, 0), scale=2,
                                           thickness=2, offset=10)

        # Display frame
        cv2.imshow('Output', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
