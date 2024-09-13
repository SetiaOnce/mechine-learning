# FastAPI and Uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import uvicorn

# Async and WebSocket
import asyncio
from websockets.exceptions import ConnectionClosed

# Image Processing
import cv2
import cvzone
from PIL import Image
from io import BytesIO
import easyocr

# YOLO and SORT
from ultralytics import YOLO
import numpy as np
from sort import *

# Utilities
import math
import json
import base64
import requests
import ast

app = FastAPI()

@app.websocket("/draw_area")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        requestParams = json.loads(await websocket.receive_text())

        if requestParams.get('type') == 'config':
            requestParams = requestParams.get('config')
        else:
            await websocket.send_text('error')
            return

        # Configuration Camera
        if requestParams['is_camera'] == 'Y':
            camera = cv2.VideoCapture(f"{str(requestParams['camera_ip'])}")
        else:
            camera = cv2.VideoCapture(requestParams['video_url'])

        # Trigger line
        trigger_line = ast.literal_eval(requestParams['trigger_line'])
        limit_x1, limit_y1 = trigger_line[0]
        limit_x2, limit_y2 = trigger_line[1]
        limits = [limit_x1, limit_y1, limit_x2, limit_y2]

        # Border Line
        polygon_points = np.array(ast.literal_eval(requestParams['border_line']), np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        alpha_mask = 0.2

        if not camera.isOpened():
            await websocket.send_text('error')
            return

        while True:
            success, frame = camera.read()

            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

            # Membuat overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))
            cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
            frame = cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/traffict_countings")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        requestParams = json.loads(await websocket.receive_text())

        if requestParams.get('type') == 'config':
            requestParams = requestParams.get('config')
        else:
            await websocket.send_text('error')
            return

        # Configuration Camera
        if requestParams['is_camera'] == 'Y':
            camera = cv2.VideoCapture(f"{str(requestParams['camera_ip'])}")
        else:
            camera = cv2.VideoCapture(requestParams['video_url'])

        # Yolo Utilities
        model = YOLO(f'Yolo-Weights/{requestParams["yolo_weight"]}')
        classNames = ['Bus-Besar', 'Bus-Sedang', 'Minibus', 'Motor', 'Sedan', 'Trailer', 'Truk', 'Truk-Besar',
                      'Truk-Tangki']
        # Tracker Sort
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        # Trigger line
        trigger_line = ast.literal_eval(requestParams['trigger_line'])
        limit_x1, limit_y1 = trigger_line[0]
        limit_x2, limit_y2 = trigger_line[1]
        limits = [limit_x1, limit_y1, limit_x2, limit_y2]

        # Border Line
        polygon_points = np.array(ast.literal_eval(requestParams['border_line']), np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        alpha_mask = 0.2

        # Counters for each vehicle type
        totalBusBesar = []
        totalBusSedang = []
        totalMinibus = []
        totalMotor = []
        totalSedan = []
        totalTrailer = []
        totalTruk = []
        totalTrukBesar = []
        totalTrukTangki = []

        objectCounters = {
            'Bus-Besar': totalBusBesar,
            'Bus-Sedang': totalBusSedang,
            'Minibus': totalMinibus,
            'Motor': totalMotor,
            'Sedan': totalSedan,
            'Trailer': totalTrailer,
            'Truk': totalTruk,
            'Truk-Besar': totalTrukBesar,
            'Truk-Tangki': totalTrukTangki
        }
        frame_count = 0
        skip_frames = 2
        while True:
            frame_count += 1
            success, frame = camera.read()
            if not success or frame_count % skip_frames != 0:
                continue
            success, frame = camera.read()
            if success:
                # Perform detection
                results = model(frame)[0]
                # 1.OBJECT TRACKING
                detections = np.empty((0, 5))
                for box in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    centroid_x = int(x1 + w / 2)
                    centroid_y = int(y1 + h / 2)

                    inside_polygon = cv2.pointPolygonTest(polygon_points, (centroid_x, centroid_y), False)
                    if inside_polygon >= 0:
                        conf = math.ceil((score * 100)) / 100
                        cls = int(class_id)
                        currentClass = classNames[cls]
                        if currentClass in objectCounters and conf > 0.5:
                            if currentClass in ['Minibus', 'Sedan']:
                                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(234, 57, 114))
                                cvzone.putTextRect(frame, f'{currentClass}', (max(0, x1), max(35, y1)),
                                                   colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)
                            if currentClass in ['Bus-Besar', 'Bus-Sedang']:
                                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(255, 151, 62))
                                cvzone.putTextRect(frame, f'{currentClass}', (max(0, x1), max(35, y1)),
                                                   colorR=(255, 151, 62), scale=0.6, thickness=1, offset=10)
                            if currentClass in ['Trailer', 'Truk', 'Truk-Besar', 'Truk-Tangki']:
                                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(108, 65, 241))
                                cvzone.putTextRect(frame, f'{currentClass}', (max(0, x1), max(35, y1)),
                                                   colorR=(108, 65, 241), scale=0.6, thickness=1, offset=10)
                            if currentClass in ['Motor']:
                                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(234, 57, 114))
                                cvzone.putTextRect(frame, f'{currentClass}', (max(0, x1), max(35, y1)),
                                                   colorR=(234, 57, 114), scale=0.6, thickness=1, offset=10)

                            currentArray = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, currentArray))

                # 2.OBJECT COUNTING
                resultTracker = tracker.update(detections)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
                for result in resultTracker:
                    x1, y1, x2, y2, Id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    # Check jika melewati line yang sudah ditentukan
                    if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
                        # Check apakah Id Sudah masuk Ke dalam counting
                        for vehicle_type in objectCounters.keys():
                            if currentClass == vehicle_type and Id not in objectCounters[vehicle_type]:
                                objectCounters[vehicle_type].append(Id)
                                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                                resultOfCounting = {
                                    'total_bus_besar': len(totalBusBesar),
                                    'total_bus_sedang': len(totalBusSedang),
                                    'total_minibus': len(totalMinibus),
                                    'total_motor': len(totalMotor),
                                    'total_sedan': len(totalSedan),
                                    'total_trailer': len(totalTrailer),
                                    'total_truk': len(totalTruk),
                                    'total_truk_besar': len(totalTrukBesar),
                                    'total_tanki': len(totalTrukTangki)
                                }
                                await websocket.send_json(resultOfCounting)
                                # send counting
                                await websocket.send_json({'counting_category': cls})

                # Membuat overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_points], (0, 255, 0))
                cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                frame = cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=1)

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    await websocket.send_bytes(buffer.tobytes())

                await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/sumbu_counting")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        requestParams = json.loads(await websocket.receive_text())

        if requestParams.get('type') == 'config':
            requestParams = requestParams.get('config')
        else:
            await websocket.send_text('error')
            return

        # Configuration Camera
        if requestParams['is_camera'] == 'Y':
            camera = cv2.VideoCapture(f"{str(requestParams['camera_ip'])}")
        else:
            camera = cv2.VideoCapture(requestParams['video_url'])

        # Trigger line
        trigger_line = ast.literal_eval(requestParams['trigger_line'])
        limit_x1, limit_y1 = trigger_line[0]
        limit_x2, limit_y2 = trigger_line[1]
        limits = [limit_x1, limit_y1, limit_x2, limit_y2]

        # Border Line
        polygon_points = np.array(ast.literal_eval(requestParams['border_line']), np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        alpha_mask = 0.2

        # Yolo Utilities
        modelTire = YOLO('Yolo-Weights/tirefinder.pt')
        modelYolo = YOLO("Yolo-Weights/yolov8n.pt")

        classNamesTire = ["ban"]
        classNamesYolo = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                          "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                          "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                          "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                          "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
                          ]

        tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
        def is_within_box(box, x, y):
            x1, y1, x2, y2 = box
            return x1 <= x <= x2 and y1 <= y <= y2

        frame_count = 0
        skip_frames = 2
        while True:
            frame_count += 1
            ret, frame = camera.read()
            if not ret or frame_count % skip_frames != 0:
                continue
            else:
                resultsTire = modelTire(frame, stream=True)
                resultsYolo = modelYolo(frame, stream=True)

                vehicle_boxes = []
                tire_boxes = []

                # Deteksi Kendaraan
                detectionsId = np.empty((0, 5))
                for r in resultsYolo:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])

                        w, h = x2 - x1, y2 - y1
                        centroid_x = int(x1 + w / 2)
                        centroid_y = int(y1 + h / 2)

                        inside_polygon = cv2.pointPolygonTest(polygon_points, (centroid_x, centroid_y), False)
                        if inside_polygon >= 0:
                            if classNamesYolo[cls] in ["car", "truck", "bus"]:
                                # cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 255))
                                # cvzone.putTextRect(frame, f'{classNamesYolo[cls]}', (max(0, x1), max(35, y1)), scale=0.6,
                                #                    thickness=1, offset=3,
                                #                    colorR=(0, 255, 0))
                                vehicle_boxes.append((x1, y1, x2, y2))
                                currentArray = np.array([x1, y1, x2, y2, conf])
                                detections = np.vstack((detectionsId, currentArray))

                # Deteksi roda
                for r in resultsTire:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        cls = int(box.cls[0])
                        currentClass = classNamesTire[cls]
                        # detect ban di dalam bounding box
                        w, h = x2 - x1, y2 - y1
                        centroid_x = int(x1 + w / 2)
                        centroid_y = int(y1 + h / 2)
                        inside_polygon = cv2.pointPolygonTest(polygon_points, (centroid_x, centroid_y), False)
                        if inside_polygon >= 0:
                            if currentClass == "ban" and conf > 0.3:
                                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(0, 255, 255))
                                cvzone.putTextRect(frame, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                                   scale=0.6, thickness=1, offset=3, colorR=(0, 255, 0))
                                tire_boxes.append((x1, y1, x2, y2))

                resultTracker = tracker.update(detectionsId)
                for result in resultTracker:
                    print(result)
                    x1, y1, x2, y2, Id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(frame, f'{int(Id)}', (max(0, x1), max(35, y1)),
                                       scale=0.6, thickness=1, offset=10)

                # Hitung roda pada kendaraan
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 2)
                for vehicle in vehicle_boxes:
                    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle
                    tire_count = 0
                    for tire in tire_boxes:
                        tire_x1, tire_y1, tire_x2, tire_y2 = tire
                        if (is_within_box(vehicle, tire_x1, tire_y1) or
                                is_within_box(vehicle, tire_x2, tire_y2) or
                                is_within_box(vehicle, tire_x1, tire_y2) or
                                is_within_box(vehicle, tire_x2, tire_y1)):
                            tire_count += 1

                    # Display tire count for each vehicle
                    cvzone.putTextRect(frame, f'Sumbu : {tire_count}', (vehicle_x1, vehicle_y2 + 30), scale=1,
                                       thickness=2, offset=3, colorR=(0, 255, 0))

                # Membuat overlay
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon_points], (0, 0, 255))
                cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                frame = cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=1)

                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    await websocket.send_bytes(buffer.tobytes())

                await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/plate_detection")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        requestParams = json.loads(await websocket.receive_text())

        if requestParams.get('type') == 'config':
            requestParams = requestParams.get('config')
        else:
            await websocket.send_text('error')
            return

        # Configuration Camera
        camera = cv2.VideoCapture(
            requestParams['camera_ip'] if requestParams['is_camera'] == 'Y' else requestParams['video_url'])

        # Trigger line
        trigger_line = ast.literal_eval(requestParams['trigger_line'])
        limit_x1, limit_y1 = trigger_line[0]
        limit_x2, limit_y2 = trigger_line[1]
        limits = [limit_x1, limit_y1, limit_x2, limit_y2]

        # Border Line
        polygon_points = np.array(ast.literal_eval(requestParams['border_line']), np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        alpha_mask = 0.2

        # Yolo Utilities
        license_plate_detector = YOLO('Yolo-Weights/license_plate_detector.pt')
        license_plate_recognition = YOLO('Yolo-Weights/licence_plate_recognition.pt')

        # Define license plate class labels
        license_plate_class = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

        frame_count = 0
        skip_frames = 2
        while True:
            frame_count += 1
            ret, frame = camera.read()
            if not ret or frame_count % skip_frames != 0:
                continue
            else:
                # detect license plates
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    centroid_x = int(x1 + w / 2)
                    centroid_y = int(y1 + h / 2)
                    inside_polygon = cv2.pointPolygonTest(polygon_points, (centroid_x, centroid_y), False)
                    if inside_polygon >= 0:
                        # Crop license plate
                        license_plate_crop = frame[y1:y2, x1:x2, :]

                        cx, cy = x1 + w // 2, y1 + h // 2
                        cvzone.cornerRect(frame, (x1, y1, w, h), l=10, rt=2, colorR=(234, 57, 114))
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                        resultsPlateRecognitions = license_plate_recognition(license_plate_crop, stream=True)
                        sorted_boxes = []
                        for r in resultsPlateRecognitions:
                            boxes = r.boxes
                            for box in boxes:
                                xplate1, yplate1, xplate2, yplate2 = box.xyxy[0]
                                xplate1, yplate1, xplate2, yplate2 = int(xplate1), int(yplate1), int(xplate2), int(
                                    yplate2)
                                conf = math.ceil((box.conf[0] * 100)) / 100
                                cls = int(box.cls[0])

                                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2,
                                                  colorR=(152, 255, 152))
                                # Append the detected plate
                                sorted_boxes.append((xplate1, cls))

                        # Sort the boxes based on the x-coordinate (left to right)
                        sorted_boxes.sort(key=lambda x: x[0])
                        # Combine the characters in sorted order
                        detected_text = "".join(license_plate_class[cls] for _, cls in sorted_boxes)
                        if detected_text:
                            cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)
                            if len(detected_text) >= 5 and len(detected_text) <= 9:
                                # Check jika melewati line yang sudah ditentukan
                                vertical_tolerance, horizontal_tolerance = 10, 10
                                if (limits[0] - horizontal_tolerance < cx < limits[2] + horizontal_tolerance) and \
                                        (limits[1] - vertical_tolerance < cy < limits[3] + vertical_tolerance):

                                    retPlate, bufferPlate = cv2.imencode('.jpg', license_plate_crop)
                                    resultOfRecognition = {
                                        'plate_number': detected_text,
                                        'plate_image': base64.b64encode(bufferPlate).decode('utf-8')
                                    }
                                    await websocket.send_json(resultOfRecognition)
                                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]),
                                             (0, 255, 0), 4)
            # Line Trigger
            cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 4)
            # Membuat overlay Area
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_points], (0, 0, 255))
            cv2.addWeighted(overlay, alpha_mask, frame, 1 - alpha_mask, 0, frame)
            frame = cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/dimension01")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        requestParams = json.loads(await websocket.receive_text())

        if requestParams.get('type') == 'config':
            requestParams = requestParams.get('config')
        else:
            await websocket.send_text('error')
            return

        # Configuration Camera
        camera = cv2.VideoCapture(
            requestParams['camera_ip'] if requestParams['is_camera'] == 'Y' else requestParams['video_url'])

        # Alpha mask
        alpha_mask = 0.2
        # Border Line reference
        polygon_reference_points = np.array(ast.literal_eval(requestParams['border_line']), np.int32)
        polygon_reference_points = polygon_reference_points.reshape((-1, 1, 2))
        # Border Line object
        polygon_object_points = np.array(ast.literal_eval(requestParams['border_object_line']), np.int32)
        polygon_object_points = polygon_object_points.reshape((-1, 1, 2))

        # Yolo Utilities
        model_references = YOLO("Yolo-Weights/paper_segmentation.pt")
        model_object = YOLO("Yolo-Weights/yolov8n.pt")

        # Panjang Real Reference
        real_width_cm = 21.0
        real_length_cm = 29.7

        # Variabel untuk menghitung referensi
        reference_width_in_pixels = None
        reference_length_in_pixels = None
        scale_factor_width = None
        scale_factor_length = None

        frame_count = 0
        skip_frames = 2
        while True:
            frame_count += 1
            ret, frame = camera.read()
            if not ret or frame_count % skip_frames != 0:
                continue
            else:
                # Detect References
                references = model_references(frame)[0]
                if len(references.boxes) > 0:
                    for ref in references.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = map(int, ref[:6])

                        w, h = x2 - x1, y2 - y1
                        centroid_x = int(x1 + w / 2)
                        centroid_y = int(y1 + h / 2)
                        inside_polygon = cv2.pointPolygonTest(polygon_reference_points, (centroid_x, centroid_y), False)

                        if inside_polygon >= 0:
                            if reference_width_in_pixels is None:
                                # Ukur lebar dan tinggi referensi dalam piksel
                                reference_width_in_pixels = w
                                reference_length_in_pixels = h

                                # Hitung faktor skala
                                scale_factor_width = real_width_cm / reference_width_in_pixels
                                scale_factor_length = real_length_cm / reference_length_in_pixels

                            # Titik-titik sudut bounding box
                            top_left_corner = (x1, y1)
                            top_right_corner = (x2, y1)
                            bottom_right_corner = (x2, y2)
                            bottom_left_corner = (x1, y2)

                            # Gambar area polygon
                            corners = [top_left_corner, top_right_corner, bottom_right_corner, bottom_left_corner]
                            box_points = np.array(corners, np.int32)
                            cv2.polylines(frame, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Mengukur dimensi objek lain dalam piksel
                phones = model_object(frame)[0]
                if len(phones.boxes) > 0:
                    for phone in phones.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = phone
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        w, h = x2 - x1, y2 - y1
                        centroid_x = int(x1 + w / 2)
                        centroid_y = int(y1 + h / 2)
                        inside_object_polygon = cv2.pointPolygonTest(polygon_object_points, (centroid_x, centroid_y),
                                                              False)

                        if inside_object_polygon >= 0 and class_id == 67:
                            # Mengurangi ukuran bounding box
                            margin = 5
                            new_x1, new_y1 = max(x1 + margin, 0), max(y1 + margin, 0)
                            new_x2, new_y2 = x2 - margin, y2 - margin
                            # Titik-titik sudut bounding box
                            top_left_corner = (new_x1, new_y1)
                            top_right_corner = (new_x2, new_y1)
                            bottom_right_corner = (new_x2, new_y2)
                            bottom_left_corner = (new_x1, new_y2)

                            # Gambar area polygon
                            corners = [top_left_corner, top_right_corner, bottom_right_corner,bottom_left_corner]
                            box_points = np.array(corners, np.int32)
                            box_points = box_points.reshape((-1, 1, 2))
                            cv2.polylines(frame, [box_points], isClosed=True, color=(233, 196, 128), thickness=1)

                            # Gambar Titik-titik sudut bounding box
                            cv2.circle(frame, top_left_corner, 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, top_right_corner, 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, bottom_right_corner, 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, bottom_left_corner, 5, (0, 255, 0), cv2.FILLED)

                            # Hitung lebar dan panjang dalam piksel
                            pixel_width = math.sqrt((top_right_corner[0] - top_left_corner[0]) ** 2 +
                                                    (top_right_corner[1] - top_left_corner[1]) ** 2)
                            pixel_length = math.sqrt((top_left_corner[0] - bottom_left_corner[0]) ** 2 +
                                                     (top_left_corner[1] - bottom_left_corner[1]) ** 2)

                            # Menghitung dimensi dalam cm menggunakan faktor skala
                            width_dimension, length_dimension = 0, 0
                            if scale_factor_width is not None:
                                width_dimension = pixel_width * scale_factor_width
                            if scale_factor_length is not None:
                                length_dimension = pixel_length * scale_factor_length

                            # Kirim hasil pengukuran via WebSocket
                            resultOfMeasuring = {
                                'lebar': round(width_dimension, 1),
                                'panjang': round(length_dimension, 1),
                            }
                            await websocket.send_json(resultOfMeasuring)

                # Frame Overlay
                overlay_reference = frame.copy()
                cv2.fillPoly(overlay_reference, [polygon_reference_points], (0, 0, 255))
                cv2.addWeighted(overlay_reference, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                frame = cv2.polylines(frame, [polygon_reference_points], isClosed=True, color=(0, 0, 255), thickness=1)
                # Frame Overlay
                overlay_object = frame.copy()
                cv2.fillPoly(overlay_object, [polygon_object_points], (210, 0, 0))
                cv2.addWeighted(overlay_object, alpha_mask, frame, 1 - alpha_mask, 0, frame)
                frame = cv2.polylines(frame, [polygon_object_points], isClosed=True, color=(210, 0, 0), thickness=1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0)
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)