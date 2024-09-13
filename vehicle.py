# FastAPI and Uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
import uvicorn

# Async and WebSocket
import asyncio
from websockets.exceptions import ConnectionClosed

# Image Processing
import cv2
import cvzone

# YOLO and SORT
from ultralytics import YOLO
import numpy as np
from sort import *

# Utilities
import math
import json
import ast

app = FastAPI()
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
                    vertical_tolerance, horizontal_tolerance = 20, 20
                    if (limits[0] - horizontal_tolerance < cx < limits[2] + horizontal_tolerance) and \
                            (limits[1] - vertical_tolerance < cy < limits[3] + vertical_tolerance):
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

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8006)
    # uvicorn.run(app, host='0.0.0.0', port=8005)