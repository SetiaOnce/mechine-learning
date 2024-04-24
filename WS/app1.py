from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from fastapi.templating import Jinja2Templates
import uvicorn    #  WEBSOCKET
import asyncio
import cv2
from ultralytics import YOLO

app = FastAPI()
#camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera = cv2.VideoCapture(r"C:\Users\Admin\PythonLession\pic\people1.mp4")
templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            success, frame = camera.read()

            if not success:
                break
            else:

                model = YOLO(r"C:\Users\Admin\PythonLession\YoloModel\yolov8n.pt")
                result = model.predict(frame, device = [0])

                frame = result[0].plot()
                cv2.rectangle(frame,(10,5),(40,300),(255,0,0),2)

                ret, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_text("WEB CAM PHAM XUAN KY")
                await websocket.send_bytes(buffer.tobytes())
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")


if __name__ == '__main__':
    #uvicorn.run(app, host='localhost', port=8000)
    #uvicorn.run(app, host='192.168.1.12', port=8000)
    uvicorn.run(app, port=8000, host='0.0.0.0')
