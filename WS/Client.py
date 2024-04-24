from websockets.exceptions import ConnectionClosed
import websockets
import numpy as np
import asyncio
import cv2


async def main():
    url = 'ws://127.0.0.1:8000/ws'

    async for websocket in websockets.connect(url):
        try:
            # count = 1
            while True:
                contents = await websocket.recv()
                arr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                cv2.imshow('frame', frame)
                cv2.waitKey(1)

                # cv2.imwrite("frame%d.jpg" % count, frame)
                # count += 1
        except ConnectionClosed:
            continue  # attempt reconnecting to the server (otherwise, call break)


asyncio.run(main())