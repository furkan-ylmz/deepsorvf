import asyncio
import websockets
import time
import cv2

async def stream_nmea(file_path, uri):
    prev_ts = None
    async with websockets.connect(uri) as ws:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if prev_ts is not None:
                    await asyncio.sleep(0.5)  # sabit gecikme
                prev_ts = time.time()

                await ws.send(line)
                print(f"AIS gönderildi: {line}")

async def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)

        # 33 ms bekle (yaklaşık 30 FPS)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def main():
    await asyncio.gather(
        stream_nmea("ais_all_sorted.nmea", "ws://127.0.0.1:10110"),
        asyncio.to_thread(play_video, "2022_06_04_12_05_12_12_07_02_b.mp4")  # video işlemi ayrı thread'de
    )

if __name__ == "__main__":
    asyncio.run(main())
