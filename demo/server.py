# server.py
import asyncio
import websockets

async def handler(websocket):
    async for message in websocket:
        print("Geldi:", message)

async def main():
    async with websockets.serve(handler, "127.0.0.1", 10110):
        print("Server başladı ws://127.0.0.1:10110")
        await asyncio.Future()  # sonsuza kadar bekle

asyncio.run(main())
