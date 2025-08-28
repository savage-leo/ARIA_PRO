import asyncio
import websockets
import json

async def run():
    uri = "ws://127.0.0.1:8100/ws"
    try:
        async with websockets.connect(uri) as ws:
            print(f"✅ Connected to {uri}")
            for i in range(5):
                msg = await ws.recv()
                data = json.loads(msg)
                print(f"← Message {i+1}:", data)
                await asyncio.sleep(1)
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(run())
