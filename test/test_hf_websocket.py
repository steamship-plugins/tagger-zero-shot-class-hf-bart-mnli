# !pip install websockets
import asyncio
import json
import uuid
import websockets
import os

MODEL_ID = "facebook/bart-large-mnli"
COMPUTE_TYPE = "cpu"  # or "gpu"

async def send(websocket, payloads, hf_token):
    # You need to login with a first message as headers are not forwarded
    # for websockets
    await websocket.send(f"Bearer {hf_token}".encode("utf-8"))
    for payload in payloads:
        await websocket.send(json.dumps(payload).encode("utf-8"))
        print("Sent ")

async def recv(websocket, last_id):
    outputs = []
    while True:
        data = await websocket.recv()
        payload = json.loads(data)
        if payload["type"] == "results":
            # {"type": "results", "outputs": JSONFormatted results, "id": the id we sent}
            print(payload["outputs"])
            outputs.append(payload["outputs"])
            if payload["id"] == last_id:
                return outputs
        else:
            # {"type": "status", "message": "Some information about the queue"}
            print(f"< {payload['message']}")
            pass

async def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test-config.json'), 'r') as config_file:
        config = json.load(config_file)
    hf_token = config['hf_api_bearer_token']



    uri = f"wss://api-inference.huggingface.co/bulk/stream/{COMPUTE_TYPE}/{MODEL_ID}"
    async with websockets.connect(uri) as websocket:
        # inputs and parameters are classic, "id" is a way to track that query
        payloads = [
            {
                "id": str(uuid.uuid4()),
                "inputs": "This is a test about a new dress shop than opened up on 5th avenue",
                "parameters": {"candidate_labels": ["medical", "fashion", "politics"]},
            }
            for i in range(10)
        ]
        last_id = payloads[-1]["id"]
        future = send(websocket, payloads, hf_token)
        future_r = recv(websocket, last_id)
        _, outputs = await asyncio.gather(future, future_r)
    results = [out["labels"][0] for out in outputs]
    return results


loop = asyncio.get_event_loop()
# if loop.is_running():
#     # When running in notebooks
#     import nest_asyncio
#     nest_asyncio.apply()
results = loop.run_until_complete(main())
# All results are "fashion"