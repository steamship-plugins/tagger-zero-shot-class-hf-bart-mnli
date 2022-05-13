# !pip install websockets
import asyncio
import json
import uuid
import websockets
import os
from typing import List


"""
This class is a helper for plugins to use models hosted on Hugging Face.
It handles creation of a websocket for sending a large number of requests as quickly as possible.
The code is adapted from HuggingFace's websocket python example.
"""
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

async def call_model_async(block_texts: List[str], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):

    uri = f"wss://api-inference.huggingface.co/bulk/stream/{hf_compute_type}/{hf_model_path}"
    async with websockets.connect(uri) as websocket:
        # inputs and parameters are classic, "id" is a way to track that query
        payloads = [
            {
                "id": str(i),
                "inputs": text,
                "parameters": {"candidate_labels": ["medical", "fashion", "politics"]},
            }
            for i, text in enumerate(block_texts)
        ]
        last_id = payloads[-1]["id"]
        future = send(websocket, payloads, hf_bearer_token)
        future_r = recv(websocket, last_id)
        _, outputs = await asyncio.gather(future, future_r)
    results = [out["labels"][0] for out in outputs]
    return results

def call_model(block_texts: List[str], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(call_model_async(block_texts, hf_bearer_token, hf_model_path, hf_compute_type))