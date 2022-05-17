import asyncio
import json
import websockets
from typing import List
from steamship import Block
import uuid
import logging

"""
This class is a helper for plugins to use models hosted on Hugging Face.
It handles creation of a websocket for sending a large number of requests as quickly as possible.
The code is adapted from HuggingFace's websocket python example.
"""

async def send(websocket, payloads, hf_token):
    # You need to login with a first message as headers are not forwarded
    # for websockets
    await websocket.send(f"Bearer {hf_token}".encode("utf-8"))
    for payload in payloads:
        await websocket.send(json.dumps(payload).encode("utf-8"))

async def recv(websocket, last_id):
    outputs = []
    while True:
        data = await websocket.recv()
        payload = json.loads(data)
        if payload["type"] == "results":
            # {"type": "results", "outputs": JSONFormatted results, "id": the id we sent}
            outputs.append(payload["outputs"])
            if payload["id"] == last_id:
                return outputs
        else:
            # {"type": "status", "message": "Some information about the queue"}
            logging.info(f"HF Status message: {payload['message']}")
            pass

async def call_model_async(block_texts: List[str], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):

    uri = f"wss://api-inference.huggingface.co/bulk/stream/{hf_compute_type}/{hf_model_path}"
    async with websockets.connect(uri) as websocket:
        # inputs and parameters are classic, "id" is a way to track that query
        payloads = [
            {
                "id": str(uuid.uuid4()),
                "inputs": text
            }
            for i, text in enumerate(block_texts)
        ]
        last_id = payloads[-1]["id"]
        future = send(websocket, payloads, hf_bearer_token)
        future_r = recv(websocket, last_id)
        _, outputs = await asyncio.gather(future, future_r)
    return outputs

def call_model(blocks: List[Block], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(call_model_async([block.text for block in blocks], hf_bearer_token, hf_model_path, hf_compute_type))

def call_model_on_text(blocks: List[str], hf_bearer_token : str, hf_model_path: str, hf_compute_type: str = 'cpu'):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(call_model_async(blocks, hf_bearer_token, hf_model_path, hf_compute_type))