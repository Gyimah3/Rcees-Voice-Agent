from typing import AsyncIterator
from starlette.websockets import WebSocket
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def websocket_stream(websocket: WebSocket) -> AsyncIterator[str]:
    while True:
        data = await websocket.receive_text()
        # logger.info(f"data yielded{data}")
        yield data
