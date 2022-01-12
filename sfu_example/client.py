import asyncio
import json

import requests
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription


class ClientConnector:

    @staticmethod
    async def negotiate():  # request):
        pc = RTCPeerConnection()
        # sessionDesc = pc.createOffer()
        # pcs.add(pc)

        pc.addTransceiver('video', direction='recvonly')

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        data = json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type})

        response = requests.post("http://0.0.0.0:8080/client", data=data).json()

        answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])

        await pc.setRemoteDescription(answer)

        return web.Response(content_type="text/plain", text="Just text")


if __name__ == '__main__':
    connector = ClientConnector()
    asyncio.run(connector.negotiate())
