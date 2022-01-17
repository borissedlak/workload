import argparse
import json
import logging
import os
import ssl

import requests
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

ROOT = os.path.dirname(__file__)


async def negotiate(request):
    pc = RTCPeerConnection()
    pc.addTransceiver('video', direction='sendonly')

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    if args.media_source:
        options = {"framerate": "30", "video_size": "640x480"}
        mediaSource = MediaPlayer(args.media_source, options=options)
    else:
        options = {"framerate": "30", "video_size": "640x480"}
        mediaSource = MediaPlayer(
            "video=USB-Videogerät", format="dshow", options=options
        )

    pc.addTrack(mediaSource.video)

    data = json.dumps({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type})

    response = requests.post("http://localhost:4000/client", data=data).json()
    answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])

    await pc.setRemoteDescription(answer)

    return web.Response(content_type="text/plain", text="Connected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8081, help="Port for HTTP server (default: 8081)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--media-source", help="Path of video or leave empty for webcam")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None
    app = web.Application()

    app.router.add_get("/connect", negotiate)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
