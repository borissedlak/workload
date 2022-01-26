# from https://sparkle-mdm.medium.com/python-real-time-facial-recognition-identification-with-cuda-enabled-4819844ffc80
# and https://www.youtube.com/watch?v=HsuKxjQhFU0
# and https://www.youtube.com/watch?v=GXcy7Di1oys
# and https://github.com/aiortc/aiortc/blob/main/examples/server/server.py

# used 10.2 cuda version and 8.3.2.44_cuda10.2 for cuDNN
# used 11.4.3 cuda version and 8.2.2.26 for cuDNN
# commented in rtcrtpreceiver.py

import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

from aiohttp import web
from aiohttp.web_request import Request
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpreceiver import RemoteStreamTrack

import ModelParser
import Models
from AudioTransformTrack import AudioTransformTrack
from VideoTransformTrack import VideoTransformTrack

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
providers = set()
providerTracks = list()
transformedTracks = list()
consumers = set()
consumerTracks = set()
relay = MediaRelay()

activeModel = None


async def index(request):
    content = open(os.path.join(ROOT, "consumer/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "consumer/client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def consume(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    consumers.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            consumers.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            print("How come audio is here?")
            # pc.addTrack(player.audio)
            # pc.addTrack(video_player.audio)

        elif track.kind == "video":
            pc.addTrack(relay.subscribe(transformedTracks[-1]))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def provide(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type='offer')
    tag = params['tag'] if 'tag' in params else None

    pc = RTCPeerConnection()
    providers.add(pc)
    print('Number of clients: ', len(providers))

    @pc.on('track')
    async def on_track(track: RemoteStreamTrack):
        providerTracks.append(track)

        chain = activeModel.getChainForSource(track.kind, tag)
        if track.kind == 'video':
            transformTrack = VideoTransformTrack(track, privacy_chain=chain)
        else:
            transformTrack = AudioTransformTrack(track, privacy_chain=chain)

        transformTrack.run()
        transformedTracks.append(transformTrack)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': 'answer'
        })
    )


async def updatePrivacyModel(request: Request):
    global activeModel
    model_raw = await request.text()
    model_parsed = ModelParser.parseModel(model_raw)

    activeModel = model_parsed
    for track in transformedTracks:
        c = activeModel.getChainForSource(track.kind, track.tag)
        c.printInfo()
        track.update_model(c)


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in providers] + [pc.close() for pc in consumers]
    await asyncio.gather(*coros)
    providers.clear()
    consumers.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio / video / data-channels demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--privacy-model", default=Models.faces_pixelate, help="Privacy model encoded as String")
    parser.add_argument("--port", type=int, default=4000, help="Port for HTTP server (default: 4000)")
    parser.add_argument("--verbose", "-v", action="count")
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

    activeModel = ModelParser.parseModel(args.privacy_model)
    for chain in activeModel.chains:
        chain.printInfo()

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post('/provide', provide)
    app.router.add_post("/consume", consume)
    app.router.add_post("/privacyModel", updatePrivacyModel)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
