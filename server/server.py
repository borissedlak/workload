# from https://sparkle-mdm.medium.com/python-real-time-facial-recognition-identification-with-cuda-enabled-4819844ffc80
# and https://www.youtube.com/watch?v=HsuKxjQhFU0
# and https://www.youtube.com/watch?v=GXcy7Di1oys
# and https://github.com/aiortc/aiortc/blob/main/examples/server/server.py

# used 10.2 cuda version and 8.3.2.44_cuda10.2 for cuDNN

import argparse
import asyncio
import json
import logging
import os
import ssl
import time
import uuid
from contextvars import ContextVar

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame


from Detector import *
from sfu_example.server import TrackMux

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
providers = set()
providerTracks = list()
transformedTracks = list()
consumers = set()
consumerTracks = set()
relay = MediaRelay()

# transformTrack = None

detector = Detector(use_cuda=True)


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.prev_time_receive = 0
        self.new_time_receive = 0
        self.prev_time_transform = 0
        self.new_time_transform = 0
        self.frame_queue = asyncio.Queue()

    async def run(self):
        asyncio.create_task(self.__run_receive())

    async def __run_receive(self):
        while True:
            frame = await self.track.recv()
            self.new_time_receive = time.time()
            dif = self.new_time_receive - self.prev_time_receive

            if dif != 0:
                fps = 1 / dif
                self.prev_time_receive = self.new_time_receive
                print("Queue Receive FPS: " + str(fps))

            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame)

    async def recv(self):
        # await self.run()
        # try:
        # except LookupError:
        #     frame_queue = asyncio.Queue(maxsize=30)
        #     global_queue.set(frame_queue)

        frame = await self.frame_queue.get()

        self.new_time_transform = time.time()
        fps = 1 / (self.new_time_transform - self.prev_time_transform)
        self.prev_time_transform = self.new_time_transform
        print("Transformation FPS: " + str(fps))

        img = frame.to_ndarray(format="bgr24")
        detector.processImage(img=img)

        new_frame = VideoFrame.from_ndarray(detector.img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
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

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    video_player = MediaPlayer(os.path.join(ROOT, "demo/lukas-detection.mp4"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

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
        # global transformTrack
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            # pc.addTrack(player.audio)
            recorder.addTrack(track)
            # pc.addTrack(video_player.audio)

        elif track.kind == "video":
            # if transformTrack is None:
            #     # transformTrack = VideoTransformTrack(
            #     #     relay.subscribe(video_player.video), transform=params["video_transform"]
            #     # )
            #     transformTrack = VideoTransformTrack(
            #         relay.subscribe(providerTracks[0]), transform=params["video_transform"]
            #     )

            pc.addTrack(transformedTracks[0])

            # track = ListenerTrack()
            # consumerTracks.add(track)
            # pc.addTrack(track)
            # consumers.add(pc)

            # pc.addTrack(providerTracks[0])

            # if args.record_to:
            #     recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

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

    pc = RTCPeerConnection()
    providers.add(pc)
    print('Number of clients: ', len(providers))

    @pc.on('track')
    async def on_track(track):
        providerTracks.append(track)

        transformTrack = VideoTransformTrack(
            relay.subscribe(track), transform="dontknow"
        )
        await transformTrack.run()
        transformedTracks.append(transformTrack)
        # tm = TrackMux(track)
        # providerTrackMucks.append(tm)
        # for ct in consumerTracks:
        #     print('Added track to listener: ', track, ct)
        #
        # frame_queue = asyncio.Queue(maxsize=30)
        # global_queue.set(frame_queue)
        # tm.addListener(global_queue.get())

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


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in providers] + [pc.close() for pc in consumers]
    await asyncio.gather(*coros)
    providers.clear()
    consumers.clear()


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
        "--port", type=int, default=4000, help="Port for HTTP server (default: 4000)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
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

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post('/client', provide)
    app.router.add_post("/consume", consume)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
