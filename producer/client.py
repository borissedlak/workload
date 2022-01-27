import argparse
import json
import logging
import os
import ssl

import requests
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCStatsReport
from aiortc.contrib.media import MediaPlayer
from requests import ConnectTimeout

from util import getTupleFromStats

ROOT = os.path.dirname(__file__)


class Client:
    def __init__(self):
        self.pc = None
        self.producer_rtts = list()

    async def createOffer(self, transceiverType):
        self.pc = RTCPeerConnection()
        self.pc.addTransceiver(transceiverType, direction='sendonly')

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

    async def connectVideo(self, request):

        await self.createOffer("video")

        # TODO: should I maybe only add the track once the connection stands?
        #  Otherwise I always get 2s of video at start
        if args.media_source:
            options = {"framerate": "30", "video_size": "640x480"}
            mediaSource = MediaPlayer(args.media_source, options=options)
        else:
            options = {"framerate": "30", "video_size": "640x480"}
            mediaSource = MediaPlayer(
                "video=USB-Videogerät", format="dshow", options=options
            )

        # TODO: Maybe try reconnect if state lost?
        @self.pc.on('signalingstatechange')
        async def signalingstatechange():
            print("signalingState: " + self.pc.signalingState)

        @self.pc.on('connectionstatechange')
        async def connectionstatechange():
            print("connectionState: " + self.pc.connectionState)

        @self.pc.on('iceconnectionstatechange')
        async def iceconnectionstatechange():
            print("iceConnectionState: " + self.pc.iceConnectionState)

        self.pc.addTrack(mediaSource.video)

        data = json.dumps({
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type,
            "tag": "webcam"})

        try:
            response = requests.post("http://localhost:4000/provide", timeout=10.0, data=data).json()
        except ConnectTimeout:
            mediaSource.video.stop()
            await self.pc.close()
            print("Error: Could not connect to remote server ...")
            return web.Response(status=504, content_type="text/plain", text="Connection request timed out")
        answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])

        await self.pc.setRemoteDescription(answer)

        return web.Response(content_type="text/plain", text="Started")

    async def connectAudio(self, request):

        await self.createOffer("audio")

        options = {"ch": "1", "bits": "16", "rate": "32000"}
        mediaSource = MediaPlayer(
            "audio=Mikrofonarray (Realtek(R) Audio)", format="dshow", options=options
        )

        self.pc.addTrack(mediaSource.audio)

        data = json.dumps({
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type})

        try:
            response = requests.post("http://localhost:4000/provide", timeout=10.0, data=data).json()
        except ConnectTimeout:
            mediaSource.video.stop()
            await self.pc.close()
            print("Error: Could not connect to remote server ...")
            return web.Response(status=504, content_type="text/plain", text="Connection request timed out")
        answer = RTCSessionDescription(sdp=response["sdp"], type=response["type"])

        await self.pc.setRemoteDescription(answer)
        return web.Response(content_type="text/plain", text="Started")

    async def stop(self, request):
        if self.pc is not None and self.pc.connectionState == "connected":
            await self.pc.close()
            return web.Response(content_type="text/plain", text="Connection closed")
        else:
            return web.Response(status=503, content_type="text/plain", text="Was not even connected...")

    async def calculate_stats(self, request):

        if self.pc is None:
            return web.Response(status=503, content_type="text/plain", text="Producer stream not yet started")

        consumer_stats: RTCStatsReport = await self.pc.getStats()
        rtt, timestamp = getTupleFromStats(consumer_stats)
        self.producer_rtts.append((rtt, timestamp))

        return web.Response(content_type="text/plain", text=f"Added new rtt to temp list, {rtt}, {timestamp}")

    async def persist_stats(self, request):

        if len(self.producer_rtts) == 0:
            return web.Response(status=406, content_type="text/plain", text="No RTT measurements collected")

        rtts = self.producer_rtts.copy()
        self.producer_rtts.clear()

        f = open('../evaluation/csv_export/producer_rtt.csv', 'w+')

        f.write('rtt,timestamp\n')
        for rtt in rtts:
            f.write(f'{rtt[0]},{rtt[1]}\n')

        f.close()
        return web.Response(content_type="text/plain", text=f"Wrote {len(rtts)} tuples to csv")


if __name__ == "__main__":
    client = Client()
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
    parser.add_argument("--auto-start", help="Should the server start on load?", default=True),
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

    # if args.auto_start:
    #     asyncio.run(client.connect(None))

    app.router.add_get("/startVideo", client.connectVideo)
    app.router.add_get("/startAudio", client.connectAudio)
    app.router.add_get("/stop", client.stop)
    app.router.add_get("/calculate_stats", client.calculate_stats)
    app.router.add_post("/persist_stats", client.persist_stats)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
