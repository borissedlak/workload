# from https://github.com/aiortc/aiortc/issues/100
# and https://stackoverflow.com/questions/56582908/webrtc-connection-on-localhost-without-an-internet-connection

import asyncio
import json
import time
from contextvars import ContextVar

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from VideoDetector import VideoDetector

clients = set()
listeners = set()
listenerTracks = set()

detector = VideoDetector(use_cuda=True, output_width=100)

global_queue = ContextVar('global_queue')

class TrackMux:
    def __init__(self, track):
        self.listeners = set()
        self.track = track
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def addListener(self, listener):
        self.listeners.add(listener)

    async def run(self):
        try:
            global_queue.get()
        except LookupError:
            frame_queue = asyncio.Queue(maxsize=30)
            global_queue.set(frame_queue)
        asyncio.create_task(self.__run_mux())

    async def __run_mux(self):
        while True:
            frame = await self.track.recv()
            self.new_frame_time = time.time()
            dif = self.new_frame_time - self.prev_frame_time
            if dif != 0:
                fps = 1 / dif
                self.prev_frame_time = self.new_frame_time
                print("Producer FPS: " + str(fps))


            # print('No of Listeners', len(self.listeners))
            frame_queue = global_queue.get()
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait(frame)


class ListenerTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.__queue = asyncio.Queue()

    def queue(self):
        return self.__queue

    async def recv(self):
        frame = await self.__queue.get()
        print('Recieved frame: ', frame)
        return frame


async def listener_sdp(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type='offer')
    print(params)
    pc = RTCPeerConnection()
    print('Number of listeners: ', len(listeners))
    track = ListenerTrack()
    listenerTracks.add(track)
    pc.addTrack(track)
    listeners.add(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    print(answer)
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({
            'sdp': pc.localDescription.sdp,
            'type': 'answer'
        })
    )


async def client_sdp(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type='offer')

    pc = RTCPeerConnection()
    clients.add(pc)
    print('Number of clients: ', len(clients))

    @pc.on('track')
    async def on_track(track):
        tm = TrackMux(track)
        for lt in listenerTracks:
            print('Added track to listener: ', track, lt)
            tm.addListener(lt.queue())
        await tm.run()

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
    coros = [pc.close() for pc in clients] + [pc.close() for pc in listeners]
    await asyncio.gather(*coros)
    clients.clear()
    listeners.clear()


if __name__ == '__main__':
    app = web.Application()
    app.on_shutdown.append(on_shutdown)

    app.router.add_post('/client.py', client_sdp)
    # app.router.add_post('/listener', listener_sdp)

    # cors = aiohttp_cors.setup(app, defaults={
    #     "*": aiohttp_cors.ResourceOptions(
    #         allow_credentials=True,
    #         expose_headers="*",
    #         allow_headers="*",
    #     )
    # })
    #
    # for route in list(app.router.routes()):
    #     cors.add(route)

    web.run_app(app, port=4000)
