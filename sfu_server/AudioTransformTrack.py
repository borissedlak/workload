# TODO: If the track does not receive anymore frames from the remote peer, remove it and stop it
import asyncio

import webrtcvad
from aiortc import MediaStreamTrack
from av import VideoFrame

from VideoDetector import VideoDetector
from util import FPS_

detector = VideoDetector()
vad = webrtcvad.Vad(3)


class AudioTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, privacy_chain, provision_timeout=10.0):
        super().__init__()
        self.track = track
        self.privacy_chain = privacy_chain
        self.receive_fps = FPS_("Queue Receive FPS: ", calculate_avg=960)
        self.transform_fps = FPS_("Transformation FPS: ", calculate_avg=960)
        self.provision_timeout = provision_timeout

        self.frame_queue = asyncio.Queue(maxsize=960)
        self.task = None

    def run(self):
        # Runs the receiving loop in the background
        self.task = asyncio.get_event_loop().create_task(self.__run_receive())

    async def update_frame(self):
        frame = await self.track.recv()
        # self.receive_fps.update_and_print()

        if self.frame_queue.full():
            self.frame_queue.get_nowait()
        self.frame_queue.put_nowait(frame)
        f = frame.to_ndarray().tobytes()
        active = vad.is_speech(f[0:2880], 48000)  # There could be something wrong here...

        print('1' if active else '_')

    async def __run_receive(self):
        while True:
            try:
                await asyncio.wait_for(self.update_frame(), timeout=self.provision_timeout)
            except asyncio.TimeoutError:
                # TODO: Close the pc from this side as well
                self.task.cancel()
                print('\nTrack timed out after {}s without any frame incoming'.format(self.provision_timeout))

    async def recv(self):

        frame = await self.frame_queue.get()
        self.transform_fps.update_and_print()

        img = frame.to_ndarray(format="bgr24")
        detector.processImage(img=img)

        new_frame = VideoFrame.from_ndarray(detector.img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
