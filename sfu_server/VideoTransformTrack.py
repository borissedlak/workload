# TODO: If the track does not receive anymore frames from the remote peer, remove it and stop it
import asyncio

from aiortc import MediaStreamTrack
from av import VideoFrame

from VideoDetector import VideoDetector
from util import FPS_

detector = VideoDetector(use_cuda=True)


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, privacyModel=None, provision_timeout=10.0):
        super().__init__()
        self.track = track
        self.receive_fps = FPS_("Queue Receive FPS: ", calculate_avg=30)
        self.transform_fps = FPS_("Transformation FPS: ", calculate_avg=30)
        self.provision_timeout = provision_timeout

        self.frame_queue = asyncio.Queue(maxsize=30)
        self.task = None
        detector.privacy_model = privacyModel

    def run(self):
        # Runs the receiving loop in the background
        self.task = asyncio.get_event_loop().create_task(self.__run_receive())

    async def update_frame(self):
        frame = await self.track.recv()
        self.receive_fps.update_and_print()

        if self.frame_queue.full():
            self.frame_queue.get_nowait()
        self.frame_queue.put_nowait(frame)

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
