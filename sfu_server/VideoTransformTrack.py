import asyncio
from datetime import datetime

from aiortc import MediaStreamTrack
from av import VideoFrame

from ModelParser import PrivacyChain
from VideoDetector import VideoDetector
from util import FPS_, diffAsStringInMS


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"
    detector = VideoDetector(display_stats=False)

    def __init__(self, track, tag=None, privacy_chain=None, provision_timeout=10.0, measure_live_time=False):
        super().__init__()
        self.track = track
        self.receive_fps = FPS_("Queue Receive FPS: ", calculate_avg=30)
        self.transform_fps = FPS_("Transformation FPS: ", calculate_avg=30)
        self.provision_timeout = provision_timeout
        self.tag = tag

        self.frame_queue = asyncio.Queue(maxsize=30)
        self.task = None
        self.detector.privacy_chain = privacy_chain

        self.measure_live_time = measure_live_time
        self.first_frame_received = datetime.now()
        self.live_time_received = []
        self.live_time_transformed = []

    def update_model(self, new_model: PrivacyChain):
        self.detector.privacy_chain = new_model

    def run(self):
        # Runs the receiving loop in the background
        self.task = asyncio.get_event_loop().create_task(self.__run_receive())

    async def update_frame(self):
        frame = await self.track.recv()

        if self.measure_live_time:
            delta = (datetime.now() - self.first_frame_received)
            if delta.seconds > 30:
                print("Exporting live times after 30s")
                self.first_frame_received = datetime.now()
                self.export_live_time()
                self.live_time_received = []
                self.live_time_transformed = []

        dif_receive = self.receive_fps.update_and_print()
        if self.measure_live_time:
            self.live_time_received.append((int(1000 * dif_receive), datetime.now()))

        if self.frame_queue.full():
            self.frame_queue.get_nowait()
        self.frame_queue.put_nowait(frame)

    async def __run_receive(self):
        while True:
            try:
                await asyncio.wait_for(self.update_frame(), timeout=self.provision_timeout)
            except asyncio.TimeoutError:
                self.task.cancel()
                print('\nTrack timed out after {}s without any frame incoming'.format(self.provision_timeout))

    async def recv(self):

        frame = await self.frame_queue.get()

        before = datetime.now()

        self.transform_fps.update_and_print()

        img = frame.to_ndarray(format="bgr24")
        self.detector.processImage(img=img)

        new_frame = VideoFrame.from_ndarray(self.detector.img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        if self.measure_live_time:
            dif_transformed = diffAsStringInMS(datetime.now(), before)
            self.live_time_transformed.append((dif_transformed, datetime.now()))
        return new_frame

    def export_live_time(self):

        f = open('../evaluation/csv_export/function_time/global_time_received.csv', 'w+')

        f.write('delta,timestamp\n')
        for time in self.live_time_received:
            f.write(f'{time[0]},{time[1]}\n')

        f.close()

        f = open('../evaluation/csv_export/function_time/global_time_transformed.csv', 'w+')

        f.write('delta,timestamp\n')
        for time in self.live_time_transformed:
            f.write(f'{time[0]},{time[1]}\n')

        f.close()
