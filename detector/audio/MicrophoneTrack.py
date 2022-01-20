import pyaudio
from aiortc import MediaStreamTrack
from av.frame import Frame


class MicrophoneTrack(MediaStreamTrack):

    kind = "audio"

    # The WebRTC VAD only accepts 16-bit mono PCM audio,
    # sampled at 8000, 16000, 32000 or 48000 Hz.
    # A frame must be either 10, 20, or 30 ms in duration:

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 48000  # Throws error on invalid sample rate!
    CHUNK_DURATION_MS = 30
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM

    mic = pyaudio.PyAudio()

    stream = mic.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=SAMPLE_RATE,
                      input=True,
                      start=True,
                      frames_per_buffer=CHUNK_SIZE)

    def __init__(self):
        super().__init__()

    async def recv(self) -> Frame:
        return self.stream.read(self.CHUNK_SIZE)