# From https://github.com/wangshub/python-vad/blob/master/vad.py
from array import array

import pyaudio
import webrtcvad

# class AudioDetector:

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

# aggressiveness mode, which is an integer between 0 and 3.
# 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.

vad = webrtcvad.Vad(3)
raw_data = array('h')

while True:
    chunk = stream.read(CHUNK_SIZE)
    raw_data.extend(array('h', chunk))
    active = vad.is_speech(chunk, SAMPLE_RATE)

    print('1' if active else '_')

