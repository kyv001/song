"""把左右声道合成一个立体声 wav

song.py 一次只渲染一个声道：
    SIDE=L python song.py      ->  L.wav
    SIDE=R python song.py      ->  R.wav
然后：
    python merge.py            ->  song.wav（立体声）

Windows 上也可以用 add.bat（走 ffmpeg）。
"""
import wave

import numpy as np

with wave.open("L.wav", "rb") as f:
    L = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    rate = f.getframerate()
with wave.open("R.wav", "rb") as f:
    R = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)

n = min(len(L), len(R))
stereo = np.empty(n * 2, dtype=np.int16)
stereo[0::2] = L[:n]
stereo[1::2] = R[:n]

with wave.open("song.wav", "wb") as f:
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(rate)
    f.writeframes(stereo.tobytes())

print("song.wav 已生成：{:.1f} 秒 / {} Hz / 立体声".format(n / rate, rate))
