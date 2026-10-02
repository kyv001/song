"""看一眼渲染出来的东西：波形 + 频谱 + 高潮放大

    python plot.py            # 读 L.wav，出 tide.png
"""
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SR = 44100
BPM = 140
BEAT = 60 / BPM

with wave.open("L.wav", "rb") as f:
    x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0

rms = float(np.sqrt(np.mean(x ** 2)))
peak = float(np.max(np.abs(x)))
print("时长 {:.1f}s  峰值 {:.3f}  RMS {:.4f}  波峰因数 {:.1f}dB  削顶 {} 个".format(
    len(x) / SR, peak, rms, 20 * np.log10(peak / rms), int(np.sum(np.abs(x) > 0.999))))

SECS = [("I 潮起", 0, 27.4), ("II 潮涌", 27.4, 41.1), ("III 主题", 41.1, 68.6),
        ("IV 退潮", 68.6, 82.3), ("V 高潮", 82.3, 123.4), ("VI 余波", 123.4, len(x) / SR)]

print("\n频段能量占比：")
for nm, a, b in SECS:
    seg = x[int(a * SR):int(b * SR)]
    X = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    tot = np.sum(X ** 2)
    out = []
    for lo, hi, lab in ((0, 60, "<60"), (60, 150, "60-150"), (150, 600, "150-600"),
                        (600, 3000, "0.6-3k"), (3000, 8000, "3-8k"), (8000, 22050, ">8k")):
        band = X[(f >= lo) & (f < hi)]
        out.append("%s %4.1f%%" % (lab, 100 * np.sum(band ** 2) / tot))
    r = float(np.sqrt(np.mean(seg ** 2)))
    print("  %-8s RMS %.4f (%+5.1f dB)  %s" % (nm, r, 20 * np.log10(r / 0.28), " | ".join(out)))

fig, ax = plt.subplots(3, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [1, 1.3, 1]})
step = 512
m = len(x) // step * step
blk = x[:m].reshape(-1, step)
tt = np.arange(blk.shape[0]) * step / SR
ax[0].plot(tt, np.max(np.abs(blk), axis=1), lw=0.5, color="#444", label="peak")
ax[0].plot(tt, np.sqrt(np.mean(blk ** 2, axis=1)), lw=1.1, color="#d33", label="RMS")
for i, (nm, a, b) in enumerate(SECS):
    if i % 2 == 0:
        ax[0].axvspan(a, b, color="#000", alpha=0.04)
    ax[0].axvline(a, color="#aaa", ls=":", lw=0.8)
    ax[0].text((a + b) / 2, 0.93, nm, ha="center", fontsize=9, color="#333")
ax[0].set_xlim(0, len(x) / SR); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("幅度"); ax[0].legend(loc="upper left", framealpha=0.9)
ax[0].set_title("《潮汐》Tide — 140 BPM — G minor — 2:17 — 左声道")

ax[1].specgram(x, NFFT=2048, Fs=SR, noverlap=1024, cmap="magma", vmin=-118, vmax=-22)
ax[1].set_ylim(0, 16000); ax[1].set_xlim(0, len(x) / SR); ax[1].set_ylabel("Hz")
for nm, a, b in SECS:
    ax[1].axvline(a, color="#fff", ls=":", lw=0.6, alpha=0.45)

z0, z1 = int(100 * SR), int(102 * SR)
ax[2].plot(np.arange(z1 - z0) / SR, x[z0:z1], lw=0.5, color="#1d4e6b")
for k in range(9):
    ax[2].axvline(k * BEAT, color="#c00", ls=":", lw=0.8)
ax[2].set_xlim(0, 2); ax[2].set_ylim(-1, 1)
ax[2].set_xlabel("秒"); ax[2].set_ylabel("幅度")
ax[2].set_title("高潮段放大 2 秒（红虚线 = 每拍，四踩底鼓 + rolling bass）")

plt.tight_layout()
plt.savefig("tide.png", dpi=110)
print("\n已写出 tide.png")
