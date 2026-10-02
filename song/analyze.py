"""渲染后自检：客观指标 + 出图

    cd song && python analyze.py embers.wav
    cd song && python analyze.py embers.wav --png embers.png

指标按 README 里那套验收标准来（波峰因数、频段能量、削顶、立体声相关度），
另外加了逐段 RMS 和「中高频相关度」——整首的相关度会被低频拉高，
真正决定「宽不宽」的是 300 Hz 以上那一部分。

只用标准库 + numpy / scipy / matplotlib 读 wav，不依赖 moviepy。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

BANDS = [(20, 60), (60, 150), (150, 400), (400, 900), (900, 3000),
         (3000, 8000), (8000, 16000)]
BAND_NAMES = ["20-60", "60-150", "150-400", "400-900", "0.9-3k", "3-8k", "8k+"]


def load(path):
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:
        x = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32:
        x = d.astype(np.float64) / 2147483648.0
    else:
        x = d.astype(np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return sr, x.T  # (2, N)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def db(x, ref=1.0):
    return 20 * np.log10(max(float(x), 1e-12) / ref)


def crest(x):
    p = float(np.max(np.abs(x)))
    return 20 * np.log10(max(p, 1e-12) / max(rms(x), 1e-12))


def correlation(a, b):
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 1.0


def band_energy(x, sr):
    f, p = sps.welch(x, sr, nperseg=16384)
    tot = np.sum(p[(f >= 20) & (f < 16000)])
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(100.0 * np.sum(p[m]) / max(tot, 1e-18))
    return out


def structure_for(path):
    """如果是《余烬》，直接把段落表拿过来；否则不分段。

    注意别在这里 import song.py —— 那个文件的编曲是模块级的，一 import
    就把整首《潮汐》渲染一遍（好几分钟）。
    """
    if os.path.basename(path).startswith("embers"):
        try:
            import embers
            return embers.STRUCTURE, embers.BAR
        except Exception as exc:  # pragma: no cover
            print(f"(拿不到段落表: {exc})")
    return None, None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "embers.wav"
    png = None
    if "--png" in sys.argv:
        png = sys.argv[sys.argv.index("--png") + 1]
    else:
        png = os.path.splitext(path)[0] + "_analysis.png"

    sr, x = load(path)
    n = x.shape[1]
    mono = (x[0] + x[1]) / 2
    print("=" * 66)
    print(f"文件      {path}")
    print(f"时长      {n / sr:.2f} 秒   {sr} Hz   2 声道")
    print(f"峰值      {np.max(np.abs(x)):.4f}")
    print(f"RMS       {rms(x):.4f}   ({db(rms(x)):+.1f} dBFS)")
    print(f"波峰因数  {crest(x):.1f} dB        （目标 12~16）")
    print(f"削顶      {int(np.sum(np.abs(x) > 0.999))}    NaN {int(np.sum(~np.isfinite(x)))}")
    print(f"左右相关  {correlation(x[0], x[1]):.3f}          （目标 0.6~0.8）")

    # 中高频相关度：低频天然单声道，会把整首的相关度拉高
    hi = np.stack([sps.lfilter(*sps.butter(2, 300 / (sr / 2), "high"), c)
                   for c in x])
    print(f"300Hz 以上相关 {correlation(hi[0], hi[1]):.3f}")

    be = band_energy(mono, sr)
    print("\n频段能量（整首）")
    for nm, v in zip(BAND_NAMES, be):
        bar = "#" * int(round(v / 2))
        print(f"  {nm:>8}  {v:5.1f}%  {bar}")

    # 高潮段的频段能量
    idx = int(np.argmax([rms(mono[i:i + sr]) for i in range(0, max(n - sr, 1), sr // 2)]))
    seg = mono[idx * (sr // 2): idx * (sr // 2) + sr * 20]
    if len(seg) > sr:
        beh = band_energy(seg, sr)
        print("\n最响的 20 秒（" + f"{idx * 0.5:.0f}s 起）")
        for nm, v in zip(BAND_NAMES, beh):
            print(f"  {nm:>8}  {v:5.1f}%")

    struct, bar = structure_for(path)
    sections = None
    vals = []
    if struct:
        pos, acc = [], 0
        for nm, bars in struct:
            pos.append((nm, acc, acc + bars))
            acc += bars
        pos.append(("尾巴", acc, n / sr / bar))
        sections = pos
        print("\n逐段（RMS 相对最响的段落）")
        for nm, b0, b1 in pos:
            i0, i1 = int(b0 * bar * sr), min(int(b1 * bar * sr), n)
            if i1 <= i0:
                continue
            seg = x[:, i0:i1]
            vals.append((nm, rms(seg), crest(seg), i0 / sr, i1 / sr))
        loud = max(v[1] for v in vals)
        print(f"{'段落':<10}{'起':>7}{'止':>8}{'RMS':>9}{'相对':>9}{'波峰':>8}")
        for nm, r, c, t0, t1 in vals:
            print(f"{nm:<10}{t0:>7.1f}{t1:>8.1f}{r:>9.4f}{db(r, loud):>8.1f}dB{c:>7.1f}dB")

    # ---------------- 画图 ----------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.2)

    ax = fig.add_subplot(gs[0, :])
    t = np.arange(n) / sr
    step = max(1, n // 20000)
    ax.plot(t[::step], mono[::step], lw=0.5, color="#333")
    if sections:
        for nm, b0, b1 in sections:
            if b0 > 0:
                ax.axvline(b0 * bar, color="#c33", lw=0.8, alpha=0.6)
            # 图里用罗马数字 / ASCII，免得中文字体缺失时全是方框
            tag = nm.split(".")[0] if "." in nm else "tail"
            ax.text((b0 + b1) / 2 * bar, 1.02, tag, ha="center",
                    va="bottom", fontsize=8, color="#c33")
    ax.set_title(f"{os.path.basename(path)}  waveform   {n / sr:.1f}s  "
                 f"crest {crest(x):.1f} dB  corr {correlation(x[0], x[1]):.2f}")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, n / sr)

    ax = fig.add_subplot(gs[1, 0])
    if sections:
        # 尾巴几乎是静音，画进去只会把柱子压成一根针，去掉
        shown = [v for v in vals if v[1] > 1e-4]
        names = [v[0].split(".")[0] if "." in v[0] else "tail" for v in shown]
        rs = [db(v[1], loud) for v in shown]
        ax.bar(range(len(rs)), rs, color="#4682b4")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("dB (rel. loudest)")
        ax.set_title("per-section RMS")
    else:
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(range(len(be)), be, color="#b4776a")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(BAND_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of power")
    ax.set_title("band energy (whole track)")

    ax = fig.add_subplot(gs[2, :])
    f, p = sps.welch(mono, sr, nperseg=16384)
    ax.semilogx(f[1:], 10 * np.log10(p[1:] + 1e-18), lw=0.8, color="#2a6")
    ax.set_xlim(20, 20000)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB/Hz")
    ax.set_title("average spectrum")
    ax.grid(alpha=0.3)

    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"\n图已写出 {png}")


if __name__ == "__main__":
    main()
