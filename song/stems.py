"""把《余烬》导出成分轨，给 visualize.py 画波形监视器用

    cd song && python stems.py            # -> stems/*.wav（13 条）

分轨是**母带之前**的干声：没有混响总线、没有总线压缩和削峰，所以 13 条相加
不等于 embers.wav（少了混响和延迟的湿声，电平也没走母带）。画波形监视器正好——
哪一轨在响、哪一拍进了什么，一眼就能看出来。

每条轨单独归一化到峰值 0.95：调音台的绝对电平（-30 ~ -10 dB）在波形图上
根本看不见，而且可视化要的是「有没有在响」而不是「响了多响」。

导出走 embers.build_parts()，和成品是同一次编曲、同一个随机数顺序，
所以两边严格对得上（engine 的 seed 固定）。整首重算一遍约 2.5 分钟。
"""

from __future__ import annotations

import os
import time

import numpy as np

import embers as M
import engine as E

OUT_DIR = "stems"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    print("构造六段（和成品同一条路径）…")
    parts = M.build_parts()
    print("  完成 %.0f 秒" % (time.time() - t0))
    print("  各段累计声部：" + " | ".join(
        "%s %d" % (p.name.split(".")[0], len(p.stems)) for p in parts))

    # 尾巴和 mixdown 用的一样长，这样分轨和 embers.wav 等长、对得上时间轴
    t0 = time.time()
    groups = E.collect_group_stems(parts, M.STEM_GROUPS, tail_bars=5.0)
    print("合并成 %d 组，用时 %.0f 秒" % (len(groups), time.time() - t0))

    n = next(iter(groups.values())).shape[1]
    print(f"总长 {n / E.SR:.1f} 秒\n")
    print(f"{'分轨':<12}{'峰值(原始)':>12}{'RMS(原始)':>12}{'归一化后峰值':>14}")

    for name in M.STEM_ORDER:
        y = groups[name]
        raw_peak = E.peak(y)
        raw_rms = E.rms(y)
        # 逐轨归一化：低了看不见，高了会画出格子
        y = E.normalize(y, 0.95) if raw_peak > 1e-9 else y
        path = os.path.join(OUT_DIR, name.replace(" ", "_") + ".wav")
        E.write_wav_mono(path, y, E.SR)
        print(f"{name:<12}{raw_peak:>12.5f}{raw_rms:>12.5f}{E.peak(y):>14.3f}")

    # 自检：分轨之和应该和成品的干声高度相关（差的只有混响/延迟和母带）
    print("\n自检 1：分轨之和 vs embers.wav")
    tot = sum(groups[k] for k in M.STEM_ORDER)
    tot = (tot[0] + tot[1]) / 2
    try:
        from scipy.io import wavfile
        sr, d = wavfile.read("embers.wav")
        wav = d.astype(np.float64) / 32768.0
        wav = (wav[:, 0] + wav[:, 1]) / 2
        m = min(len(tot), len(wav))
        a, b = tot[:m], wav[:m]
        cc = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
        print(f"  相关系数 {cc:.3f}（干声和成品不会完全一样：成品多了混响、"
              f"延迟、母带）")
    except Exception as exc:
        print("  跳过（%s）" % exc)

    # 自检 2：逐段逐轨的静音占比。
    # 混合之后看不出「某一条轨在应该响的地方断了」——低音掉 0.8 秒，频谱上
    # 只是低频少一点。这张表是抓那类 bug 的：预期该连续的轨（pad / sub /
    # acid / 16 分钉钉）静音占比应该很低，突然出现一个高值就是断点。
    print("\n自检 2：逐段静音窗口占比（50ms 窗口，RMS < 0.02 算静音）")
    print("  " + " " * 8 + "".join("%9s" % n[:8] for n in M.STEM_ORDER))
    bar, sr = M.BAR, E.SR
    pos = 0
    for name, bars in M.STRUCTURE:
        i0, i1 = int(pos * bar * sr), int((pos + bars) * bar * sr)
        pos += bars
        row = "  %-8s" % name.split(".")[0]
        for nm in M.STEM_ORDER:
            x = groups[nm][:, i0:i1]
            x = (x[0] + x[1]) / 2
            w = int(0.05 * sr)
            if len(x) <= w:
                row += "%8s " % "-"
                continue
            env = np.sqrt(np.convolve(x * x, np.ones(w) / w, "valid"))[::w]
            row += "%8.0f%%" % (100 * float(np.mean(env < 0.02)))
        print(row)
    print("  预期该连续的轨（Pad / Sub Bass / Acid B.（16分）/ Hats）出现高占比"
          " = 有断点")

    print(f"\n已写出 {len(M.STEM_ORDER)} 条分轨到 {OUT_DIR}/，"
          f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
