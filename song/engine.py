"""《余烬》合成引擎 —— 全向量化的立体声合成器 + 混音台

跟同目录下 song.py 里那套老引擎比，这里换掉的东西：

  * **真立体声**：一次渲染出两声道。老引擎是 SIDE=L / SIDE=R 跑两遍再拼，
    两遍之间只有噪声和混响失谐去相关，本质上是「单声道 + 假的宽度」。
  * **没有逐采样 Python 循环**：老的 sawtooth / square / triangle / distortion /
    limiter / slide 全是 `for`，这里全部换成 numpy。整首曲子（约 3 分半）
    渲染时间从「几分钟 × 2」降到几十秒，才敢写这么长的编制。
  * **note() 是准的**：标准 A4 = 440，写什么音就是什么音。老引擎硬编码了
    +3 个半音（"D#调"），害得《潮汐》写着 E 小调、听着是 G 小调。
  * **相位用 arange 累加**（`TAU*f*n/SR`），天生整周期，不需要 declick 擦咔哒。
  * **滤波器是真的双二阶**（RBJ，带共振 Q），不是老引擎那种「FFT 把频点乘 0」
    的砖墙硬切。能扫、能共振、能做 303。
  * **混响走 send bus**：整首只卷积一次 IR.wav（2.61 秒真立体声脉冲响应），
    不是每条轨各卷一遍。
  * **确定性**：所有随机数来自 seed 固定的 `default_rng`，同一份代码重渲
    结果逐采样一致（老引擎每次都不一样，没法比对）。改 `SEED` 环境变量即可换一版。

信号约定：单声道是 1D 数组，立体声是 (2, N)。所有内部计算 float64，
只在写盘时转 int16。
"""

from __future__ import annotations

import os
import re
import wave

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

# ---------------------------------------------------------------------------
#  全局
# ---------------------------------------------------------------------------

SR = 44100
TAU = 2 * np.pi

BPM = 124.0
BEAT = 60.0 / BPM
BAR = BEAT * 4


def set_tempo(bpm: float):
    """改速度。BEAT / BAR 是模块级全局，编曲层要拿返回值用。"""
    global BPM, BEAT, BAR
    BPM = float(bpm)
    BEAT = 60.0 / BPM
    BAR = BEAT * 4
    return BEAT, BAR


SEED = int(os.environ.get("SEED", "20250902"))
_rng = np.random.default_rng(SEED)


def reseed(seed: int | None = None):
    """重设随机种子。同一 seed 下整首曲子逐采样可复现。"""
    global _rng, SEED
    if seed is not None:
        SEED = int(seed)
    _rng = np.random.default_rng(SEED)
    return _rng


def rng():
    return _rng


# ---------------------------------------------------------------------------
#  音高：标准十二平均律，A4 = 440
# ---------------------------------------------------------------------------

_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
_RE_NOTE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def note(name: str) -> float:
    """音名 -> 频率。'A4'=440，'D3'=146.83，'Bb2'=116.54。

    规则：音名 + 可选升降号 + 八度（可以多位数）。C4 是中央 C。
    """
    m = _RE_NOTE.match(name.strip())
    if not m:
        raise ValueError(f"看不懂的音名: {name!r}")
    letter, acc, octv = m.groups()
    pc = _PC[letter.upper() + acc]
    return 440.0 * 2.0 ** ((pc + (int(octv) - 4) * 12 - 9) / 12.0)


def chord(names) -> list[float]:
    """['D3','F3','A3'] -> [146.83, 174.61, 220.0]"""
    if isinstance(names, str):
        names = names.split()
    return [note(x) for x in names]


# ---------------------------------------------------------------------------
#  小工具
# ---------------------------------------------------------------------------


def nsamp(dur: float) -> int:
    return int(round(dur * SR))


def taxis(n: int) -> np.ndarray:
    return np.arange(n) / SR


def db2lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def lin2db(x) -> float:
    return 20.0 * np.log10(max(float(np.max(np.abs(x))), 1e-12))


def rms(x) -> float:
    a = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(a ** 2)))


def peak(x) -> float:
    return float(np.max(np.abs(np.asarray(x, dtype=np.float64))))


def crest_db(x) -> float:
    return 20.0 * np.log10(max(peak(x), 1e-12) / max(rms(x), 1e-12))


def fit(x, n: int):
    """把数组末尾补零/截断到 n（最后一个轴）。"""
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[-1]
    if m == n:
        return x
    if m > n:
        return x[..., :n]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n - m)]
    return np.pad(x, pad)


def normalize(x, target: float = 1.0):
    p = peak(x)
    if p < 1e-12:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) / p * target


def balance(x, pan: float = 0.0):
    """立体声源的声像（balance）：只改左右比例，不动内部的宽度。"""
    if abs(pan) < 1e-6:
        return np.asarray(x, dtype=np.float64)
    th = (float(np.clip(pan, -1, 1)) + 1.0) * np.pi / 4.0
    gl = np.cos(th) * np.sqrt(2.0)
    gr = np.sin(th) * np.sqrt(2.0)
    x = np.asarray(x, dtype=np.float64)
    return np.stack([x[0] * gl, x[1] * gr])


def to_stereo(x, pan: float = 0.0, n: int | None = None):
    """单声道 -> 等功率声像；已经是立体声就走 balance（可选补齐长度）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        th = (float(np.clip(pan, -1, 1)) + 1.0) * np.pi / 4.0
        y = np.stack([x * np.cos(th), x * np.sin(th)])
    else:
        y = balance(x, pan)
    if n is not None:
        y = fit(y, n)
    return y


def fade(x, in_s: float = 0.004, out_s: float = 0.008):
    """在首尾加很小的淡入淡出，防拼接爆音。已经过 declick 的信号不需要，
    但段落拼接处用一下最省心。"""
    x = np.array(x, dtype=np.float64, copy=True)
    ni, no = nsamp(in_s), nsamp(out_s)
    if ni > 1:
        ramp = np.linspace(0, 1, ni) ** 1.5
        x[..., :ni] *= ramp
    if no > 1:
        ramp = np.linspace(1, 0, no) ** 1.5
        x[..., -no:] *= ramp
    return x


def mid_side(x):
    """(2,N) -> (mid, side)。"""
    return (x[0] + x[1]) / 2.0, (x[0] - x[1]) / 2.0


def correlation(x) -> float:
    a, b = x[0], x[1]
    d = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / d) if d > 0 else 1.0


# ---------------------------------------------------------------------------
#  包络
# ---------------------------------------------------------------------------


def env_pts(n: int, pts, sr: int = SR):
    """折线包络。pts = [(秒, 电平), ...]，两端保持端点值。"""
    t = np.arange(n) / sr
    ts = np.array([p[0] for p in pts], dtype=np.float64)
    vs = np.array([p[1] for p in pts], dtype=np.float64)
    return np.interp(t, ts, vs)


def adsr(n: int, a=0.01, d=0.12, s=0.7, r=0.2, sr: int = SR):
    """标准 ADSR。r 太长时会自动缩到音符长度以内。"""
    dur = n / sr
    r = min(r, dur * 0.6)
    sus = max(s, 1e-4)
    pts = [(0.0, 0.0), (min(a, dur * 0.5), 1.0)]
    ad = min(a + d, dur - r)
    pts.append((ad, sus))
    if dur - r > ad:
        pts.append((dur - r, sus))
    pts.append((dur, 0.0))
    return env_pts(n, pts, sr)


def exp_env(n: int, tau: float, sr: int = SR, start: float = 1.0):
    """指数衰减。tau 是时间常数（秒）。"""
    return start * np.exp(-np.arange(n) / (tau * sr))


def perc_env(n: int, attack: float = 0.002, tau: float = 0.25, sr: int = SR):
    """打击乐包络：极快起音 + 指数衰减。"""
    t = np.arange(n) / sr
    a = np.clip(t / attack, 0.0, 1.0) if attack > 1e-6 else np.ones(n)
    return a * np.exp(-t / tau)


def gate_env(n: int, attack: float = 0.004, release: float = 0.02, sr: int = SR):
    """门包络：起音 -> 满 -> 收尾。用于 acid / stab 这类要干脆断开的音色。"""
    dur = n / sr
    r = min(release, dur * 0.4)
    a = min(attack, dur * 0.3)
    return env_pts(n, [(0.0, 0.0), (a, 1.0), (dur - r, 1.0), (dur, 0.0)], sr)


def swell(n: int, curve: float = 1.0, sr: int = SR):
    """两头归零的隆起，当 riser / 渐强用（numpy，不是逐采样循环）。"""
    return np.sin(np.linspace(0, np.pi, n)) ** curve


# ---------------------------------------------------------------------------
#  振荡器（全部向量化；相位是弧度，2π 一个周期）
# ---------------------------------------------------------------------------


def phase(freq, dur: float, phase0: float = 0.0):
    """相位数组。freq 可以是标量，也可以是逐采样数组（滑音 / 音高包络）。"""
    n = nsamp(dur)
    if np.isscalar(freq) or (isinstance(freq, np.ndarray) and freq.ndim == 0):
        return phase0 + TAU * float(freq) * np.arange(n) / SR
    f = np.asarray(freq, dtype=np.float64)
    if len(f) != n:
        f = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f)), f)
    return phase0 + TAU * np.cumsum(f) / SR


def sine(ph):
    return np.sin(ph)


def saw_naive(ph):
    """朴素锯齿，2*(frac-0.5)。混叠是音色的一部分，只在低频用。"""
    fr = ph / TAU
    return 2.0 * (fr - np.floor(fr)) - 1.0


def square_naive(ph, duty: float = 0.5):
    fr = ph / TAU
    return np.where((fr - np.floor(fr)) < duty, 1.0, -1.0)


def tri_naive(ph):
    fr = ph / TAU
    x = fr - np.floor(fr)
    return 4.0 * np.abs(x - 0.5) - 1.0


def _hmax_for(freq: float, cap: int) -> int:
    if freq and freq > 0:
        return max(1, min(cap, int(SR * 0.45 / freq)))
    return cap


def bl_saw(ph, freq: float = 0.0, cap: int = 32):
    """加法带限锯齿：sum sin(k·ph)/k。天然不混叠，代价是谐波数有限。

    低频音色（pad / bass）用 32 个谐波足够亮；高频（主音）会被 SR/2
    自动砍掉谐波数，所以不会混叠。
    """
    h = _hmax_for(freq, cap)
    a = np.zeros_like(ph)
    for k in range(1, h + 1):
        a += np.sin(ph * k) / k
    return a * (2.0 / np.pi)


def bl_square(ph, freq: float = 0.0, cap: int = 31):
    """加法带限方波：只叠奇次谐波。"""
    h = _hmax_for(freq, cap)
    a = np.zeros_like(ph)
    for k in range(1, h + 1, 2):
        a += np.sin(ph * k) / k
    return a * (4.0 / np.pi)


def bl_pulse(ph, duty: float = 0.35, freq: float = 0.0, cap: int = 24):
    """带限脉冲波（可调占空比）。用来做中空的簧管 / 电钢音色。"""
    h = _hmax_for(freq, cap)
    a = np.zeros_like(ph)
    for k in range(1, h + 1):
        a += np.sin(ph * k) * (2.0 / (k * np.pi)) * np.sin(np.pi * k * duty)
    return a


def white(n: int):
    return _rng.random(n) * 2.0 - 1.0


def pink(n: int):
    """粉噪声：白噪声过一个 -3dB/oct 的近似滤波器（Voss 级联的频域做法）。"""
    w = white(n)
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return normalize(y)


# ---------------------------------------------------------------------------
#  滤波器：RBJ 双二阶，支持逐块时变截止频率（扫频 / 包络）
# ---------------------------------------------------------------------------

_COEF_CACHE: dict = {}
_LOG_LO, _LOG_HI, _LOG_N = np.log(20.0), np.log(20000.0), 512


def _quant(f0: float) -> float:
    """把截止频率量化到对数网格上，好让系数缓存不再爆炸。"""
    f0 = float(np.clip(f0, 20.0, 20000.0))
    k = round((np.log(f0) - _LOG_LO) / (_LOG_HI - _LOG_LO) * _LOG_N)
    return float(np.exp(_LOG_LO + k * (_LOG_HI - _LOG_LO) / _LOG_N))


def _coefs(kind: str, f0: float, q: float):
    key = (kind, round(f0, 2), round(q, 3))
    c = _COEF_CACHE.get(key)
    if c is not None:
        return c
    w0 = TAU * min(f0, SR * 0.45) / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(q, 0.05))
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        A = 10 ** (q / 40.0)  # 这里 q 复用为增益 dB
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    c = (b, a)
    _COEF_CACHE[key] = c
    return c


def _biquad(x, kind, cutoff, q, block):
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(cutoff, dtype=np.float64) if not np.isscalar(cutoff) else None
    y = np.empty_like(x)
    if c is None or c.ndim == 0:
        b, a = _coefs(kind, _quant(float(cutoff)), q)
        return sps.lfilter(b, a, x)
    zi = np.zeros(2)
    for i in range(0, len(x), block):
        seg = x[i:i + block]
        b, a = _coefs(kind, _quant(c[i]), q)
        out, zi = sps.lfilter(b, a, seg, zi=zi)
        y[i:i + block] = out
    return y


def lpf(x, cutoff, q: float = 0.707, block: int = 96):
    """二阶低通。cutoff 标量 = 静态滤波（快），数组 = 逐块时变（扫频）。"""
    return _biquad(x, "lp", cutoff, q, block)


def hpf(x, cutoff, q: float = 0.707, block: int = 96):
    return _biquad(x, "hp", cutoff, q, block)


def bpf(x, cutoff, q: float = 1.0, block: int = 96):
    return _biquad(x, "bp", cutoff, q, block)


def high_shelf(x, f0: float, gain_db: float, q: float = 0.707):
    """RBJ 高架滤波：f0 以上整体抬/降 gain_db。母带的「空气感」用这个。"""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TAU * min(f0, SR * 0.45) / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
         -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
         2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return sps.lfilter(b, a, np.asarray(x, dtype=np.float64))


def low_shelf(x, f0: float, gain_db: float, q: float = 0.707):
    """RBJ 低架滤波。"""
    A = 10.0 ** (gain_db / 40.0)
    w0 = TAU * min(f0, SR * 0.45) / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    b = [A * ((A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha),
         2 * A * ((A - 1) - (A + 1) * cw),
         A * ((A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha,
         -2 * ((A - 1) + (A + 1) * cw),
         (A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return sps.lfilter(b, a, np.asarray(x, dtype=np.float64))


def hpf_stereo(x, cutoff, q: float = 0.707):
    return np.stack([hpf(x[0], cutoff, q), hpf(x[1], cutoff, q)])


def lpf_stereo(x, cutoff, q: float = 0.707):
    return np.stack([lpf(x[0], cutoff, q), lpf(x[1], cutoff, q)])


# ---------------------------------------------------------------------------
#  失真 / 整形
# ---------------------------------------------------------------------------


def softclip(x, drive: float = 1.0, bias: float = 0.0):
    """tanh 软削波。drive 越大越脏；bias 加一点非对称（偶次谐波，更「暖」）。"""
    return np.tanh((np.asarray(x, dtype=np.float64) + bias) * drive)


def wavefold(x, drive: float = 1.0):
    """波折叠，比削波更「金属」。"""
    y = np.asarray(x, dtype=np.float64) * drive
    return np.sin(y * np.pi / 2)


def bitcrush(x, bits: int = 8):
    q = 2 ** (bits - 1)
    return np.round(np.asarray(x, dtype=np.float64) * q) / q


def exciter(x, freq: float = 3000.0, amount: float = 0.3):
    """高频激励：把高频单独削一下再叠回去，让音色「亮」而不只是「响」。"""
    hi = hpf(x, freq, 0.707)
    return x + softclip(hi * 3.0, 1.0) * amount


# ---------------------------------------------------------------------------
#  动态 / 空间
# ---------------------------------------------------------------------------


def duck_env(bars: float, hits_per_bar: float = 4.0, depth: float = 0.55,
             release: float = 0.13, offset: float = 0.0):
    """侧链包络：每个 kick 位置瞬间掉到 (1-depth)，再指数回升。

    整段一次性算好，直接 `track * duck_env(...)`，没有逐采样循环。
    """
    n = nsamp(bars * BAR)
    env = np.ones(n)
    step = BAR / hits_per_bar
    m = min(n, nsamp(release * 4.0))
    shape_t = np.arange(m) / SR
    shape = 1.0 - depth * np.exp(-shape_t / release)
    t = offset
    while t < bars * BAR:
        i0 = int(t * SR)
        if i0 >= n:
            break
        seg = min(m, n - i0)
        env[i0:i0 + seg] = np.minimum(env[i0:i0 + seg], shape[:seg])
        t += step
    return env


def compress(x, thresh_db: float = -18.0, ratio: float = 3.0,
             attack: float = 0.008, release: float = 0.12, makeup_db: float = 0.0):
    """前馈压缩器（单极点检波 + 软拐点）。够用、不炸、不做怪声。"""
    x = np.asarray(x, dtype=np.float64)
    env = np.abs(x)
    # 检波用固定的单极点（attack/release 取几何平均，避免分支递归）
    tau = float(np.sqrt(max(attack, 1e-4) * max(release, 1e-4)))
    a = np.exp(-1.0 / (tau * SR))
    env = sps.lfilter([1 - a], [1.0, -a], env)
    over = 20.0 * np.log10(np.maximum(env, 1e-12)) - thresh_db
    gain_db = np.where(over > 0, -over * (1.0 - 1.0 / ratio), 0.0)
    return x * 10.0 ** ((gain_db + makeup_db) / 20.0)


def limiter(x, ceiling: float = 0.97, release: float = 0.05):
    """前馈峰值限制器：增益平滑 + 硬保证不超过 ceiling。

    最后那一步 `min(g, ceiling/|x|)` 是关键——平滑会让增益跟不上瞬态，
    补这一下才能保证真不削顶。
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.exp(-1.0 / (release * SR))
    env = sps.lfilter([1 - a], [1.0, -a], np.abs(x))
    g = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    mg = np.minimum(g, ceiling / np.maximum(np.abs(x), 1e-9))
    return x * np.minimum(mg, 1.0)


def pingpong(x, delay_s: float, feedback: float = 0.45, mix: float = 0.3,
             damp: float = 5000.0, taps: int = 8):
    """乒乓延迟（立体声）。第 k 个回声在左右之间来回跳，并且越来越暗。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    d = nsamp(delay_s)
    wet = np.zeros_like(x)
    cur = x
    for k in range(taps):
        cur = np.concatenate([np.zeros((2, d)), cur[:, :n - d]], axis=1) * feedback
        cur = lpf_stereo(cur, damp)
        # 交替左右
        if k % 2 == 0:
            wet[0] += cur[1] * (0.9 ** k)
            wet[1] += cur[0] * (0.9 ** k)
        else:
            wet[0] += cur[0] * (0.9 ** k)
            wet[1] += cur[1] * (0.9 ** k)
    return x + wet * mix


def chorus(x, rate_hz: float = 0.35, depth_ms: float = 9.0, mix: float = 0.35,
           voices: int = 3, spread: float = 1.0):
    """立体声合唱：几条被 LFO 调制延迟的副本，左右用不同相位。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, x])
    n = x.shape[-1]
    t = taxis(n)
    out = np.zeros_like(x)
    for v in range(voices):
        ph = TAU * (v / voices)
        lfo_l = np.sin(TAU * rate_hz * t + ph)
        lfo_r = np.sin(TAU * rate_hz * t + ph + np.pi / 2)
        d_l = int(round(depth_ms * 1e-3 * SR))
        for ch, lfo, dly in ((0, lfo_l, d_l), (1, lfo_r, d_l)):
            mod = dly + (lfo * dly * 0.5).astype(int)
            mod = np.clip(mod, 1, dly * 2)
            idx = np.arange(n) - mod
            idx = np.clip(idx, 0, n - 1)
            out[ch] += x[ch][idx]
    out /= voices
    # 左右错开几十个采样再加宽（用补零平移，不能用 roll——roll 会把尾巴绕到
    # 开头，接缝处一个爆音）
    sh = int(spread * 37)
    if sh:
        out[1] = np.concatenate([np.zeros(sh), out[1][:n - sh]])
    return x * (1 - mix) + out * mix


def width(x, w: float = 1.0):
    """中/侧宽度。w<1 变窄，w>1 变宽（1.0 原样）。"""
    m, s = mid_side(np.asarray(x, dtype=np.float64))
    return np.stack([m + s * w, m - s * w])


def widen_lowless(x, w: float = 1.3, keep_below: float = 200.0):
    """只把中高频变宽，低频保持单声道（避免低频相位问题）。"""
    x = np.asarray(x, dtype=np.float64)
    lo = lpf_stereo(x, keep_below)
    hi = x - lo
    return lo + width(hi, w)


# ---------------------------------------------------------------------------
#  卷积混响（send bus 用）
# ---------------------------------------------------------------------------

_IR_CACHE: dict = {}


def load_ir(path: str = "IR.wav", length_s: float = 2.4, hpf_hz: float = 130.0,
            fade_s: float = 0.15):
    """读 IR.wav（32bit float 立体声）-> 归一化脉冲响应。

    截到 length_s、末尾淡出（免得尾巴被硬切）、高通掉低频（免得混响在
    低频堆成一坨）。老引擎的 reverb 会把湿信号单独 maximize 到 1.0，
    安静音色会被淹掉；这里湿/干比例由编曲层控制，不存在这个问题。
    """
    key = (path, length_s, hpf_hz)
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    sr, d = wavfile.read(path)
    if d.ndim == 1:
        d = np.stack([d, d], axis=1)
    d = np.asarray(d, dtype=np.float64)
    d = d[: min(len(d), nsamp(length_s))]
    d = d / max(peak(d), 1e-9)
    nf = nsamp(fade_s)
    if nf > 1:
        d[-nf:] *= np.linspace(1, 0, nf)[:, None]
    for ch in range(d.shape[1]):
        d[:, ch] = hpf(d[:, ch], hpf_hz)
    # 按能量归一：白噪过一遍，输出 RMS ≈ 输入 RMS。这样混响的「水位」
    # 就完全由编曲层的送出量和 rev_amt 决定，换个 IR 也不用重调。
    for ch in range(d.shape[1]):
        e = np.sqrt(np.sum(d[:, ch] ** 2))
        d[:, ch] /= max(e, 1e-12)
    _IR_CACHE[key] = d
    return d


def convolve(x, h):
    """重叠相加的分块卷积，内存友好（整首 3 分钟 × 2.4 秒 IR 也不会爆）。

    x 可以是 1D 或 (2,N)；h 是 (M,) 或 (2,M)。返回长度 len(x)+len(h)-1。
    FFT 长度取 IR 长度的两倍以上的 2 的幂，块长必须大于 IR。
    """
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    mono_x = x.ndim == 1
    mono_h = h.ndim == 1
    if mono_x:
        x = np.stack([x, x])
    if mono_h:
        h = np.stack([h, h])
    out = np.zeros((x.shape[0], x.shape[1] + h.shape[1] - 1))
    nfft = 1 << int(np.ceil(np.log2(max(2 * h.shape[1], 1 << 16))))
    step = nfft - h.shape[1] + 1
    for ch in range(x.shape[0]):
        ir = h[ch % h.shape[0]]
        H = np.fft.rfft(ir, nfft)
        for i in range(0, x.shape[1], step):
            seg = x[ch, i:i + step]
            if len(seg) == 0:
                break
            y = np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)
            sl = out[ch, i:i + nfft]
            sl += y[:len(sl)]
    return out[0] if mono_x else out


# ---------------------------------------------------------------------------
#  乐器
# ---------------------------------------------------------------------------
#  约定：单声道乐器返回 1D，天生立体声的返回 (2,N)。电平大致落在
#  峰值 0.3~1.0，编曲层用 dB 增益去摆位；同一音色不同音符之间的强弱
#  是保留的（不做逐音符归一化）。
# ---------------------------------------------------------------------------


def _stack(freqs) -> list[float]:
    if isinstance(freqs, (int, float)):
        return [float(freqs)]
    return [float(f) for f in np.atleast_1d(freqs)]


def pad(freqs, dur: float, bright: float = 1400.0, voices: int = 5,
        detune: float = 0.008, a: float = 0.7, r: float = 0.9,
        drift: float = 0.0, sub_oct: float = 0.0, spread: float = 0.8):
    """温暖模拟铺底（立体声）：每音 5 路失谐带限锯齿 + 低通 + 缓慢起落。

    失谐的每一路按失谐量摆到声场里，和弦里不同的音再各自偏一点，
    所以铺底天生是「一片」而不是「一个点」——这是这套编曲里宽度的主要来源。
    `drift` 给低通加一个很慢的 LFO，让长和弦呼吸。
    `sub_oct` 是低八度正弦的**电平**（0 就是不加）。铺底的低八度很容易和
    sub / 底鼓在 40-100 Hz 撞车，安静段落给 0.2 就够，别给满。
    谐波数按低通截止频率反推：超过截止 2.5 倍频的谐波反正会被滤掉。
    """
    fs = _stack(freqs)
    n = nsamp(dur)
    L = np.zeros(n)
    R = np.zeros(n)
    for i, f in enumerate(fs):
        cap = int(np.clip(bright * 2.5 / max(f, 20.0) + 2, 4, 32))
        off = ((i % 3) - 1) * 0.28  # 和弦里各音错开摆位
        for v in range(voices):
            d = (v - (voices - 1) / 2) / max((voices - 1) / 2, 1)
            fv = f * (1.0 + d * detune)
            ph = phase(fv, dur, _rng.random() * TAU)
            sig = bl_saw(ph, fv, cap) * (1.0 - 0.12 * abs(d))
            if sub_oct and v == 0:
                sig = sig + np.sin(phase(fv / 2, dur)) * float(sub_oct)
            p = float(np.clip(d * spread + off, -1, 1))
            th = (p + 1.0) * np.pi / 4.0
            L += sig * np.cos(th)
            R += sig * np.sin(th)
    out = np.stack([L, R]) / (max(len(fs), 1) ** 0.5 * voices ** 0.5)
    cut = bright
    if drift > 0:
        t = taxis(n)
        cut = bright * (1.0 + 0.45 * drift * np.sin(TAU * 0.06 * t + 0.4))
    out = lpf_stereo(out, cut, q=0.9)
    out = hpf_stereo(out, 60.0)
    out *= adsr(n, a=a, d=max(a * 1.2, 0.2), s=0.85, r=min(r, dur * 0.5))
    return out * 0.75


def choir(freqs, dur: float, vowel: str = "ah", bright: float = 2600.0,
          a: float = 0.9, r: float = 1.1, vib: float = 0.004):
    """人声化的铺底：锯齿 + 脉冲做声源，三个共振峰做「啊 / 呜」的元音。

    共振峰频率是标准元音数据：ah = 730/1090/2440，ooh = 300/870/2240，
    eh = 530/1840/2480。这是这套引擎里最「不合成器」的一个音色。
    """
    FORMANT = {
        "ah": [(730, 1.0, 9), (1090, 0.55, 11), (2440, 0.28, 13)],
        "ooh": [(300, 1.0, 8), (870, 0.45, 10), (2240, 0.15, 12)],
        "eh": [(530, 1.0, 9), (1840, 0.5, 11), (2480, 0.3, 12)],
    }[vowel]
    fs = _stack(freqs)
    n = nsamp(dur)
    t = taxis(n)
    L = np.zeros(n)
    R = np.zeros(n)
    for i, f in enumerate(fs):
        for j, d in enumerate((-0.006, 0.0, 0.006)):
            fv = f * (1.0 + d) * (1.0 + vib * np.sin(TAU * 5.1 * t + _rng.random() * TAU))
            ph = phase(fv, dur, _rng.random() * TAU)
            sig = bl_saw(ph, f, 24) * 0.7 + bl_pulse(ph, 0.42, f, 20) * 0.3
            p = float(np.clip((j - 1) * 0.5 + ((i % 3) - 1) * 0.22, -1, 1))
            th = (p + 1.0) * np.pi / 4.0
            L += sig * np.cos(th)
            R += sig * np.sin(th)
    out = np.stack([L, R]) / (max(len(fs), 1) ** 0.5 * 3 ** 0.5)
    voiced = np.zeros((2, n))
    for f0, w, q in FORMANT:
        voiced[0] += bpf(out[0], f0, q) * w
        voiced[1] += bpf(out[1], f0, q) * w
    out = lpf_stereo(voiced, bright, 0.9)
    out += lpf_stereo(out, 900, 0.7) * 0.5
    out = hpf_stereo(out, 120)
    out *= adsr(n, a=a, d=0.6, s=0.9, r=min(r, dur * 0.5))
    return out * 3.2


def sub(freq, dur: float, a: float = 0.02, r: float = 0.12, harm: float = 0.12):
    """纯正弦低音 + 一点点二次谐波（小喇叭上才听得见）。"""
    n = nsamp(dur)
    ph = phase(freq, dur)
    out = np.sin(ph) + np.sin(ph * 2) * harm
    out *= adsr(n, a=a, d=0.3, s=0.85, r=min(r, dur * 0.5))
    return out * 0.8


def fm_bell(freq, dur: float, ratio: float = 3.5, index: float = 5.5,
            tau: float = 0.45, amp_tau: float = 2.0, ratio2: float = 7.0,
            index2: float = 1.6, tau2: float = 0.12):
    """两对算子的 FM 钟 / 钢片琴。

    载波 = 基频，调制器 = 基频 × ratio（非整数比 = 非谐波 = 金属感）。
    调制指数快速衰减，于是「叮」的那一下有泛音、延音是纯正弦——
    这就是钟和电钢的差别所在。
    """
    n = nsamp(dur)
    t = taxis(n)
    ph = phase(freq, dur)
    m1 = np.sin(ph * ratio) * index * np.exp(-t / tau)
    m2 = np.sin(ph * ratio2) * index2 * np.exp(-t / tau2)
    out = np.sin(ph + m1 + m2)
    out *= perc_env(n, 0.0015, amp_tau)
    return out * 0.85


def glass(freq, dur: float, tau: float = 0.9, tine: float = 0.05):
    """玻璃拨弦 / 电钢：本体正弦 + 14 倍频的「毛刺」快速衰减。"""
    n = nsamp(dur)
    t = taxis(n)
    ph = phase(freq, dur)
    tine_op = np.sin(ph * 14.0) * 3.2 * np.exp(-t / tine)
    body = np.sin(ph + tine_op) * np.exp(-t / tau)
    body += np.sin(ph * 2.0) * 0.18 * np.exp(-t / (tau * 0.4))
    body += np.sin(ph * 3.0) * 0.08 * np.exp(-t / (tau * 0.25))
    body *= perc_env(n, 0.001, tau)
    return body * 0.8


def acid(freq, dur: float, base: float = 220.0, env_amt: float = 3.2,
         q: float = 8.0, decay: float = 0.16, accent: float = 0.0,
         drive: float = 1.6, slide_from: float | None = None,
         wave: str = "saw", res_glide: float = 0.0):
    """303 味儿的酸性低音：锯齿 + 共振低通 + 快速截止包络。

    `env_amt` 是截止频率包络的八度跨度，`accent` 再往上顶一点（经典的重音），
    `slide_from` 给滑音（前一个音高滑过来）。`res_glide` 让截止频率随音高走，
    高音更亮——这是 303 的另一个标志。
    """
    n = nsamp(dur)
    t = taxis(n)
    f = freq
    if slide_from is not None:
        f = np.linspace(slide_from, freq, n)
    ph = phase(f, dur)
    if wave == "saw":
        raw = bl_saw(ph, freq)
    elif wave == "square":
        raw = bl_square(ph, freq)
    else:
        raw = bl_pulse(ph, 0.3, freq)
    env = np.exp(-t / decay)
    cut = (base + freq * res_glide) * 2.0 ** (env_amt * env * (1.0 + 1.6 * accent))
    cut = np.clip(cut, 40.0, 12000.0)
    out = lpf(raw, cut, q=q)
    out = softclip(out * drive, 1.0)
    out *= gate_env(n, 0.003, 0.015)
    return out * (0.85 + 0.35 * accent)


def supersaw(freq, dur: float, voices: int = 7, detune: float = 0.016,
             bright: float = 6500.0, spread: float = 0.85, a: float = 0.01,
             r: float = 0.12, hp: float = 180.0, q: float = 1.1):
    """超级锯（立体声）：7 路失谐，每路按失谐量摆到不同的声像位置。

    副歌那种「一大片」的宽度就是这么来的——不是加混响，是把失谐的
    每一路真的摆到左右两边去。
    """
    n = nsamp(dur)
    L = np.zeros(n)
    R = np.zeros(n)
    for v in range(voices):
        d = (v - (voices - 1) / 2) / max((voices - 1) / 2, 1)
        fv = freq * (1.0 + d * detune)
        ph = phase(fv, dur, _rng.random() * TAU)
        sig = bl_saw(ph, freq, 26)
        p = d * spread
        th = (p + 1.0) * np.pi / 4.0
        L += sig * np.cos(th)
        R += sig * np.sin(th)
    out = np.stack([L, R]) / (voices ** 0.5)
    out = lpf_stereo(out, bright, q)
    out = hpf_stereo(out, hp)
    out *= adsr(n, a=a, d=0.25, s=0.85, r=min(r, dur * 0.5))
    return out * 0.55


def saw_pluck(freq, dur: float, base: float = 300.0, env_amt: float = 3.0,
              q: float = 3.0, decay: float = 0.12, amp_tau: float = 0.35,
              voices: int = 3, detune: float = 0.01):
    """减法合成拨弦：失谐锯齿 + 快速截止包络。比 FM 拨弦更「合成器」。"""
    n = nsamp(dur)
    t = taxis(n)
    out = np.zeros(n)
    for v in range(voices):
        d = (v - (voices - 1) / 2) / max((voices - 1) / 2, 1)
        fv = freq * (1.0 + d * detune)
        out += bl_saw(phase(fv, dur, _rng.random() * TAU), freq, 28)
    out /= voices ** 0.5
    cut = base * 2.0 ** (env_amt * np.exp(-t / decay))
    out = lpf(out, np.clip(cut, 60, 14000), q=q)
    out *= perc_env(n, 0.002, amp_tau)
    return out * 0.45


def stab(freqs, dur: float, bright: float = 3000.0, q: float = 2.0):
    """和弦短促 stab：锯齿和弦 + 低通 + 门包络。"""
    fs = _stack(freqs)
    n = nsamp(dur)
    out = np.zeros(n)
    for f in fs:
        for d in (-0.004, 0.0, 0.004):
            fv = f * (1 + d)
            out += bl_saw(phase(fv, dur, _rng.random() * TAU), f, 26)
    out /= max(len(fs), 1) ** 0.5 * 3 ** 0.5
    out = lpf(out, bright, q)
    out = hpf(out, 180)
    return out * gate_env(n, 0.004, 0.05) * 0.42


def noise_riser(dur: float, f0: float = 400.0, f1: float = 12000.0,
                q: float = 1.4, tone: bool = True, curve: float = 1.6):
    """上扫的 riser：带通噪声 + 一条跟着上行的正弦 + 上行八度的闪烁。

    截止频率按 `curve` 指数上行，听感是「越来越紧」而不是「越来越亮」。
    """
    n = nsamp(dur)
    t = taxis(n)
    x = np.linspace(0, 1, n) ** curve
    cut = f0 * (f1 / f0) ** x
    nz = white(n)
    out = bpf(nz, cut, q) * 1.2
    if tone:
        f = 220.0 * 2.0 ** (3.0 * x)
        out += np.sin(phase(f, dur)) * 0.35 * x
        out += np.sin(phase(f * 2, dur)) * 0.15 * x ** 2
    out = hpf(out, 180)
    out *= swell(n, 1.4)
    return out * 0.8


def subdrop(f0: float, f1: float, dur: float):
    """下滑正弦，段落交接处用。"""
    n = nsamp(dur)
    f = np.linspace(f0, f1, n)
    out = np.sin(phase(f, dur))
    out *= env_pts(n, [(0, 0), (0.01, 1), (dur * 0.7, 0.7), (dur, 0.0)])
    return out * 0.9


def reverse_swell(dur: float, bright: float = 5000.0):
    """反镲：噪声 + 上行包络，尾巴正好落在下一小节第一拍上。"""
    n = nsamp(dur)
    nz = hpf(white(n), 900)
    nz = lpf(nz, bright, 0.8)
    env = (np.arange(n) / n) ** 2.2
    return nz * env * 0.7


# --- 鼓组 -------------------------------------------------------------------


def kick(freq: float = 49.0, dur: float = 0.55, pitch_amt: float = 4.2,
         pitch_tau: float = 0.028, amp_tau: float = 0.24, click: float = 0.45,
         drive: float = 1.5):
    """底鼓：音高包络正弦 + 高通噪声 click + 软削波。"""
    n = nsamp(dur)
    t = taxis(n)
    f = freq * (1.0 + (pitch_amt - 1.0) * np.exp(-t / pitch_tau))
    body = np.sin(phase(f, dur)) * np.exp(-t / amp_tau)
    body *= np.clip(t / 0.0012, 0, 1)
    cl = hpf(white(n), 1800) * np.exp(-t / 0.0018)
    out = softclip(body * 1.5 + cl * click * 2.0, drive)
    out += np.sin(phase(f, dur)) * np.exp(-t / (amp_tau * 2.4)) * 0.35
    return normalize(out, 0.98)


def snare(freq: float = 210.0, dur: float = 0.4, tone_amt: float = 0.75,
          noise_tau: float = 0.13):
    """军鼓：两个鼓皮音 + 高通噪声。"""
    n = nsamp(dur)
    t = taxis(n)
    tone = (np.sin(phase(freq, dur)) * 0.8
            + np.sin(phase(freq * 1.6, dur)) * 0.6
            + np.sin(phase(freq * 2.4, dur)) * 0.35) * np.exp(-t / 0.055)
    nz = hpf(white(n), 1400) * np.exp(-t / noise_tau)
    nz += bpf(white(n), 3200, 0.8) * np.exp(-t / (noise_tau * 0.5))
    out = softclip(tone * tone_amt + nz * 1.1, 1.4)
    return normalize(out, 0.9)


def clap(dur: float = 0.45, taps=(0.0, 0.009, 0.019, 0.030), tau: float = 0.013,
         tail_tau: float = 0.16):
    """拍手：几个错开的短噪声 + 一条弥散的长尾。"""
    n = nsamp(dur)
    out = np.zeros(n)
    for off in taps:
        i = nsamp(off)
        m = n - i
        out[i:] += white(m) * np.exp(-taxis(m) / tau)
    out += white(n) * np.exp(-taxis(n) / tail_tau) * 0.75
    out = bpf(out, 1300, 0.9)
    out = hpf(out, 600)
    out = softclip(out, 1.3)
    return normalize(out, 0.85)


def hat(dur: float = 0.05, base: float = 320.0, cut: float = 7800.0,
        tau: float = 0.022, decay_curve: float = 1.0):
    """808 味儿的金属 hi-hat：6 个非谐波方波叠加 + 高通 + 极快衰减。"""
    n = nsamp(dur)
    t = taxis(n)
    ratios = (2.0, 3.0, 4.16, 5.43, 6.79, 8.21)
    out = np.zeros(n)
    for r in ratios:
        out += square_naive(phase(base * r, dur))
    out /= len(ratios)
    out = hpf(out, cut, 0.8)
    out *= np.exp(-(t / tau) ** decay_curve)
    out *= np.clip(t / 0.0004, 0, 1)
    return out * 2.2


def crash(dur: float = 2.6, base: float = 300.0, tau: float = 0.85,
          noise_amt: float = 0.7, cut: float = 3200.0):
    """长尾镲：一大把非谐波正弦 + 噪声，高频，慢慢衰减。"""
    n = nsamp(dur)
    t = taxis(n)
    ratios = (1.0, 1.41, 1.68, 2.13, 2.71, 3.14, 3.87, 4.62, 5.31, 6.42,
              7.11, 8.9, 10.4, 12.1, 14.6, 17.3, 20.7, 24.9)
    out = np.zeros(n)
    for i, r in enumerate(ratios):
        f = base * r
        if f > SR * 0.45:
            continue
        out += np.sin(phase(f, dur) + _rng.random() * TAU) / (1 + i * 0.45)
    out /= 2.2
    out += hpf(white(n), 6000) * noise_amt
    out = hpf(out, cut, 0.7)
    out *= np.exp(-t / tau) * np.clip(t / 0.0015, 0, 1)
    return normalize(out, 0.8)


def ride(dur: float = 1.4, base: float = 520.0, tau: float = 0.5):
    """叮叮镲：比 crash 更「有形」，高频成分更集中。"""
    n = nsamp(dur)
    t = taxis(n)
    ratios = (1.0, 1.5, 2.02, 2.61, 3.24, 4.05, 5.12, 6.3, 7.8, 9.6, 12.2)
    out = np.zeros(n)
    for i, r in enumerate(ratios):
        f = base * r
        if f > SR * 0.45:
            continue
        out += np.sin(phase(f, dur) + _rng.random() * TAU) / (1 + i * 0.5)
    out /= 1.8
    out += hpf(white(n), 5000) * 0.35
    out = bpf(out, 6500, 0.55)
    out *= np.exp(-t / tau) * np.clip(t / 0.002, 0, 1)
    return normalize(out, 0.6)


def tom(freq: float = 110.0, dur: float = 0.5, pitch_amt: float = 2.0,
        amp_tau: float = 0.22):
    """嗵鼓：音高包络正弦 + 一点噪声。"""
    n = nsamp(dur)
    t = taxis(n)
    f = freq * (1.0 + (pitch_amt - 1.0) * np.exp(-t / 0.06))
    body = np.sin(phase(f, dur)) * np.exp(-t / amp_tau)
    body += np.sin(phase(f * 1.9, dur)) * np.exp(-t / (amp_tau * 0.3)) * 0.25
    nz = hpf(white(n), 2500) * np.exp(-t / 0.012) * 0.3
    out = softclip(body * 1.3 + nz, 1.2)
    return normalize(out, 0.85)


def rim(dur: float = 0.09, freq: float = 1700.0):
    """边击 / 木鱼：极短的高频噪声 + 一个音。"""
    n = nsamp(dur)
    t = taxis(n)
    out = hpf(white(n), 2200) * np.exp(-t / 0.004)
    out += np.sin(phase(freq, dur)) * np.exp(-t / 0.012) * 0.6
    out += np.sin(phase(freq * 2.7, dur)) * np.exp(-t / 0.006) * 0.3
    return normalize(out, 0.7)


def impact(dur: float = 1.6, freq: float = 45.0):
    """低频轰鸣，段落交界处砸一下。"""
    n = nsamp(dur)
    t = taxis(n)
    f = freq * (1.0 + 1.2 * np.exp(-t / 0.12))
    out = np.sin(phase(f, dur)) * np.exp(-t / 0.5)
    out += lpf(white(n), 300) * np.exp(-t / 0.35) * 0.5
    out = softclip(out * 1.4, 1.2)
    return normalize(out, 0.9)


# ---------------------------------------------------------------------------
#  编曲 / 混音台
# ---------------------------------------------------------------------------


class Section:
    """一个段落：干声 + 两条效果送出（混响 / 延迟）分开攒。

    这样整首曲子只需要把混响总线卷积一次，而不是每条轨各卷一遍。
    """

    def __init__(self, name: str, bars: float, level_db: float = 0.0,
                 ref_rms: float = 0.1):
        self.name = name
        self.bars = bars
        self.n = nsamp(bars * BAR)
        self.level_db = level_db
        self.ref_rms = ref_rms
        self.dry = np.zeros((2, self.n))
        self.rev = np.zeros((2, self.n))
        self.dly = np.zeros((2, self.n))
        self.parts: list[str] = []
        #  按声部名另攒一份干声：给可视化 / 分轨导出用（见 collect_group_stems）。
        #  只在 name 非空时才有内容，正常混音不受影响。
        self.stems: dict[str, np.ndarray] = {}

    def _stem(self, name: str, arr):
        st = self.stems.get(name)
        if st is None:
            st = self.stems[name] = np.zeros((2, self.n))
        st += arr

    def add(self, track, gain_db: float = 0.0, pan: float = 0.0,
            rev: float = 0.0, dly: float = 0.0, name: str = "",
            offset_bars: float = 0.0, duck=None):
        """把一条轨混进来。

        gain_db 是这条轨的电平；pan 只对单声道音色有效；rev / dly 是送进
        两条总线的量（线性，0~1 量级）；offset_bars 让轨从段内某处开始。
        """
        a = to_stereo(track, pan)
        a = fit(a, self.n)
        if offset_bars:
            off = nsamp(offset_bars * BAR)
            if off >= self.n:
                return self  # 起点已经在段末之后，整条轨都不用加了
            a = np.concatenate([np.zeros((2, off)), a[:, :self.n - off]], axis=1)
        # 侧链对齐的是「段内绝对时间」，所以要在摆位之后再乘
        if duck is not None:
            d = fit(duck, self.n)
            a = a * d
        g = db2lin(gain_db)
        self.dry += a * g
        if rev:
            self.rev += a * g * rev
        if dly:
            self.dly += a * g * dly
        if name:
            self.parts.append(name)
            self._stem(name, a * g)

    def add_seq(self, events, gain_db: float = 0.0, pan: float = 0.0,
                rev: float = 0.0, dly: float = 0.0, name: str = "", duck=None):
        """把一串 `(起始秒, 声音)` 事件混成一条轨再加进来。

        比逐个 `add()` 快得多：每个事件只往本地缓冲写一次，最后跟三条总线
        各加一次。事件里的声音可以是数组，也可以是无参函数（延迟生成，
        省内存，也保证随机数按固定顺序消耗）。
        """
        buf = np.zeros((2, self.n))
        for start, a in events:
            x = a() if callable(a) else a
            s = to_stereo(x, pan)
            i = nsamp(start)
            if i < 0:
                s = s[:, -i:]
                i = 0
            if i >= self.n:
                continue
            seg = s[:, : self.n - i]
            buf[:, i:i + seg.shape[1]] += seg
        if duck is not None:
            buf = buf * fit(duck, self.n)
        g = db2lin(gain_db)
        self.dry += buf * g
        if rev:
            self.rev += buf * g * rev
        if dly:
            self.dly += buf * g * dly
        if name:
            self.parts.append(name)
            self._stem(name, buf * g)
        return self

    def finish(self):
        """按 RMS 把整段定到 level_db（干湿一起缩放，比例不变）。

        干湿之间不相关，所以合起来用平方和开根，不能用算术和。
        """
        cur = np.sqrt(rms(self.dry) ** 2 + rms(self.rev) ** 2 + rms(self.dly) ** 2)
        if cur < 1e-12:
            return self
        target = self.ref_rms * db2lin(self.level_db)
        k = target / cur
        self.dry *= k
        self.rev *= k
        self.dly *= k
        for st in self.stems.values():
            st *= k  # 分轨跟干声用同一个缩放，导出来的比例才跟成品一致
        return self

    def info(self):
        return (self.name, rms(self.dry), peak(self.dry), crest_db(self.dry),
                rms(self.dry + self.rev + self.dly))


def collect_group_stems(sections, groups: dict, tail_bars: float = 0.0):
    """把各段按声部名攒下的干声合并成「组」分轨。

    groups = {"Pad": ("pad",), "Hats": ("hat", "openhat"), ...}
    返回 {组名: (2, N)}，N = 各段长度之和 + 尾巴。**只累加到组，不保留单轨**，
    所以内存是「组数 × 总长」而不是「声部数 × 总长」。

    注意这是**母带之前**的干声：没有混响总线、没有总线压缩和削峰，所以各组
    相加不等于 embers.wav。画分轨图正好——哪一轨在响看得一清二楚。
    """
    rev = {}
    for gname, names in groups.items():
        for nm in names:
            rev[nm] = gname
    total = sum(s.n for s in sections) + nsamp(tail_bars * BAR)
    out = {g: np.zeros((2, total)) for g in groups}
    pos = 0
    for s in sections:
        for nm, arr in s.stems.items():
            g = rev.get(nm)
            if g is None:
                continue
            out[g][:, pos:pos + s.n] += arr
        pos += s.n
    return out


def write_wav_mono(path: str, x, sr: int = SR):
    """写 16bit 单声道 wav（分轨用：可视化只关心波形形状，单声道省一半空间）。"""
    import wave as _wave
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = (x[0] + x[1]) / 2.0
    d = np.clip(x, -1.0, 1.0)
    d = (d * 32767.0).astype(np.int16)
    with _wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(d.tobytes())
    return path


def mixdown(sections, tail_bars: float = 4.0, rev_amt: float = 1.0,
            dly_amt: float = 1.0, delay_beat: float = 0.75, delay_fb: float = 0.42,
            delay_mix: float = 0.55, ir_path: str = "IR.wav",
            ir_len: float = 2.4, predelay: float = 0.02,
            ir_hpf: float = 130.0, master_ceiling: float = 0.97,
            master_hpf: float = 28.0, master_air=(3500.0, 0.0),
            master_low=(70.0, 0.0), master_width: float = 0.0,
            master_clip: float = 0.0,
            comp_thresh: float = -14.0, comp_ratio: float = 2.0,
            comp_makeup: float = 1.5,
            fade_in_bars: float = 0.0, fade_out_bars: float = 3.0,
            verbose: bool = True):
    """把各段拼起来、过全局混响/延迟、母带、归一化。

    返回 (stereo float64, 报告 list)。

    母带顺序：总线压缩 -> 高通（切掉 28 Hz 以下的无效能量）-> 高低架音色
    -> 中高频加宽 -> 限制器 -> 归一化。
    """
    tail = nsamp(tail_bars * BAR)
    dry = np.concatenate([s.dry for s in sections], axis=1)
    rev = np.concatenate([s.rev for s in sections], axis=1)
    dly = np.concatenate([s.dly for s in sections], axis=1)
    dry = np.pad(dry, ((0, 0), (0, tail)))
    rev = np.pad(rev, ((0, 0), (0, tail)))
    dly = np.pad(dly, ((0, 0), (0, tail)))

    if peak(rev) > 1e-9:
        ir = load_ir(ir_path, ir_len, ir_hpf)
        pre = nsamp(predelay)
        if 0 < pre < rev.shape[1]:
            rev = np.concatenate([np.zeros((2, pre)), rev[:, :rev.shape[1] - pre]], axis=1)
        wet = fit(convolve(rev, ir), dry.shape[1])
    else:
        wet = np.zeros_like(dry)

    if peak(dly) > 1e-9:
        wd = fit(pingpong(dly, delay_beat * BEAT, delay_fb, delay_mix), dry.shape[1])
    else:
        wd = np.zeros_like(dry)

    mix = dry + wet * rev_amt + wd * dly_amt

    # 母带
    mix = compress(mix, thresh_db=comp_thresh, ratio=comp_ratio,
                   makeup_db=comp_makeup)
    if master_hpf:
        mix = np.stack([hpf(mix[0], master_hpf), hpf(mix[1], master_hpf)])
    if master_low[1]:
        mix = np.stack([low_shelf(mix[0], master_low[0], master_low[1]),
                        low_shelf(mix[1], master_low[0], master_low[1])])
    if master_air[1]:
        mix = np.stack([high_shelf(mix[0], master_air[0], master_air[1]),
                        high_shelf(mix[1], master_air[0], master_air[1])])
    if master_width:
        mix = widen_lowless(mix, master_width, 200.0)
    if master_clip:
        # 软削峰：母带提高响度最直接的手段。只有瞬态会碰到这个阈值，
        # 所以代价是「鼓点更实」，而不是整首变脏。
        mix = master_clip * np.tanh(mix / master_clip)
    mix = normalize(mix, 0.99)
    mix = limiter(mix, ceiling=master_ceiling)

    n_in = nsamp(fade_in_bars * BAR)
    if n_in > 1:
        mix[:, :n_in] *= np.linspace(0, 1, n_in) ** 1.5
    n_out = nsamp(fade_out_bars * BAR)
    if n_out > 1:
        mix[:, -n_out:] *= np.linspace(1, 0, n_out) ** 1.5

    mix = normalize(mix, 0.985)

    report = []
    if verbose:
        print(f"\n{'段落':<12}{'RMS':>9}{'峰值':>9}{'波峰因数':>11}")
        for s in sections:
            r = rms(s.dry)
            report.append((s.name, r, peak(s.dry), crest_db(s.dry)))
            print(f"{s.name:<12}{r:>9.4f}{peak(s.dry):>9.3f}{crest_db(s.dry):>9.1f} dB")
    return mix, report


def write_wav(path: str, x, sr: int = SR):
    """写 16bit 立体声 wav（只用标准库，不依赖 moviepy）。"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, x])
    d = np.clip(x, -1.0, 1.0)
    d = (d * 32767.0).astype(np.int16).T  # (N,2)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(d.tobytes())
    return path


# ---------------------------------------------------------------------------
#  自检：python engine.py 会把每个音色渲一小段，打印峰值 / RMS / 耗时
# ---------------------------------------------------------------------------


def _bench():
    import time

    def show(name, a):
        t0 = time.time()
        x = a if not callable(a) else a()
        ms = (time.time() - t0) * 1000
        p, r = peak(x), rms(x)
        print(f"{name:<22}{p:>7.3f}{r:>8.3f}   {ms:>7.0f} ms")

    b = BEAT
    print(f"SEED={SEED}  BPM={BPM}")
    print(f"{'音色':<20}{'峰值':>7}{'RMS':>8}   {'耗时':>9}")
    show("pad (3音, 2拍)", lambda: pad(chord("D3 F3 A3 C4 E4"), b * 2))
    show("choir (3音, 2拍)", lambda: choir(chord("D3 F3 A3"), b * 2))
    show("sub", lambda: sub(note("D2"), b))
    show("fm_bell", lambda: fm_bell(note("D5"), b * 2))
    show("glass", lambda: glass(note("D5"), b))
    show("acid", lambda: acid(note("D2"), b * 0.5))
    show("supersaw", lambda: supersaw(note("D4"), b))
    show("saw_pluck", lambda: saw_pluck(note("D4"), b * 0.5))
    show("stab", lambda: stab(chord("D3 F3 A3"), b))
    show("noise_riser", lambda: noise_riser(b * 2))
    show("kick", lambda: kick())
    show("snare", lambda: snare())
    show("clap", lambda: clap())
    show("hat", lambda: hat())
    show("openhat", lambda: hat(0.34, tau=0.16))
    show("crash", lambda: crash())
    show("ride", lambda: ride())
    show("tom", lambda: tom(note("A1")))
    show("rim", lambda: rim())
    show("impact", lambda: impact())
    sec = Section("test", 1)
    sec.add(pad(chord("D3 F3 A3"), b * 4), -8, 0, 0.4)
    sec.add(kick(), -6)
    sec.finish()
    print("section info:", ["%.4f" % v for v in sec.info()[1:]])


if __name__ == "__main__":
    _bench()
