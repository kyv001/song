from copy import deepcopy
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from random import randint
from moviepy import AudioFileClip

rate = 44100
channels = 1
bpm = 140
no_reverb = False
no_convolve = False
do_plot = False
side = os.environ.get("SIDE", "L").upper() != "R" # 用 SIDE=R 渲染右声道，默认左声道
play = os.name == "nt" # Windows 上渲染完自动播放；其它平台默认只写文件
if no_reverb:
    print("Reverb is disabled.")

sin = np.sin # sin

def sample(fname):
    audio = AudioFileClip(fname).to_soundarray()
    audio = audio.T
    audio = audio[0]
    audio /= max(abs(audio))
    return audio # sample

def note(n):
    name = n[:-1]
    if not(name.endswith("#") or name.endswith("b")):
        name = name[0]
    oct_ = int(n[-1]) - 5
    a5 = 440
    l = {
        'C': 3,
        'C#': 4,
        'D': 5,
        'D#': 6,
        'E': 7,
        'F': 8,
        'F#': 9,
        'G': 10,
        'G#': 11,
        'A': 12,
        'A#': 13,
        'B': 14,
    }
    freq = a5 * (2 ** ((l[name] + oct_ * 12 + 3) / 12)) # D#调
    return freq # note

def sawtooth(array_in) -> np.array: # 锯齿波
    array = array_in / np.pi / 2
    result = []
    for index in range(len(array)):
        result.append((array[index] - int(array[index]) - 0.5) * 2)
    return np.array(result) # sawtooth

def triangle(array_in) -> np.array: # 三角波
    array = array_in / np.pi / 2
    result = []
    for index in range(len(array)):
        result.append(abs((array[index] - int(array[index]) - 0.5) * 2) * 2 - 1)
    return np.array(result) # triangle

def build_note(freq, duration, func=sawtooth, volume=1, direct_length=False) -> np.array: # 旋律
    if direct_length:
        length = round(duration)
    else:
        length = round(duration * rate)
    try:
        if volume == 0:
            return np.array([0 for _ in range(length)])
    except:
        if volume.all() == 0:
            return np.array([0 for _ in range(length)])
    i = 0
    arr = []
    if type(freq) in [int, float]:
        arr = np.linspace(0, freq * 2 * np.pi * duration, length)
    else:
        for _ in range(length):
            i += freq[_] * 2 * np.pi / rate
            arr.append(i)
    tone_wave = func(np.array(arr)) * volume

    return tone_wave # build_melody

def fnoise(array_in) -> np.array:
    arr = array_in * 0
    array_in = square(array_in)
    n = 0
    for _ in range(len(array_in)):
        if array_in[_] > array_in[_ - 1]:
            n = np.random.randn(1) / 10
        arr[_] = n
    return arr # old noise

def noise(array_in) -> np.array:
    arr = np.random.random(len(array_in)) * 2 - 1
    return arr # white noise

def square(array_in) -> np.array: # 方波
    array = array_in / np.pi / 2
    result = []
    for index in range(len(array)):
        if array[index] % 1 >= 0.5:
            result.append(1)
        else:
            result.append(-1)
    return np.array(result) # square

def lp_saw(array_in):
    a = array_in * 0
    for _ in range(1, 31, 1):
        a += sin(array_in * _) / _ * ((32 - _) / 31)
    return maximize(a) # lp_saw

def lp_saw2(array_in):
    a = array_in * 0
    for _ in range(1, 5, 1):
        a += sin(array_in * _) / _ * ((6 - _) / 5)
    return maximize(a) # lp_saw2

def lp_saw_nosub(array_in):
    a = array_in * 0
    for _ in range(1, 15, 1):
        if _ != 1:
            a += sin(array_in * _) / _ * ((16 - _) / 15)
    return maximize(a) # lp_saw_nosub

def lp_square(array_in):
    a = array_in * 0
    for _ in range(1, 31, 2):
        a += sin(array_in * _) / _ * ((32 - _) / 31)
    return maximize(a) # lp_square

def lp_square2(array_in):
    a = array_in * 0
    for _ in range(1, 6, 2):
        a += sin(array_in * _) / _ * ((7 - _) / 6)
    return maximize(a) # lp_square2

def bass2(array_in):
    arr = sawtooth(array_in * 2) * 0.6 + square(array_in) - sin(array_in)
    return maximize(arr) # bass2

def reese(array_in) -> np.array:
    sawtooth_arr = lp_saw_nosub((array_in + np.random.rand() * np.pi * 2) * 1.01) * 0.3
    sawtooth_arr += lp_saw_nosub((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.3
    sawtooth_arr += lp_saw_nosub((array_in + np.random.rand() * np.pi * 2) * 0.99) * 0.3
    sawtooth_arr = maximize(sawtooth_arr) / 2
    sawtooth_arr += maximize(sin(array_in)) / 2
    return maximize(sawtooth_arr) # reese

def distorted_reese(array_in) -> np.array:
    r = reese(array_in)
    return distortion(r, 0.2, 0.6) # distorted_reese

def strings(array_in) -> np.array:
    sawtooth_arr  = lp_saw((array_in + np.random.rand() * np.pi * 2) * 1.02) * 0.2
    sawtooth_arr += lp_saw((array_in + np.random.rand() * np.pi * 2) * 1.01) * 0.2
    sawtooth_arr += lp_saw((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.2
    sawtooth_arr += lp_saw((array_in + np.random.rand() * np.pi * 2) * 0.99) * 0.2
    sawtooth_arr += lp_saw((array_in + np.random.rand() * np.pi * 2) * 0.98) * 0.2
    sawtooth_arr /= max(abs(sawtooth_arr))
    arr = sawtooth_arr * slide(0, 1, len(array_in), 8000) * slide(0, 1, len(array_in), 3000)[::-1]
    return maximize(arr) # strings

def unison_saw(array_in) -> np.array:
    sawtooth_arr  = sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.02) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.01) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.99) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.98) * 0.2
    sawtooth_arr /= max(abs(sawtooth_arr))
    arr = sawtooth_arr * slide(1, 0.8, len(array_in), 2000)
    return maximize(arr) # unison_saw

def unison_saw2(array_in) -> np.array:
    sawtooth_arr  = sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.04) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.02) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.98) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.96) * 0.2
    sawtooth_arr /= max(abs(sawtooth_arr))
    arr = sawtooth_arr * slide(1, 0.8, len(array_in), 2000)
    return maximize(arr) # unison_saw2

def unison_saw_dirty(array_in) -> np.array:
    sawtooth_arr  = sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.04) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.02) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.98) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * np.pi * 2) * 0.96) * 0.2
    sawtooth_arr /= max(abs(sawtooth_arr))
    arr = sawtooth_arr * slide(1, 0.8, len(array_in), 2000)
    return maximize(arr) # unison_saw_diry

def unison_square(array_in) -> np.array:
    square_arr  = square((array_in + np.random.rand() * np.pi * 2) * 1.02) * 0.2
    square_arr += square((array_in + np.random.rand() * np.pi * 2) * 1.01) * 0.2
    square_arr += square((array_in + np.random.rand() * np.pi * 2) * 1.00) * 0.2
    square_arr += square((array_in + np.random.rand() * np.pi * 2) * 0.99) * 0.2
    square_arr += square((array_in + np.random.rand() * np.pi * 2) * 0.98) * 0.2
    square_arr /= max(abs(square_arr))
    arr = square_arr * slide(1, 0.8, len(array_in), 2000)
    return maximize(arr) # unison_square

def lead_saw1(arr):
    n = sin(arr / 40)
    s = unison_saw(arr + n * 0.3)
    return s # lead_saw1

def lead_saw2(arr):
    n = sin(arr / 60)
    s = unison_saw2(arr + n * 0.5)
    return s # lead_saw2

def lead_pulse1(arr):
    n = sin(arr / 50)
    s = unison_square(arr + n * 0.3)
    return s # lead_pulse1

def hardlead(arr):
    arr += slide(50, 0, len(arr), 600)
    s = lead_saw1(arr) * 0.4 + lead_saw2(arr) * 0.3 + lead_pulse1(arr) * 0.8
    arr /= 2
    s += (lead_saw1(arr) * 0.4 + lead_saw2(arr) * 0.3 + lead_pulse1(arr) * 0.3) * 0.6
    arr /= 2
    s += (lead_saw1(arr) * 0.4 + lead_saw2(arr) * 0.3 + lead_pulse1(arr) * 0.3) * 0.5
    arr *= 8
    s += (lead_saw1(arr) * 0.4 + lead_saw2(arr) * 0.3 + lead_pulse1(arr) * 0.3) * 0.8
    s = maximize(s)
    s = distortion(s, 0.3, 0.6)
    s = highgain(s, 1500, 1.3)
    s = highpass(s, 300)
    s = maximize(s)
    s *= slide(1, 0.8, len(arr), 1000)
    s[-round(60 / bpm / 16 * rate):] *= 0
    return s # hardlead

def hardchord(arr):
    arr += slide(50, 0, len(arr), 600)
    s = unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25
    arr *= 2
    s += unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25
    arr *= 2
    s += (unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25) * 0.8
    arr /= 8
    s += (unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25) * 0.8
    s = maximize(s)
    s = distortion(s, 0.5, 0.55)
    s = bandgain(s, 2000, 1000, 1.3)
    s = highpass(s, 300)
    s = maximize(s)
    s *= slide(1, 0.8, len(arr), 1000)
    s[-round(60 / bpm / 16 * rate):] *= 0
    return s # hardchord

def _screech(arr):
    n = arr / randint(30, 80)
    t = arr / 10
    s = unison_saw_dirty(arr + n * 2) * sin(t)
    s = bandgain(s, 5000, 8000)
    s = highpass(s, 300)
    s = distortion(s, 0.2, 0.7)
    return s # _screech

def screech(freq_start, freq_end, duration):
    freq = np.linspace(freq_start, freq_end, round(duration * rate))
    volume = slide(0, 1, len(freq), 1000)
    return build_note(freq, duration, _screech, volume) # screech

def pluck(arr):
    arr /= 2
    n = sin(arr * 2)
    s = sin(arr + n * slide(3, 0, len(arr), 2000))
    n2 = sin(arr * 3)
    s2 = sin(arr + n2 * slide(4, 0, len(arr), 3000))
    return s * slide(1, 0, len(arr), 8000) + s2 * slide(1, 0, len(arr), 8000) # pluck

def raw_tail(freq, duration=60 / bpm / 4 * 3):
    sub = build_note(freq, duration, triangle, 1)
    sub += maximize(highpass(lowpass(build_note(freq, duration, noise), 3000), 100).astype(np.float64)) * 0.04
    crunch = build_note(freq * 12, duration, triangle, 1.5) + build_note(freq, duration, noise, 0.2)
    crunch *= slide(1, 0, round(duration * rate), 10000)[::-1] * 0.2
    sub += crunch
    sub = distortion(sub, 0.2, 1)
    sub += build_note(freq, duration, np.sin, 0.4)
    sub = maximize(sub)
    sub = distortion(sub, 0.8, 0.9)
    return sub * slide(0.5, 1, round(duration * rate), 2000) # raw_tail

def raw_tail2(freq, duration=60 / bpm / 4 * 3):
    if duration <= 60 / bpm / 4 * 3:
        sub = build_note(np.linspace(freq * 1.5, freq, round(60 / bpm / 4 * 3 * rate))[:round(duration * rate)], duration, lp_saw2, 1)
        sub *= slide(0, 1, round(duration * rate), 1000)
        sub *= slide(0, 1, round(duration * rate), 1000)[::-1]
        return maximize(sub)
    sub = build_note(np.linspace(freq * 1.5, freq, round(60 / bpm / 4 * 3 * rate)), 60 / bpm / 4 * 3, lp_saw2, 1)
    sub *= slide(0, 1, round(60 / bpm / 4 * 3 * rate), 1000)
    sub *= slide(0, 1, round(60 / bpm / 4 * 3 * rate), 1000)[::-1]
    empty = np.array([0 for _ in range(round(duration * rate) - len(sub))])
    sub = np.append(sub, empty)
    return maximize(sub) # raw_tail2

def raw_tail3(freq, duration=60 / bpm / 4 * 3):
    n = long_noise[:round(duration * rate)]
    n = lowpass(n, freq * 8)
    n = bandgain(n, freq + 2, freq - 2, 30)
    n = highpass(n, 40)
    n = maximize(n)
    n = distortion(n, 0.4, 0.6)
    n = highgain(n, freq * 8, 5)
    n = highpass(n, 40)
    n = maximize(n)
    return maximize(n * slide(0, 1, round(duration * rate), 3000) * slide(0, 1, round(duration * rate), 1500)[::-1]) # raw_tail3

def lp_square_tail(freq, duration=60 / bpm / 4 * 3):
    return distortion(build_note(freq, duration, lp_square), 0.2, 0.6) * slide(0, 1, round(duration * rate), 3000) # lp_square_tail

def psy_tail(freq, n_parts=3):
    freq2 = freq * 12
    sub = build_note(freq, 60 / bpm / 4, sin) * 0.9 + short_noise * 0.1
    sub = highpass(lowpass(distortion(sub, 0.2, 0.8), 10000), 20)
    sub *= slide(0, 1, len(sub), 800)
    sub = distortion(maximize(sub), 0.2, 0.4)
    sub = maximize(sub * slide(0, 1, len(sub), 2000)[::-1])
    click = highpass(bandgain(short_noise, freq2 * 1.2, freq2 * 0.9, 40), 400)
    click1 = distortion(click, 0.5, 0.6) * slide(1, 0.01, len(click), 80)
    sub1 = sub + click1 * 2
    sub1 = limit(sub1, 1, -1) * 0.95
    if n_parts == 1:
        return sub1

    click2 = distortion(click, 0.3, 0.6) * slide(1, 0.02, len(click), 90)
    sub2 = sub + click2 * 2
    sub2 = limit(sub2, 1, -1)
    if n_parts == 2:
        return np.append(sub1, sub2)

    click3 = distortion(click, 0.2, 0.8) * slide(1, 0.03, len(click), 100)
    sub3 = sub + click3 * 2
    sub3 = limit(sub3, 1, -1)
    return np.append(np.append(sub1, sub2), sub3) # psy_tail


long_noise = sample("./long_noise_L.wav") if side else sample("./long_noise_R.wav")
# 一个十六分音符的噪声，长度跟着 bpm 走（原来是写死的 rate // 10，即 150bpm 的十六分音符）
short_noise = long_noise[:round(60 / bpm / 4 * rate)]

def kick(freq):
    p = build_note(slide(freq * 8, freq, round(60 / bpm / 4 * rate), 600), 60 / bpm / 4, sin) * slide(0.5, 1, round(60 / bpm / 4 * rate), 2500) * slide(0, 1, round(60 / bpm / 4 * rate), 800)[::-1]
    n = short_noise * slide(0.5, 0, round(60 / bpm / 4 * rate), 200)
    return maximize(p + n) # kick

def raw_kick(freq):
    freq2 = freq * 12
    s1 = bandgain(short_noise, freq2 * 1.1, freq2 * 0.9, 300) #
    s1 = limit(s1 * 2, 1, -1)
    s1 = distortion(maximize(s1), 0.2, 0.8) * slide(0, 1, round(60 / bpm / 4 * rate), 800)
    s2 = distortion(short_noise, 0.1, 0.8) * slide(1, 0, round(60 / bpm / 4 * rate), 100)
    tik = punch3(freq, 60 / bpm / 4)
    s1 += tik + s2
    sub = punch_sub(freq, 60 / bpm / 4)
    s1 += maximize(sub) * 0.5
    return maximize(s1) # raw_kick

def raw_kick2(freq):
    freq2 = freq * 8
    s1 = bandgain(short_noise, freq2 * 1.1, freq2 * 0.9, 300) #
    s1 = limit(s1 * 2, 1, -1)
    s1 = distortion(maximize(s1), 0.2, 0.8) * slide(0, 1, round(60 / bpm / 4 * rate), 800)
    s2 = distortion(short_noise, 0.1, 0.8) * slide(1, 0, round(60 / bpm / 4 * rate), 100)
    tik = punch3(freq, 60 / bpm / 4)
    s1 += tik + s2
    sub = punch_sub(freq, 60 / bpm / 4)
    s1 += maximize(sub) * 0.5
    return maximize(s1) # raw_kick2

def raw_kick3(freq):
    freq2 = freq * 6
    s1 = bandgain(short_noise, freq2 * 1.1, freq2 * 0.9, 300) #
    s1 = limit(s1 * 2, 1, -1)
    s1 = distortion(maximize(s1), 0.2, 0.8) * slide(0, 1, round(60 / bpm / 4 * rate), 800)
    s2 = distortion(short_noise, 0.1, 0.8) * slide(1, 0, round(60 / bpm / 4 * rate), 100)
    tik = punch3(freq, 60 / bpm / 4)
    s1 += tik + s2
    sub = punch_sub(freq, 60 / bpm / 4)
    s1 += maximize(sub) * 0.5
    return maximize(s1) # raw_kick3

def psy_punch(freq):
    freq2 = freq * 12
    p1 = highpass(bandgain(short_noise, freq2 * 1.2, freq2 * 0.9, 40), 400)
    p1 = distortion(p1, 0.2, 0.8) * limit(slide(10, 0, len(p1), 150), 1, 0)
    p1 += punch3(freq, 60 / bpm / 4) * limit(slide(10, 0, len(p1), 150), 1, 0)
    p1 = maximize(p1)
    sub = maximize(punch_sub(freq, 60 / bpm / 4)) # 原本写死成 C2，改成跟着音高走
    reso = highpass(bandgain(short_noise, freq2 * 1.05, freq2 * 0.95, 80), 400)
    reso = distortion(reso, 0.4, 0.6)
    reso[round(len(p1) / 8):] *= slide(0, 1, len(reso) - round(len(p1) / 8), 1000) * slide(0, 1, len(reso) - round(len(p1) / 8), 400)[::-1]
    reso[:round(len(p1) / 8)] *= 0
    punch = maximize(p1 + maximize(sub * 0.4 + reso))
    return punch # psy_punch

def laser_kick(freq, duration=60 / bpm / 4):
    freq2 = freq * 64
    freq4 = slide(freq2, freq, round(duration * rate), 1000) * slide(2, 1, round(duration * rate), 150)
    kick_bass = limit(build_note(freq4, duration, sin, 1) * slide(10, 0.8, round(duration * rate), 150), 1, -1)
    return kick_bass # laser_kick

def frenchcore_kick(freq, duration):
    freq = slide(freq * 20, freq, round(duration * rate), 800)
    k = build_note(freq, duration, triangle)
    k += build_note(freq * 8, duration, triangle) * slide(0, 0.6, len(k), 2000)
    k = distortion(k, 0.1, 1)
    return maximize(k) # frenchcore_kick

def punch1(freq, duration):
    freq = slide(freq * 20, freq, round(duration * rate), 400) + slide(freq * 10, 0, round(duration * rate), 20)
    m = build_note(freq, duration, np.sin) * slide(1, 0.3, round(duration * rate), 800) * slide(1, 3, round(duration * rate), 5000) * slide(0, 1, round(duration * rate), 1000)[::-1]
    return m # punch1

def punch3(freq, duration):
    freq = slide(freq * 80, freq * 8, round(duration * rate), 150)
    m = build_note(freq, duration, np.sin) * slide(10, 0, round(duration * rate), 80)
    return limit(m, 1, -1) # punch3

def punch_sub(freq, duration):
    freq = slide(freq * 8, freq, round(duration * rate), 500)
    m = build_note(freq, duration, np.sin) * slide(0, 1, len(freq), 5000) * slide(0, 1, len(freq), 5000)[::-1]
    return m # punch_sub

def bass_808(freq, duration):
    t = 1
    freq = slide((freq / t * 2), (freq / t), round(duration * rate * t), 80)
    m = build_note(freq, duration * t, np.sin)
    m *= slide(1, 0, round(duration * rate * t), 10000)
    return m[::t] # bass_808

def FM_growl(freq, duration, x, y, nosub=False):
    sub = build_note(freq, duration, sin) * x
    f = build_note(freq, duration, lambda x:x)
    osc12 = triangle(f * 12) * x * 1.5
    osc16 = square(f * 16 + sub * 4) * y
    osc2 = triangle(f * 2 + osc12 + osc16) * x
    osc3 = triangle(f * 3 - osc12 + osc16) * x
    if nosub:
        sub *= 0
    return distortion(maximize(maximize(osc2 + osc3) + sub), 0.4, 0.6) # FM_growl

def FM_growl_oneshot(freq, duration):
    return FM_growl(slide(freq * 2, freq, round(duration * rate), 1000), duration, slide(1, 0, round(duration * rate), 5000), slide(1, 0, round(duration * rate), 5000), False) # FM_growl

def _comb_filter(arr, freq, t=0):
    arr_dry = arr * 1
    if t < 1:
        arr += np.append(build_note(0, rate / freq, sin, 0, True), arr)[:len(arr_dry)]
        return _comb_filter(arr, freq, t + 1)
    return limiter(arr_dry)[:len(arr_dry)] # _comb_filter

def comb_filter(arr, freq, wet=1):
    return arr * (1 - wet) + _comb_filter(arr, freq) * wet # comb_filter

def hihat(duration):
    n = build_note(1, duration, noise)
    n = bandgain(n, 5000, 3000, 1.5)
    n = highpass(n, 2000)
    return n * slide(0.5, 0, round(duration * rate), 1000) * slide(0, 1, round(duration * rate), 2000) # hihat

def crash(duration):
    n = build_note(1, duration, noise)
    n = bandgain(n, 1000, 3000, 100)
    n = bandgain(n, 1500, 1600, 200)
    n = highpass(n, 1000)
    return maximize(n) * slide(1, 0, round(duration * rate), 10000) # crash

def impact(duration):
    n = build_note(1, duration, noise)
    n = lowpass(n, 300)
    n = highpass(n, 10)
    return maximize(n) * slide(1, 0, round(duration * rate), 10000) # impact

def snare(freq, duration):
    t = 5
    freq = slide((freq / t * 6), (freq / t * 4), round(duration * rate * t), 8000)
    m = build_note(freq, duration * t, np.sin)
    m *= slide(1, 0.6, round(duration * rate * t), 5000)
    m *= 1.2
    m = limit(m, 1, -1)
    n = build_note(freq, duration * t, noise)
    n *= slide(1, 0.6, round(duration * rate * t), 7000)
    m += n
    m = maximize(m)
    m *= slide(1, 0, round(duration * rate * t), 8000)
    return distortion(m[::t], 0.2, 0.5) # snare

def subdrop(freq):
    freq = np.linspace(freq * 2, freq, round(60 / bpm * 2 * rate))
    a = build_note(freq, 60 / bpm * 2, sin, 1)
    return a # subdrop

def sweep_up(freq):
    freq1 = np.linspace(freq, freq * 8, round(60 / bpm * 16 * rate))
    n = build_note(freq1, 60 / bpm * 16, noise, 0.8)
    n += build_note(freq1, 60 / bpm * 16, np.sin, 0.2)
    freq2 = np.linspace(0.2, 10, round(60 / bpm * 16 * rate))
    n *= 0.5 - build_note(freq2, 60 / bpm * 16, np.sin, 0.5)
    return n # sweep_up

def empty(duration, direct_length=False):
    return build_note(1, duration, sin, 0, direct_length) # empty

def slide(start, end, length, rate) -> np.array:
    now = 0
    res = []
    for _ in range(round(length)):
        now += (1 - now) / rate
        res.append(start + (end - start) * now)
    return np.array(res) # slide

def limit(array, largest, smallest) -> np.array:
    for i in range(len(array)):
        array[i] = max(smallest, array[i])
        array[i] = min(largest, array[i])
    return array # limit

def build_chord(freq_l, duration, func=sawtooth, volume=1) -> np.array:
    res = np.array([0 for _ in range(round(duration * rate))]).astype(np.float32)
    for freq in freq_l:
        res += build_note(freq, duration, func, volume).astype(np.float32)
    res /= len(freq_l)
    return res # build_chord

def highpass(arr_in, freq):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq] *= 0
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return np.ascontiguousarray(arr1.real) # highpass

def lowpass(arr_in, freq):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= 0
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return np.ascontiguousarray(arr1.real) # lowpass

def highgain(arr_in, freq, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return np.ascontiguousarray(arr1.real) # highgain

def lowgain(arr_in, freq, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return np.ascontiguousarray(arr1.real) # lowgain

def bandgain(arr_in, freq_h, freq_l, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq_h] *= times
    fft_arr[abs(fft_freqs) < freq_l] /= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return np.ascontiguousarray(arr1.real) # bandgain

def compile_tracks(tracks, volumes, effects, master):
    tracks_l = []
    for track in tracks:
        track_a = np.array([])
        for note in track:
            track_a = np.append(track_a, note.astype(np.float32))
        tracks_l.append(track_a)

    song = tracks_l.pop()
    print(song.shape)
    print(max(abs(song)))
    v = volumes.pop()
    e = effects.pop()
    song = e(song)
    song *= v
    song = song.astype(np.float32)
    print(len(song))
    for track_i in range(len(tracks_l)):
        track = tracks_l[track_i]
        e = effects[track_i]
        v = volumes[track_i]
        print(len(track))
        track = e(track)
        track *= v
        track = track.astype(np.float32)
        song += track

    song = master(song)
    if max(abs(song)) > 1:
        song = limiter(song)

    if do_plot:
        plt.plot(song)
        plt.show()

    return song # compile_tracks

def maximize(arr):
    arr /= max(abs(arr))
    return arr # maximize

audio = audio = AudioFileClip("IR.wav")
n = audio.to_soundarray().T[0 if side else 1] # IR

def _reverb(x, t=0, t_max=1000):
    x1 = np.append(empty(randint(100, 300), True), x)[:len(x)] * randint(700, 800) / 1000
    if t == t_max:
        return x1
    else:
        return _reverb(x1, t + 1, t_max) # _reverb

def _er(x, t=0, t_max=30):
    x1 = np.append(empty(randint(2000, 2400), True), x)[:len(x)] * randint(30, 50) / 1000
    if t == t_max:
        return x1
    else:
        return _reverb(x1, t + 1, t_max) # _er

def reverb(x, dry=0.7):
    if not no_reverb:
        if not no_convolve:
            x1 = deepcopy(x)
            n[-1] = 0
            print("convolving")
            x = scipy.signal.fftconvolve(x, n)[:len(x1)]
            x = maximize(x)
            print("finish")
            x = x1 * dry + x * (1 - dry)
            return x # reverb
        print("doing reverb")
        x2 = highpass(lowpass(x, 100000), 400)
        x1 = maximize(_reverb(x2) + _er(x2) * 0.3) * (1 - dry) + x * dry
        print("finish")
        return x1
    return x # reverb

def delay(x, dry=0.8):
    x_dry = deepcopy(x)
    x = np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x += np.append(build_note(0, 60 / bpm / 4, sin, 0), x * 0.8)[:len(x_dry)]
    x = maximize(x)[:len(x_dry)] * (1 - dry)
    x_dry *= dry
    x += x_dry
    return x # delay

def distortion(a, x, y):
    for i in range(len(a)):
        flipped = False
        if a[i] < 0:
            a[i] = -a[i]
            flipped = True
        if abs(a[i]) < x:
            a[i] = a[i] / x * y
        else:
            a[i] = (a[i] - x) / (1 - x) * (1 - y) + y
        if flipped:
            a[i] = -a[i]
    return a # distortion

def scratch(x, t):
    x1 = x * 0
    for i in range(len(x)):
        x1[i] = x[limit(np.array([i + round(t[i])]), len(x) - 1, 0)[0]]
    return x1 # scratch

s = build_note(1 / (60 / bpm), 60 / bpm)
s += 1
s *= 2
s = limit(s, 1, 0)
sidechain = distortion(s, 0.7, 0.3)

def eff(func, *args, **kwargs):
    def f(arr):
        return func(arr, *args, **kwargs)
    return f # eff

def eff_chain(*l):
    def f(arr):
        for e in l:
            arr = e(arr)
        return arr
    return f # eff_chain

def limiter(x):
    print("compressing, max volume:{}".format(round(max(abs(x)), 2)))
    a = 1
    for i in range(len(x)):
        if abs(x[i]) > a:
            a = abs(x[i]) * 1.1
        else:
            if a > 1:
                a *= 0.998
        x[i] /= a

    print("finish")
    return maximize(x) # compressor

def declick(x):
    for i in range(len(x)):
        if len(x) - 2 > i > 0:
            #if abs(x[i]) >= 0.6 and abs(x[i - 1]) <= 1e-2 and abs(x[i - 1]) <= 1e-2:
            if abs(x[i - 1] - x[i + 1]) < 1e-1 and abs(x[i - 1] - x[i]) > 0.6:
                x[i] = x[i - 1] # de-clicking
                print("Click!")
    return x # declick

def times(x, t):
    return x * t # times


# =============================================================================
#  《潮汐》 / Tide
#  ---------------------------------------------------------------------------
#  140 BPM · 记谱是 E 自然小调，因为 note() 整体上移了三个半音，实际听感是 G 小调
#
#  和声（两小节一个和弦，八小节一个循环）
#      Em  Em | C   C  | G   G  | D   D
#      Gm  Gm | Eb  Eb | Bb  Bb | F   F          ← 耳朵里听到的
#
#  结构
#      I   潮起  16 小节   氛围铺底，低音从远处浮起来，动机在末尾露头
#      II  潮涌   8 小节   鼓组与 rolling bass 进入，潮水开始推
#      III 主题  16 小节   主旋律 A，第二遍加厚
#      IV  退潮   8 小节   抽掉鼓组，和声回落，末尾 riser 把水重新拉高
#      V   高潮  24 小节   主题 A / B 交替，加三度和声，全奏
#      VI  余波   8 小节   收束，留白，淡出
#      合计 80 小节 ≈ 2 分 17 秒
# =============================================================================
import time

beat = 60 / bpm            # 一拍
bar = beat * 4             # 一小节
print("《潮汐》 {} BPM · 一小节 {:.3f}s · 预计 {:.0f} 秒".format(bpm, bar, 80 * bar))

# 和声表：(根音, 和弦音)
PROG = [
    ("E", ["E", "G", "B"]),   # i
    ("C", ["C", "E", "G"]),   # VI
    ("G", ["G", "B", "D"]),   # III
    ("D", ["D", "F#", "A"]),  # VII
]
SCALE = ["E", "F#", "G", "A", "B", "C", "D"]   # E 自然小调


def banner(text):
    print("\n" + "=" * 62 + "\n  " + text + "\n" + "=" * 62, flush=True)


def swell(n, curve=1.0):
    """两头归零的呼吸包络，避免爆音"""
    return np.sin(np.linspace(0, np.pi, n)) ** curve


def chord_notes(ci, oct_=3):
    """第 ci 个和弦的音（三和弦 + 高八度根音）"""
    root, tones = PROG[ci % len(PROG)]
    return [note(t + str(oct_)) for t in tones] + [note(root + str(oct_ + 1))]


def root_at(bar_i, oct_=2):
    """第 bar_i 小节的低音根音"""
    return note(PROG[(bar_i // 2) % len(PROG)][0] + str(oct_))


def third_below(name):
    """小调音阶里往下数三度，用来生成和声声部"""
    letter, octv = name[:-1], int(name[-1])
    j = SCALE.index(letter) - 2
    if j < 0:
        j += 7
        octv -= 1
    return SCALE[j] + str(octv)


def mk_track(items, bars):
    """把音符片段拼成一条完整长度的轨（不足补零、超出截断）"""
    n = round(bars * bar * rate)
    items = [np.asarray(np.real(x), dtype=np.float64) for x in items if len(x)]
    if not items:
        return np.zeros(n)
    a = np.concatenate(items)
    if len(a) < n:
        a = np.concatenate([a, np.zeros(n - len(a))])
    return a[:n]


def grid(bars, events):
    """把 (小节号, 音数组) 摆到小节网格上，允许重叠相加"""
    n = round(bars * bar * rate)
    out = np.zeros(n)
    for b, a in events:
        s = round(b * bar * rate)
        if s >= n:
            continue
        e = min(n, s + len(a))
        out[s:e] += np.real(a[:e - s])
    return out


def duck(bars):
    """整段的侧链包络：跟着 kick 一起呼吸"""
    n = round(bars * bar * rate)
    return np.tile(sidechain, int(np.ceil(n / len(sidechain))))[:n]


# -----------------------------------------------------------------------------
#  各个声部
# -----------------------------------------------------------------------------
#  各乐器在「单位音量」下的实测 RMS，混音时按它反推音量，而不是凭感觉拧：
#      kick 0.49 | psy_punch 0.35 | psy_tail 0.67 | snare 0.30 | hat 0.16
#      pluck 0.85 | hardlead 0.34 | unison_saw 0.30 | pad 0.25 | 纯正弦 0.71
#  音量 = 想要的 RMS / 实测值。
# -----------------------------------------------------------------------------
def sub_track(bars, oct_=2, curve=0.4):
    """每两小节一个根音的纯正弦低音，带呼吸感"""
    out = []
    for i in range(0, bars, 2):
        n = round(beat * 8 * rate)
        # 留一点底噪不归零，免得每两小节出现一次「断气」
        out.append(build_note(root_at(i, oct_), beat * 8, sin) * (0.35 + 0.65 * swell(n, curve)))
    return mk_track(out, bars)


def pad_track(bars, seq=None, start=0, oct_=3, detune=1.005):
    """弦乐铺底：两小节一个和弦，两路轻微失谐叠加"""
    out = []
    for k, i in enumerate(range(0, bars, 2)):
        ci = seq[k] if seq else (start + i) // 2
        ns = chord_notes(ci, oct_)
        dur = beat * 8
        voices = []
        for f in ns:
            voices.append(build_note(f, dur, lp_saw))
            voices.append(build_note(f * detune, dur, lp_saw))
        c = np.sum(voices, axis=0) / len(voices)
        out.append(c * swell(len(c), 0.5))
    return mk_track(out, bars)


def rolling_track(bars, start=0, oct_=2, punch=psy_punch, tail=psy_tail):
    """psy 的 rolling bass：每拍一个 punch + 三个十六分的 tail"""
    out = []
    for i in range(bars):
        f = root_at(start + i, oct_)
        for _ in range(4):
            out.append(punch(f))
            out.append(tail(f))
    return mk_track(out, bars)


def reese_track(bars, oct_=2, voice=distorted_reese, hold=2):
    """每 hold 小节换一次的 growl 低音"""
    out = []
    for i in range(0, bars, hold):
        out.append(build_note(root_at(i, oct_), beat * 4 * hold, voice))
    return mk_track(out, bars)


def four_floor(bars, oct_=1, kicker=kick):
    """四踩底鼓（根音定在主音上，所以整首的鼓都是同一个音高）"""
    out = []
    f = note("E" + str(oct_))
    for _ in range(bars * 4):
        out.append(kicker(f))
        out.append(empty(beat * 3 / 4))
    return mk_track(out, bars)


def hat(duration, bright=6500, decay=45):
    """闭合 hi-hat。引擎自带的 hihat() 起音太慢，峰值只有 0.044，等于没声音"""
    n = highpass(build_note(1, duration, noise), bright)
    return n * np.exp(-np.linspace(0, 1, len(n)) * decay) * 1.5


def crash_wash(duration, bright=4500, decay=3.0):
    """长尾 crash。引擎自带的 crash() 衰减太快，0.2 秒就没了"""
    n = highpass(build_note(1, duration, noise), bright)
    return n * np.exp(-np.linspace(0, 1, len(n)) * decay) * 1.2


def hat_track(bars, sixteenth=False):
    """反拍八分（或十六分）hi-hat"""
    out = []
    for _ in range(bars * 4):
        if sixteenth:
            out.append(empty(beat / 4))
            out.append(hat(beat / 4))
            out.append(empty(beat / 4))
            out.append(hat(beat / 4))
        else:
            out.append(empty(beat / 2))
            out.append(hat(beat / 2))
    return mk_track(out, bars)


def backbeat(bars, freq_name="B3"):
    """二、四拍军鼓"""
    out = []
    for _ in range(bars):
        out.append(empty(beat))
        out.append(snare(note(freq_name), beat / 4))
        out.append(empty(beat * 3 / 4))
        out.append(empty(beat))
        out.append(snare(note(freq_name), beat / 4))
        out.append(empty(beat * 3 / 4))
    return mk_track(out, bars)


def roll_bar(freq_name="B3"):
    """一小节的军鼓渐密滚奏"""
    out = []
    for div, rep in ((4, 2), (8, 2), (16, 4)):
        for _ in range(rep):
            out.append(snare(note(freq_name), beat / div))
    return out


def arp_track(bars, start=0, oct_=4, voice=pluck, div=4):
    """琶音：一小节 div*4 个音，上下往返"""
    out = []
    for i in range(bars):
        ns = chord_notes((start + i) // 2, oct_)
        seq = ns + ns[-2:0:-1]
        for k in range(4 * div):
            out.append(build_note(seq[k % len(seq)], beat / div, voice))
    return mk_track(out, bars)


def lead_track(bars, spec, start_bar=0, voice=hardlead, oct_shift=0):
    """把一条旋律写成轨"""
    out = []
    if start_bar:
        out.append(empty(bar * start_bar))
    for name, beats in spec:
        out.append(build_note(note(name) * (2.0 ** oct_shift), beat * beats, voice))
    return mk_track(out, bars)


def harmony_track(bars, spec, start_bar=0, voice=unison_saw):
    """主题下方的三度和声声部"""
    out = []
    if start_bar:
        out.append(empty(bar * start_bar))
    for name, beats in spec:
        out.append(build_note(note(third_below(name)), beat * beats, voice))
    return mk_track(out, bars)


def compile_section(name, bars, tracks, volumes, effects, level):
    """把若干条等长轨混成一段，再把整段定到目标 RMS。

    段落强弱用 RMS 而不是峰值来定：峰值受瞬时尖峰影响太大，同样的峰值下
    密集的段落听起来会响得多。峰值最后交给整首统一归一化，段间比例不会被破坏。
    """
    t0 = time.time()
    banner("{} （{} 小节 / {:.1f} 秒）".format(name, bars, bars * bar))
    sec = compile_tracks([[t] for t in tracks], volumes,
                         effects, eff_chain(limiter, eff(times, 0.9)))
    sec = sec.astype(np.float64)
    rms0 = float(np.sqrt(np.mean(sec ** 2)))
    sec = (sec * (level / rms0)).astype(np.float32)
    rms1 = float(np.sqrt(np.mean(sec ** 2)))
    peak = float(np.max(np.abs(sec)))
    print("  << {} 用时 {:.1f}s  RMS {:.4f}  峰值 {:.3f}  波峰因数 {:.1f}dB".format(
        name, time.time() - t0, rms1, peak, 20 * np.log10(peak / rms1)))
    return sec


# -----------------------------------------------------------------------------
#  主题
# -----------------------------------------------------------------------------
# 主题 A：八小节乐句，和声 Em Em C C G G D D
THEME_A = [
    ("B4", 0.5), ("E5", 0.5), ("G5", 1), ("F#5", 0.5), ("E5", 0.5), ("D5", 1),
    ("E5", 1), ("D5", 0.5), ("B4", 0.5), ("E5", 2),

    ("C5", 0.5), ("E5", 0.5), ("G5", 1), ("A5", 1), ("B5", 1),
    ("A5", 1), ("G5", 0.5), ("E5", 0.5), ("G5", 2),

    ("B5", 1), ("A5", 0.5), ("G5", 0.5), ("D5", 1), ("G5", 1),
    ("F#5", 1), ("G5", 1), ("A5", 2),

    ("A5", 0.5), ("B5", 0.5), ("A5", 1), ("F#5", 1), ("D5", 1),
    ("E5", 2), ("F#5", 1), ("E5", 1),
]

# 主题 B：高潮用的变奏，节奏更冲
THEME_B = [
    ("E5", 0.5), ("G5", 0.5), ("B5", 1), ("A5", 0.5), ("G5", 0.5), ("E5", 1),
    ("G5", 0.5), ("F#5", 0.5), ("E5", 1), ("B4", 1), ("E5", 1),

    ("G5", 0.5), ("A5", 0.5), ("B5", 1), ("A5", 2),
    ("G5", 1), ("E5", 1), ("G5", 2),

    ("D5", 0.5), ("G5", 0.5), ("B5", 2), ("A5", 1),
    ("G5", 1), ("F#5", 1), ("G5", 2),

    ("A5", 0.5), ("B5", 0.5), ("A5", 1), ("F#5", 1), ("D5", 1),
    ("F#5", 1), ("A5", 1), ("B5", 2),
]

# 只取主题 A 的后半句（第 5-8 小节），用于退潮段
THEME_A2 = THEME_A[19:]


# =============================================================================
#  I. 潮起 —— 氛围层，只有低音、铺底和一点点琶音
# =============================================================================
rise = compile_section(
    "I. 潮起", 16,
    [
        sub_track(16, oct_=2),
        pad_track(16, oct_=3),
        arp_track(16, start=8, oct_=4, voice=pluck, div=2),
        grid(16, [
            (0, subdrop(note("E2"))),
            (8, impact(bar)),
            (12, crash_wash(bar * 4)),
            (12, sweep_up(note("E1"))),
        ]),
    ],
    [0.24, 0.52, 0.09, 0.30],  # 低音 / 铺底 / 琶音 / 音效
    [
        eff(highpass, 25),
        eff_chain(eff(highpass, 120), eff(reverb, 0.78)),
        eff_chain(eff(highpass, 250), eff(reverb, 0.9), eff(times, 2)),
        eff(highpass, 30),
    ],
    0.085
)

# =============================================================================
#  II. 潮涌 —— 鼓组和 rolling bass 进来，潮水开始推
# =============================================================================
surge = compile_section(
    "II. 潮涌", 8,
    [
        four_floor(8),
        rolling_track(8),
        hat_track(8),
        pad_track(8, oct_=3),
        arp_track(8, oct_=4, voice=pluck, div=2),
        grid(8, [
            (0, crash_wash(bar * 2)),
            (7, np.concatenate(roll_bar())),
        ]),
    ],
    [0.32, 0.29, 1.40, 0.34, 0.10, 0.32],  # 鼓 / 低音 / 钉钉 / 铺底 / 琶音 / 音效
    [
        eff(highpass, 30),
        eff_chain(eff(highpass, 35), eff(times, duck(8))),
        eff(highpass, 1000),
        eff_chain(eff(highpass, 150), eff(reverb, 0.85)),
        eff_chain(eff(highpass, 250), eff(reverb, 0.9)),
        eff(highpass, 30),
    ],
    0.150
)

# =============================================================================
#  III. 主题 —— 主旋律 A 走两遍
# =============================================================================
theme = compile_section(
    "III. 主题", 16,
    [
        four_floor(16),
        rolling_track(16),
        hat_track(16),
        pad_track(16, oct_=3),
        arp_track(16, oct_=4, voice=lp_saw, div=4),
        lead_track(16, THEME_A, voice=hardlead),
        lead_track(16, THEME_A, start_bar=8, voice=hardlead),
        backbeat(16),
        grid(16, [
            (0, crash_wash(bar * 2)),
            (8, crash_wash(bar * 2)),
            (15, np.concatenate(roll_bar())),
        ]),
    ],
    [0.31, 0.28, 1.35, 0.25, 0.05, 0.42, 0.42, 0.45, 0.32],
    [
        eff(highpass, 30),
        eff_chain(eff(highpass, 35), eff(times, duck(16))),
        eff(highpass, 1000),
        eff_chain(eff(highpass, 150), eff(reverb, 0.85)),
        eff_chain(eff(highpass, 300), eff(times, duck(16))),
        eff_chain(eff(highpass, 200), eff(reverb, 0.87), eff(times, duck(16))),
        eff_chain(eff(highpass, 200), eff(reverb, 0.87), eff(times, duck(16))),
        eff(highpass, 300),
        eff(highpass, 30),
    ],
    0.195
)

# =============================================================================
#  IV. 退潮 —— 抽掉鼓组，和声回落，末尾 riser 把水重新拉高
# =============================================================================
ebb = compile_section(
    "IV. 退潮", 8,
    [
        pad_track(8, oct_=3),
        sub_track(8, oct_=2, curve=0.6),
        reese_track(8, oct_=2, hold=2),
        lead_track(8, THEME_A2, voice=unison_saw),
        grid(8, [
            (0, impact(bar * 2)),
            (4, subdrop(note("C2"))),
            (6, sweep_up(note("E1"))[:round(bar * 2 * rate)]),
            (7, np.concatenate(roll_bar())),
        ]),
    ],
    [0.72, 0.20, 0.16, 0.42, 0.32],  # 铺底 / 低音 / reese / 主音 / 音效
    [
        eff_chain(eff(highpass, 120), eff(reverb, 0.78)),
        eff(highpass, 25),
        eff_chain(eff(highpass, 30), eff(reverb, 0.85)),
        eff_chain(eff(highpass, 250), eff(reverb, 0.85)),
        eff(highpass, 30),
    ],
    0.115
)

# =============================================================================
#  V. 高潮 —— 主题 A / B 交替，加三度和声，全奏
# =============================================================================
climax = compile_section(
    "V. 高潮", 24,
    [
        four_floor(24),
        rolling_track(24),
        hat_track(24, sixteenth=True),
        pad_track(24, oct_=3),
        arp_track(24, oct_=4, voice=lp_saw, div=4),
        lead_track(24, THEME_A),
        lead_track(24, THEME_B, start_bar=8),
        lead_track(24, THEME_A, start_bar=16),
        harmony_track(24, THEME_B, start_bar=8, voice=unison_saw),
        harmony_track(24, THEME_A, start_bar=16, voice=unison_saw),
        backbeat(24),
        grid(24, [
            (0, crash_wash(bar * 4)),
            (8, crash_wash(bar * 4)),
            (16, crash_wash(bar * 4)),
            (7, np.concatenate(roll_bar())),
            (15, np.concatenate(roll_bar())),
            (23, np.concatenate(roll_bar())),
        ]),
    ],
    [0.31, 0.28, 1.30, 0.26, 0.05, 0.42, 0.42, 0.42, 0.28, 0.28, 0.45, 0.32],
    [
        eff(highpass, 30),
        eff_chain(eff(highpass, 35), eff(times, duck(24))),
        eff(highpass, 1000),
        eff_chain(eff(highpass, 150), eff(reverb, 0.85)),
        eff_chain(eff(highpass, 300), eff(times, duck(24))),
        eff_chain(eff(highpass, 200), eff(reverb, 0.87), eff(times, duck(24))),
        eff_chain(eff(highpass, 200), eff(reverb, 0.87), eff(times, duck(24))),
        eff_chain(eff(highpass, 200), eff(reverb, 0.87), eff(times, duck(24))),
        eff_chain(eff(highpass, 250), eff(reverb, 0.88), eff(times, duck(24))),
        eff_chain(eff(highpass, 250), eff(reverb, 0.88), eff(times, duck(24))),
        eff(highpass, 300),
        eff(highpass, 30),
    ],
    0.280
)

# =============================================================================
#  VI. 余波 —— 收束、留白、淡出
# =============================================================================
afterglow = compile_section(
    "VI. 余波", 8,
    [
        pad_track(8, seq=[0, 1, 0, 0], oct_=3),
        sub_track(8, oct_=2, curve=0.7),
        arp_track(8, start=0, oct_=4, voice=pluck, div=2),
        lead_track(8, THEME_A[:10], start_bar=2, voice=unison_saw),
        grid(8, [
            (0, impact(bar * 2)),
            (4, subdrop(note("E2"))),
        ]),
    ],
    [0.62, 0.21, 0.11, 0.34, 0.30],
    [
        eff_chain(eff(highpass, 120), eff(reverb, 0.78)),
        eff(highpass, 25),
        eff_chain(eff(highpass, 250), eff(reverb, 0.9)),
        eff_chain(eff(highpass, 250), eff(reverb, 0.85)),
        eff(highpass, 30),
    ],
    0.080
)

# =============================================================================
#  拼装、淡入淡出、写盘
# =============================================================================
sections = [("I.潮起", rise), ("II.潮涌", surge), ("III.主题", theme),
            ("IV.退潮", ebb), ("V.高潮", climax), ("VI.余波", afterglow)]

banner("拼装")
song = np.concatenate([s for _, s in sections])
print("总长 {:.1f} 秒".format(len(song) / rate))
for nm, sec in sections:
    r = float(np.sqrt(np.mean(sec.astype(np.float64) ** 2)))
    print("   {:<8} RMS {:.4f}  ({:+.1f} dB)".format(nm, r, 20 * np.log10(r / 0.280)))

n_in = round(bar * 2 * rate)
song[:n_in] *= np.linspace(0, 1, n_in) ** 1.5
n_out = round(bar * 3 * rate)
song[-n_out:] *= np.linspace(1, 0, n_out) ** 1.5

song = song / np.max(np.abs(song)) * 0.98

song = (song * 32767).astype(np.int16)

if do_plot:
    plt.plot(song[:rate * 10])
    plt.show()

# 写入 wav
import wave

fname = "L.wav" if side else "R.wav"
with wave.open(fname, 'wb') as f_wav:
    f_wav.setnchannels(channels)
    f_wav.setsampwidth(2)
    f_wav.setframerate(rate)
    f_wav.writeframes(song.tobytes())
print("已写出 {} ：{} 秒".format(fname, round(len(song) / rate, 1)))

# 播放 wav（Windows 下渲染完自动播放，其它平台默认只写文件）
if os.name == "nt" and play:
    os.system("start {}".format(fname))
