from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
from moviepy.editor import AudioFileClip

rate = 44100
channels = 1
bpm = 150
no_reverb = False
no_convolve = False
do_plot = False
side = True
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


short_noise = sample("./short_noise_L.wav") if side else sample("./short_noise_R.wav")

long_noise = sample("./long_noise_L.wav") if side else sample("./long_noise_R.wav")

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
    sub = maximize(punch_sub(note("C2"), 60 / bpm / 4))
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
    res = np.array([0 for _ in range(round(duration * rate))]).astype(np.float16)
    for freq in freq_l:
        res += build_note(freq, duration, func, volume).astype(np.float16)
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
    return arr1 # highpass

def lowpass(arr_in, freq):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= 0
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return arr1 # lowpass

def highgain(arr_in, freq, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return arr1 # highgain

def lowgain(arr_in, freq, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return arr1 # lowgain

def bandgain(arr_in, freq_h, freq_l, times=5):
    arr = arr_in + 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq_h] *= times
    fft_arr[abs(fft_freqs) < freq_l] /= times
    arr1 = np.fft.ifft(fft_arr)
    if max(abs(arr1)) > 1:
        arr1 = maximize(arr1)
    return arr1 # bandgain

def compile_tracks(tracks, volumes, effects, master):
    tracks_l = []
    for track in tracks:
        track_a = np.array([])
        for note in track:
            track_a = np.append(track_a, note.astype(np.float16))
        tracks_l.append(track_a)

    song = tracks_l.pop()
    print(song.shape)
    print(max(abs(song)))
    v = volumes.pop()
    e = effects.pop()
    song = e(song)
    song *= v
    song = song.astype(np.float16)
    print(len(song))
    for track_i in range(len(tracks_l)):
        track = tracks_l[track_i]
        e = effects[track_i]
        v = volumes[track_i]
        print(len(track))
        track = e(track)
        track *= v
        track = track.astype(np.float16)
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
            x = np.convolve(x, n)[:len(x1)]
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

intro = compile_tracks(
    [
        [ # bass
            empty(60 / bpm * 14),
            subdrop(note("C2")),
            
            build_note(note("A1"), 60 / bpm * 4, reese), # 5
            build_note(note("F1"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 2, reese),
            build_note(note("D2"), 60 / bpm * 2, reese),
            
            build_note(note("A1"), 60 / bpm * 4, reese), # 9
            build_note(note("F1"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 2, reese),
            build_note(note("D2"), 60 / bpm * 2, reese),
            
            build_note(note("A1"), 60 / bpm * 4, reese), # 13
            build_note(note("F1"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 2, reese),
            build_note(note("D2"), 60 / bpm * 2, reese),
            
            build_note(note("A1"), 60 / bpm * 4, reese), # 17
            build_note(note("F1"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 4, reese),
            build_note(note("C2"), 60 / bpm * 4, reese),
            
            build_note(note("F1"), 60 / bpm * 4, reese), # 21

            build_note(note("A1"), 60 / bpm / 2 * 3, reese),
            build_note(note("C2"), 60 / bpm / 2 * 5, reese),
            
            crash(60 / bpm * 8)
            
        ],
        [ # pluck melody
            build_note(note("C6"), 60 / bpm / 2 * 3, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm, pluck),
            build_note(note("C6"), 60 / bpm, pluck),
            
            build_note(note("C6"), 60 / bpm / 2 * 3, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("F5"), 60 / bpm / 2, pluck),
            build_note(note("C5"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm, pluck),
            
            build_note(note("G5"), 60 / bpm / 2 * 3, pluck),
            build_note(note("G5"), 60 / bpm / 2 * 3, pluck),
            build_note(note("G5"), 60 / bpm, pluck),
            
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("F5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2 * \

                3, pluck), # 5
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),

            build_note(note("A5"), 60 / bpm, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("F6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2 * \

               5, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm * \

                2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("F5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm, pluck),
            build_note(note("D5"), 60 / bpm / 2 * \

                3, pluck), # 9
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),

            build_note(note("A5"), 60 / bpm, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("F6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2 * \

               5, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm / 2, pluck),
            build_note(note("E5"), 60 / bpm * \

                2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("G5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("D6"), 60 / bpm / 2 * 3, pluck), # 13
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("D6"), 60 / bpm / 2 * 3, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("C6"), 60 / bpm * 3, pluck),
            build_note(note("B5"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm * \

                3, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("A5"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("D6"), 60 / bpm / 2 * 3, pluck), # 17
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("D6"), 60 / bpm / 2 * 3, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("C6"), 60 / bpm * 3, pluck),
            build_note(note("B5"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm * \

                2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            build_note(note("D6"), 60 / bpm / 2, pluck),
            build_note(note("E6"), 60 / bpm / 2, pluck),
            build_note(note("C6"), 60 / bpm / 2, pluck),
            
            build_note(note("A5"), 60 / bpm * 2, pluck), # 21
            build_note(note("G5"), 60 / bpm * 2, pluck),

            build_note(note("B5"), 60 / bpm / 2 * 3, pluck),
            build_note(note("C6"), 60 / bpm / 2 * 5, pluck),
            
            crash(60 / bpm * 8)
        ],
        [ # pluck chord
            build_note(note("A5"), 60 / bpm * 4, pluck),
            build_note(note("F5"), 60 / bpm * 4, pluck),
            build_note(note("E5"), 60 / bpm * 4, pluck),
            build_note(note("C5"), 60 / bpm * 4, pluck),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck), # 5
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm, pluck),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck), # 9
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm, pluck),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck), # 13
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck), # 17
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm / 2, pluck),

            build_chord([note("C6"), note("A5"), note("F5")], 60 / bpm * 4, pluck), # 21
            build_chord([note("G5"), note("B4")], 60 / bpm / 2 * 3, pluck),
            build_chord([note("C5"), note("A4")], 60 / bpm / 2 * 5, pluck),
            
            crash(60 / bpm * 8)
        ],
        [ # arp
            empty(60 / bpm * 16),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw), # 5
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw), # 9
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw), # 13
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw), # 17
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),

            empty(60 / bpm * 8),
            
            crash(60 / bpm * 8)
        ],
        [ # strings
            empty(60 / bpm * 12 * 4),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm * 4, strings), # 13
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm * 4, strings),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm * 4, strings),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm * 2, strings),
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm * 2, strings),

            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm * 4, strings), # 17
            build_chord([note("D6"), note("A5"), note("F5")], 60 / bpm * 4, strings),
            build_chord([note("C6"), note("G5"), note("E5")], 60 / bpm * 4, strings),
            build_chord([note("C6"), note("A5"), note("E5")], 60 / bpm * 4, strings),

            empty(60 / bpm * 8),
            
            crash(60 / bpm * 8)
        ],
        [ # mid-intro kicks
            empty(60 / bpm * 4 * 22),

            highpass(lowpass(psy_punch(note("C2")), 100), 30),
            highpass(lowpass(psy_tail(note("C2")), 100), 30),
            highpass(lowpass(psy_punch(note("C2")), 200), 30),
            highpass(lowpass(psy_tail(note("C2")), 200), 30),
            highpass(lowpass(psy_punch(note("C2")), 300), 30),
            highpass(lowpass(psy_tail(note("C2")), 300), 30),
            highpass(lowpass(psy_punch(note("C2")), 400), 30),
            highpass(lowpass(psy_tail(note("C2")), 400), 30),

            highpass(lowpass(psy_punch(note("C2")), 400), 30),
            highpass(lowpass(psy_tail(note("C2")), 400), 30),
            highpass(lowpass(psy_punch(note("C2")), 400), 30),
            highpass(lowpass(psy_tail(note("C2")), 400), 30),
            highpass(lowpass(psy_punch(note("C2")), 400), 30),
            highpass(lowpass(psy_tail(note("C2")), 400), 30),
            highpass(lowpass(psy_punch(note("C2")), 400), 30),
            highpass(lowpass(psy_tail(note("C2")), 400), 30),
        ],
        [ # mid-intro build-up
            empty(60 / bpm * 4 * 22),
            
            empty(60 / bpm * 4),

            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
        ]
    ],
    [0.5, 0.35, 0.3, 0.04, 0.04, 1, 0.8],
    [lambda x:x, delay, delay, lambda x:x, lambda x:x, lambda x:x, lambda x:x],
    limiter
)
song = intro

midintro = compile_tracks(
    [
        [
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 2),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")) * 0,
            psy_tail(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_punch(note("C2")),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_tail(note("C2"), 1),
            psy_punch(note("C2")),
            psy_punch(note("C2")),
            
            psy_punch(note("C2")),
            psy_tail(note("C2")),
            psy_punch(note("C2")) * 0,
            psy_tail(note("C2")),
            psy_punch(note("C2")) * 0,
            psy_tail(note("C2")),
            psy_punch(note("C2")) * 0,
            psy_tail(note("C2")),

            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),

            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),

            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),

            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),

            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),
            raw_kick2(note("C2")),
            raw_tail(note("A1")),

            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 2),
            raw_kick2(note("C2")),

            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_tail(note("A1"), 60 / bpm / 4),
            highpass(lowpass(raw_kick2(note("C2")), 18000), 30),
            highpass(lowpass(raw_kick2(note("C2")), 9000), 30),
            highpass(lowpass(raw_kick2(note("C2")), 4500), 30),
            highpass(lowpass(raw_kick2(note("C2")), 21500), 30),
            highpass(lowpass(raw_kick2(note("C2")), 1000), 30),
            highpass(lowpass(raw_kick2(note("C2")), 500), 30),
            highpass(lowpass(raw_kick2(note("C2")), 250), 30),
            highpass(lowpass(raw_kick2(note("C2")), 125), 30),
        ],
        [
            empty(60 / bpm * 4 * 15),

            empty(60 / bpm * 2),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
        ]
    ],
    [1, 0.6],
    [lambda x:x, lambda x:x],
    lambda x:x
)
song = np.append(song, midintro)

predrop = compile_tracks(
    [
        [ # drum fill?
            kick(note("C2")),
            empty(60 / bpm / 4),
            kick(note("C2")),
            empty(60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            snare(note("C2"), 60 / bpm / 4),
            kick(note("C2")),
            snare(note("C2"), 60 / bpm / 4),
            kick(note("C2")),
            kick(note("C2")),
            snare(note("C2"), 60 / bpm / 8),
            snare(note("C2"), 60 / bpm / 8),
            snare(note("C2"), 60 / bpm / 8),
            snare(note("C2"), 60 / bpm / 8),

            empty(60 / bpm * (4 * 9 - 3))
        ],
        [ # lead
            empty(60 / bpm * 3),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            
            build_note(note("A5"), 60 / bpm, hardlead),
            build_note(note("A5"), 60 / bpm / 2, hardlead),
            build_note(note("A5"), 60 / bpm / 2, hardlead),
            build_note(note("A5"), 60 / bpm, hardlead),
            build_note(note("A5"), 60 / bpm / 2, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            
            build_note(note("E5"), 60 / bpm, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            build_note(note("E5"), 60 / bpm / 2 * 3, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("C4"), 60 / bpm / 2, hardlead),
            
            build_note(note("D5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("D5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("E5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("A4"), 60 / bpm, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("C5"), 60 / bpm / 2, hardlead),

            build_note(note("D5"), 60 / bpm, hardlead),
            build_note(note("C5"), 60 / bpm / 2, hardlead),
            build_note(note("A4"), 60 / bpm / 2, hardlead),
            empty(60 / bpm),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            
            build_note(note("A5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("A5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("A5"), 60 / bpm / 3 * 2, hardlead),
            build_note(note("A5"), 60 / bpm, hardlead),
            build_note(note("A5"), 60 / bpm / 2, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            
            build_note(note("E5"), 60 / bpm, hardlead),
            build_note(note("G5"), 60 / bpm / 2, hardlead),
            build_note(note("E5"), 60 / bpm / 2 * 3, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("C5"), 60 / bpm / 2, hardlead),
            
            build_note(note("D5"), 60 / bpm / 2 * 3, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("D5"), 60 / bpm / 2, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("D5"), 60 / bpm / 2, hardlead),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            
            build_note(note("G5"), 60 / bpm, hardlead),
            empty(60 / bpm / 2),
            build_note(note("E5"), 60 / bpm / 2, hardlead),
            build_note(note("D5"), 60 / bpm, hardlead),
            build_note(note("C5"), 60 / bpm / 2, hardlead),
            build_note(note("A4"), 60 / bpm / 2, hardlead),
        ],
        [
            empty(60 / bpm * 4),

            build_chord([note("E4"), note("C5"), note("F5"), note("A5")], 60 / bpm, hardchord),
            build_chord([note("E4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("C5"), note("F5"), note("A5")], 60 / bpm, hardchord),
            build_chord([note("E4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 2, hardchord),
            build_chord([note("D4"), note("C5"), note("E5"), note("G5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("D4"), note("A4"), note("C5"), note("E5")], 60 / bpm, hardchord),
            build_chord([note("D4"), note("A4"), note("C5"), note("G5")], 60 / bpm / 2, hardchord),
            build_chord([note("D4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2 * 3, hardchord),
            build_chord([note("D4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("D4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("G4"), note("B4"), note("D5"), note("F5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5"), note("F5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("F4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("F4"), note("G4"), note("A4"), note("E5")], 60 / bpm, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("G4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("E4"), note("G4"), note("B4"), note("D5")], 60 / bpm, hardchord),
            build_chord([note("E4"), note("G4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("F4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            empty(60 / bpm),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("E5"), note("G5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("A4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("A4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("A4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 3 * 2, hardchord),
            build_chord([note("A4"), note("C5"), note("F5"), note("A5")], 60 / bpm, hardchord),
            build_chord([note("A4"), note("C5"), note("F5"), note("A5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C5"), note("E5"), note("G5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("E4"), note("A4"), note("C5"), note("E5")], 60 / bpm, hardchord),
            build_chord([note("E4"), note("A4"), note("C5"), note("G5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2 * 3, hardchord),
            build_chord([note("E4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("A4"), note("C5"), note("E5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("E4"), note("G4"), note("B4"), note("D5")], 60 / bpm / 2 * 3, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("G4"), note("B4"), note("D5"), note("G5")], 60 / bpm, hardchord),
            empty(60 / bpm / 2),
            build_chord([note("E4"), note("G4"), note("B4"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("D4"), note("G4"), note("B4"), note("E5")], 60 / bpm, hardchord),
            build_chord([note("B3"), note("D4"), note("G4"), note("C5")], 60 / bpm / 2, hardchord),
            build_chord([note("D4"), note("F4"), note("A4"), note("C5")], 60 / bpm / 2, hardchord),
        ],
        [
            empty(60 / bpm * 4),
            build_note(note("F1"), 60 / bpm * 4, distorted_reese),
            build_note(note("A1"), 60 / bpm * 4, distorted_reese),
            build_note(note("G1"), 60 / bpm * 4, distorted_reese),
            build_note(note("G1"), 60 / bpm * 2, distorted_reese),
            build_note(note("E2"), 60 / bpm * 2, distorted_reese),

            build_note(note("F1"), 60 / bpm * 4, distorted_reese),
            build_note(note("A1"), 60 / bpm * 4, distorted_reese),
            build_note(note("G1"), 60 / bpm * 4, distorted_reese),
            build_note(note("G1"), 60 / bpm * 4, distorted_reese),
        ],
        [
            empty(60 / bpm * 4),

            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("E4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),

            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("F4"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("A4"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            build_note(note("E5"), 60 / bpm / 4, lp_saw),
            build_note(note("C5"), 60 / bpm / 4, lp_saw),
            
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
            build_note(note("G4"), 60 / bpm / 4, lp_saw),
            build_note(note("B4"), 60 / bpm / 4, lp_saw),
            build_note(note("D5"), 60 / bpm / 4, lp_saw),
            build_note(note("B5"), 60 / bpm / 4, lp_saw),
        ]
    ],
    [0.25, 0.15, 0.35, 0.5, 0.05],
    [
        lambda x:x, \
        eff_chain(eff(highpass, 20), reverb, eff(times, 20), limiter), \
        eff_chain(eff(highpass, 20), reverb, eff(times, 20), limiter),
        eff_chain(eff(highpass, 20), limiter),
        eff_chain(eff(highpass, 20), limiter)
    ],
    eff_chain(eff(highpass, 20), eff(times, 2), limiter)
)

song = np.append(song, predrop)

song = (song * 32767).astype(np.int16)
#              ^^^^^ 淦，之前写的都是1024
plt.plot(song)
plt.show()

# 写入wav
import wave

fname = "L.wav" if side else "R.wav"
with wave.open(fname, 'wb') as f_wav:
    f_wav.setnchannels(channels)
    f_wav.setsampwidth(2)
    f_wav.setframerate(rate)
    f_wav.writeframes(song.tobytes())

# 播放wav
os.system(f"start {fname}")
