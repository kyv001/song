from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import AudioFileClip

rate = 44100
channels = 1
bpm = 150

sin = np.sin # sin

def note(n):
    name = n[:-1]
    if not(name.endswith("#") or name.endswith("b")):
        name = name[0]
    print(name)
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
    freq = a5 * (2 ** ((l[name] + oct_ * 12) / 12))
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

def square(array_in) -> np.array: # 方波
    array = array_in / np.pi / 2
    result = []
    for index in range(len(array)):
        if array[index] % 1 >= 0.5:
            result.append(1)
        else:
            result.append(-1)
    return np.array(result) # square

def unison_saw(array_in) -> np.array:
    sawtooth_arr  = sawtooth((array_in + np.random.rand() * 100) * 1.06) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * 100) * 1.03) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * 100) * 1.00) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * 100) * 0.97) * 0.2
    sawtooth_arr += sawtooth((array_in + np.random.rand() * 100) * 0.94) * 0.2
    sawtooth_arr /= max(abs(sawtooth_arr))
    arr = sawtooth_arr * slide(1, 0.8, len(array_in), 2000)
    return arr # unison_saw

def lead_saw1(arr):
    n = fnoise(arr * 5)
    s = sawtooth(arr + n * 3)
    return s # lead_saw1

def lead_saw2(arr):
    n = fnoise(arr * 5)
    s = unison_saw(arr + n * 3)
    return s # lead_saw2

def lead_pulse1(arr):
    n = fnoise(arr * 2)
    s = square(arr + n * 3)
    return s # lead_pulse1

def hardlead(arr):
    arr[-round(60 / bpm / 16 * rate):] *= 0
    s = lead_saw1(arr) * 0.2 + lead_saw2(arr) * 0.6 + lead_pulse1(arr) * 0.2
    arr *= 2
    s += (lead_saw1(arr) * 0.2 + lead_saw2(arr) * 0.6 + lead_pulse1(arr) * 0.2) * 0.8
    s = distortion(s, 0.4, 0.6)
    s = highpass(s, 300)
    s = highgain(s, 1500, 1.1)
    s = maximize(s)
    s *= slide(1, 0.8, len(arr), 1000)
    return s # hardlead

def hardchord(arr):
    arr[-round(60 / bpm / 16 * rate):] *= 0
    s = unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25
    arr *= 2
    s += (unison_saw(arr) * 0.2 + lead_pulse1(arr) * 0.3 + lead_saw1(arr) * 0.25 + lead_saw2(arr) * 0.25) * 0.8
    s = distortion(s, 0.5, 0.55)
    s = highpass(s, 300)
    s = bandgain(s, 2000, 1000, 1.1)
    s = maximize(s)
    s *= slide(1, 0.8, len(arr), 1000)
    return s # hardchord

def raw_tail(freq, duration):
    sub = build_melody(freq, duration, triangle, 1)
    sub += build_melody(freq, duration, noise, 0.03)
    sub *= slide(0, 1, round(duration * rate), 800)
    sub = limit(sub * 1.2, 1, -1)
    crunch = build_melody(freq * 8, duration, triangle, 1) + build_melody(freq, duration, noise, 1)
    crunch *= slide(0, 1, round(duration * rate), 20000) * 0.6 * slide(0, 1, round(duration * rate), 1500)[::-1]
    sub += crunch
    sub = distortion(sub, 0.1, 0.95)
    sub += build_melody(freq, duration, np.sin, 0.4)
    sub = maximize(sub)
    sub = distortion(sub, 0.8, 0.95)
    return sub # raw_tail

def raw_kick(freq):
    freq2 = freq * 8
    freq3 = np.linspace(freq2, freq2 / 2, round(60 / bpm / 4 * rate))
    s1 = build_melody(freq3, 60 / bpm / 4, triangle, 0.8) * slide(0, 1, round(60 / bpm / 4 * rate), 1000)
    s2 = build_melody(freq2, 60 / bpm / 4, noise, 0.03)
    tik = kick(freq, 60 / bpm / 4) * 1
    s1 += tik + s2
    s1 = limit(s1 * 80, 1, -1) * 0.6
    tik2 = kick(freq, 60 / bpm / 4) * 0.4
    s1 += tik2
    return distortion(s1, 0.4, 0.6) # raw_kick

def raw_kick2(freq):
    freq2 = freq * 8
    freq3 = np.linspace(freq2, freq2 * 0.8, round(60 / bpm / 4 * rate))
    s1 = build_melody(freq3, 60 / bpm / 4, triangle, 1) * slide(0, 1, round(60 / bpm / 4 * rate), 2000)
    s2 = build_melody(freq2, 60 / bpm / 4, noise, 0.06)
    tik = kick(freq / 2, 60 / bpm / 2)[::2] * slide(1, 0, round(60 / bpm / 4 * rate), 1500)
    s1 += tik + s2
    s1 = limit(s1 * 2, 1, -1)
    tik2 = kick(freq / 2, 60 / bpm / 2)[::2] * 0.4
    s1 = distortion(maximize(s1), 0.2, 0.8) * 0.6
    s1 += tik2
    return s1 # raw_kick2

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
    arr = np.random.random(len(array_in))
    return maximize(arr) # white noise

def kick(freq, duration):
    t = 10
    freq1 = freq / t
    freq = slide((freq1 * 20), (freq1), round(duration * rate * t), 18000)
    freq3 = slide((freq1 * 4), (freq1), round(duration * rate * t), 80000)
    m = build_melody(freq, duration * t, np.sin)
    m *= slide(1, 0, round(duration * rate * t), 20000)
    sub = build_melody(freq3, duration * t, np.sin)
    sub *= slide(0, 1, round(duration * rate * t), 30000)
    m += sub
    m *= slide(0, 1, round(duration * rate * t), 9000)[::-1]
    return m[::t] # kick

def bass_808(freq, duration):
    t = 1
    freq = slide((freq / t * 2), (freq / t), round(duration * rate * t), 80)
    m = build_melody(freq, duration * t, np.sin)
    m *= slide(1, 0, round(duration * rate * t), 10000)
    return m[::t] # bass_808

def hihat(duration):
    n = build_melody(1, duration, noise)
    n = highpass(n, 2000)
    n = bandgain(n, 5000, 3000, 1.5)
    return n * slide(0.5, 0, round(duration * rate), 1000) * slide(0, 1, round(duration * rate), 2000) # hihat

def crash(duration):
    n = build_melody(1, duration, noise)
    n = highpass(n, 1000)
    n = bandgain(n, 1000, 3000, 1.5)
    return n * slide(1, 0, round(duration * rate), 10000) # crash

def snare(freq, duration):
    t = 5
    freq = slide((freq / t * 8), (freq / t * 6), round(duration * rate * t), 8000)
    m = build_melody(freq, duration * t, np.sin)
    m *= slide(1, 0.6, round(duration * rate * t), 5000)
    m *= 1.2
    m = limit(m, 1, -1)
    n = build_melody(freq, duration * t, noise)
    n *= slide(1, 0.6, round(duration * rate * t), 7000)
    m += n
    m = maximize(m)
    m *= slide(1, 0, round(duration * rate * t), 8000)
    return m[::t] # snare

def subdrop(freq):
    freq = np.linspace(freq * 1.5, freq, round(60 / bpm * 4 * rate))
    a = build_melody(freq, 60 / bpm * 4, triangle, 1)
    return a # subdrop

def build_melody(freq, duration, func=sawtooth, volume=1) -> np.array: # 旋律
    if volume == 0:
        return np.array([0 for _ in range(round(duration * rate))])
    tone_wave = func(2 * np.pi * np.arange(round(duration * rate)) * freq / rate) * volume
    if len(tone_wave) == round(duration * rate):
        return tone_wave
    if len(tone_wave) < round(duration * rate):
        empty = np.array([0 for _ in range(round(duration * rate) - len(tone_wave))])
        tone_wave = np.append(tone_wave, empty)
        return tone_wave
    tone_wave = tone_wave[:round(duration * rate)]

    return tone_wave # build_melody

def sweep_up(freq):
    freq1 = np.linspace(freq, freq * 8, round(60 / bpm * 16 * rate))
    n = build_melody(freq1, 60 / bpm * 16, noise, 0.8)
    n += build_melody(freq1, 60 / bpm * 16, np.sin, 0.2)
    freq2 = np.linspace(0.2, 10, round(60 / bpm * 16 * rate))
    n *= 0.5 - build_melody(freq2, 60 / bpm * 16, np.sin, 0.5)
    return n # sweep_up

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
        res += build_melody(freq, duration, func, volume).astype(np.float16)
    res /= len(freq_l)
    return res # build_chord

def sample(fname):
    audio = AudioFileClip(fname).to_soundarray()
    audio = audio.T
    audio = audio[0]
    audio /= max(abs(audio))
    return audio # sample

def highpass(arr, freq):
    arr += 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq] *= 0
    arr1 = np.fft.ifft(fft_arr)
    arr1 = maximize(arr1)
    return arr1 # highpass

def lowpass(arr, freq):
    arr += 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= 0
    arr1 = np.fft.ifft(fft_arr)
    arr1 = maximize(arr1)
    return arr1 # lowpass

def highgain(arr, freq, times=5):
    arr += 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) > freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    return arr1 # highgain

def lowgain(arr, freq, times=5):
    arr += 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq] *= times
    arr1 = np.fft.ifft(fft_arr)
    arr1 = maximize(arr1)
    return arr1 # lowgain

def bandgain(arr, freq_h, freq_l, times=5):
    arr += 1
    fft_arr = np.fft.fft(arr)
    fft_freqs = np.fft.fftfreq(arr.size, 1 / rate)
    fft_arr[abs(fft_freqs) < freq_h] *= times
    fft_arr[abs(fft_freqs) < freq_l] /= times
    arr1 = np.fft.ifft(fft_arr)
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
        song = compressor(song)

    plt.plot(song)
    plt.show()

    return song # compile_tracks

def maximize(arr):
    arr = arr - (max(arr) + min(arr)) / 2
    arr /= max(abs(arr))
    return arr # maximize

n = maximize(np.random.random(100000)) * slide(1, 0, 100000, 80000) * slide(0, 0.6, 100000, 1200) # IR

def reverb(x, dry=0.5):
    x1 = deepcopy(x)
    n[-1] = 0
    print("convolving")
    x = np.convolve(lowpass(x, 18000), n)[:len(x1)]
    x = maximize(x)
    print("finish")
    x = x1 * dry + x * (1 - dry)
    return x # reverb

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

def kickstart(x):
    s = build_melody(1 / (60 / bpm), len(x) / rate * 1.2)[:len(x)]
    s += 1
    s *= 2
    s = limit(s, 1, 0)
    s = distortion(s, 0.6, 0.4)
    x *= s
    return x # kickstart
    
def eff(func, **args):
    def f(arr):
        return func(arr, **args)
    return f # eff

def eff_chain(*l):
    def f(arr):
        for e in l:
            arr = e(arr)
        return arr
    return f # eff_chain

def compressor(x):
    a = 1
    for i in range(len(x)):
        if abs(x[i]) > a:
            a = abs(x[i])
        else:
            if a > 1:
                a *= 0.99
        i /= a
    return x # compressor


climax = compile_tracks(
    [
        [
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("A5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("A5"), 60 / bpm / 2, hardlead),
            
            build_melody(note("F#5"), 60 / bpm / 2, hardlead),
            build_melody(note("F#5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("A#5"), 60 / bpm / 2, hardlead),
            build_melody(note("A#5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("F#6"), 60 / bpm / 2, hardlead),

            
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("A5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("A5"), 60 / bpm / 2, hardlead),
            
            build_melody(note("F#5"), 60 / bpm / 2, hardlead),
            build_melody(note("F#5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("A#5"), 60 / bpm / 2, hardlead),
            build_melody(note("A#5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("D6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead),
            build_melody(note("C#6"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("B5"), 60 / bpm / 2, hardlead, 0),
            build_melody(note("B5"), 60 / bpm / 2, hardlead),
            build_melody(note("A5"), 60 / bpm / 2, hardlead),

        ],
        [
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            

            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("G4"), note("B4"), note("D5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord),
            build_chord([note("F#4"), note("A#4"), note("D5")], 60 / bpm / 2, hardchord, 0),
            
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            build_chord([note("B4"), note("D5"), note("F#5")], 60 / bpm / 2, hardchord),
            
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord, 0),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
            build_chord([note("A4"), note("C#5"), note("E5")], 60 / bpm / 2, hardchord),
        ],
        [
            build_melody(0, 60 / bpm / 2, sin, 0),
            raw_tail(note("G2"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),

            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 2),
            raw_kick2(note("C2")),


            build_melody(0, 60 / bpm / 2, sin, 0),
            raw_kick2(note("G2")),
            raw_kick2(note("E2")),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("G2"), 60 / bpm / 4 * 3),
            
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("F#2"), 60 / bpm / 4 * 3),
            
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("B2"), 60 / bpm / 4 * 3),

            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 4 * 3),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 2),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),
            raw_tail(note("A2"), 60 / bpm / 4),
            raw_kick2(note("C2")),
            raw_kick2(note("C2")),

        ]
    ],
    [0.35, 0.25, 0.4], # 精细（？）音量
    #[0, 0, 1], # only kicks
    [eff_chain(reverb, kickstart, maximize), eff_chain(reverb, kickstart, maximize), eff_chain(eff(reverb, dry=0.9), maximize)], # 细致（雾🌫️）混音
    #[lambda X:X, lambda X:X, lambda X:X], # 快速预览
    maximize # 极简母带
    )
"""
climax = compile_tracks(
[
[
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),

raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),

raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),

raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
raw_kick2(note("C2")),
raw_tail(note("C2"), 60 / bpm / 4 * 3),
]
],
[1],
[eff(reverb, dry=0.9)]
)"""
song = climax
song = (song * 32767).astype(np.int16)
#              ^^^^^ 淦，之前写的都是1024
plt.plot(song)
plt.show()

# 写入wav
import wave

fname = "song.wav"
with wave.open(fname, 'wb') as f_wav:
    f_wav.setnchannels(channels)
    f_wav.setsampwidth(2)
    f_wav.setframerate(rate)
    f_wav.writeframes(song.tobytes())

# 播放wav
os.system(f"start {fname}")
