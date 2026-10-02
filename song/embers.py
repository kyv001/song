"""《余烬》/ Embers —— 用 engine.py 写的一首曲子

    cd song && python embers.py        # 渲染 -> embers.wav
    python analyze.py embers.wav       # 客观指标 + 出图

概念：一堆看着已经冷掉的灰，里面有火。全曲六个段落走完「冷 -> 温 -> 复燃 ->
暗涌 -> 燎原 -> 归寂」，调性中心是 D，但色彩一直在换：

    灰烬 / 暗涌    D 弗里吉亚 —— ♭2（E♭）是「冷」的颜色，E♭maj9 就是那股寒气
    余温 / 复燃    D 多利亚   —— 大六度（B♮）是「暖」的颜色，G13 是它的签名和弦
    燎原           ♭VI–♭VII–i —— B♭ – C – D，最直白也最有效的「史诗终止」
    归寂           Picardy    —— 最后落到 D 大三（F♯），灰里透出光

和声全部写成数据（VOICINGS + 段落里的进行表），旋律写成 (音名, 拍数) 的列表。
`note()` 是准的，写 D 就是 D（不像 song.py 里那套老引擎整体高三个半音）。

结构（124 BPM，共 104 小节 ≈ 3 分 21 秒 + 混响尾巴）：

    段              小节   内容
    I   灰烬        16    只有铺底、人声垫、钟、低频呼吸。没有鼓
    II  余温        16    半拍底鼓、玻璃琶音、低音进来；主题 A 的前半句露头
    III 复燃        16    四踩底鼓 + 303 酸性低音 + 16 分钉钉；主题 A 全奏
    IV  暗涌         8    抽掉鼓组，弗里吉亚的寒气回来，riser 与军鼓滚奏拉起张力
    V   燎原        32    全奏。前 16 小节多利亚 vamp，后 16 小节 ♭VI–♭VII–i 圣歌
    VI  归寂        16    鼓组退出，钟与琶音回响，最后定格在 D 大三和弦
"""

from __future__ import annotations

import numpy as np

import engine as E

BEAT, BAR = E.set_tempo(124.0)
SR = E.SR
note, chord = E.note, E.chord

#  段落表（analyze.py 也读这个来分段统计）
STRUCTURE = [("I.灰烬", 16), ("II.余温", 16), ("III.复燃", 16),
             ("IV.暗涌", 8), ("V.燎原", 32), ("VI.归寂", 16)]


# =============================================================================
#  和声
# =============================================================================

#  每个和弦写成显式的排列（不是「根音 + 三度 + 五度」机械堆叠），
#  相邻和弦尽量保持共同音、让内声部级进——这就是为什么 B♭maj9 写成
#  B♭2 D3 F3 A3 C4：从 Dm9 的 D3 F3 A3 E4 过来，三个音不动，E 落到 C。
VOICINGS = {
    # D 小调 / 多利亚
    "Dm9":    ["D3", "F3", "A3", "C4", "E4"],
    "Dm9open": ["D3", "F3", "A3", "E4"],
    "Dm11":   ["D3", "F3", "G3", "C4", "E4"],
    "Dmadd9": ["D3", "F3", "A3", "E4", "A4"],
    "Dm":     ["D3", "F3", "A3"],
    "Dm_F":   ["F2", "A2", "D3", "F3"],
    # 弗里吉亚的寒气
    "Ebmaj9": ["Eb3", "G3", "Bb3", "D4", "F4"],
    "Ebmaj7": ["Eb3", "G3", "Bb3", "D4"],
    # ♭VI / ♭VII
    "Bbmaj9": ["Bb2", "D3", "F3", "A3", "C4"],
    "Bbmaj7s11": ["Bb2", "D3", "F3", "A3", "E4"],
    "Bbmaj":  ["Bb2", "D3", "F3", "A3"],
    "Cadd9":  ["C3", "E3", "G3", "D4"],
    "C6_9":   ["C3", "E3", "A3", "D4"],
    # iv / v / V
    "Gm9":    ["G2", "Bb2", "D3", "F3", "A3"],
    "G13":    ["G2", "B2", "D3", "F3", "A3", "E4"],
    "A7sus4": ["A2", "D3", "E3", "G3"],
    "A7":     ["A2", "C#3", "E3", "G3"],
    "A7b9":   ["A2", "C#3", "E3", "G3", "Bb3"],
    # 归寂的 Picardy
    "Dmaj9":  ["D3", "F#3", "A3", "C#4", "E4"],
    "Dmaj":   ["D3", "F#3", "A3", "D4"],
}

#  各段的进行表：(和弦名, 占几小节)
PROG_ASHES = [("Dmadd9", 4), ("Ebmaj9", 2), ("Bbmaj9", 2),
              ("Dm9", 2), ("Gm9", 2), ("A7sus4", 2), ("A7", 2)]

PROG_WARM = [("Dm9", 2), ("Bbmaj9", 2), ("Gm9", 2), ("A7", 2),
             ("Dm9", 2), ("Bbmaj7s11", 2), ("Gm9", 2), ("A7sus4", 2)]

PROG_REKINDLE = [("Dm9", 1), ("G13", 1), ("Dm9", 1), ("Bbmaj9", 1),
                 ("G13", 1), ("Cadd9", 1), ("Dm9", 1), ("A7", 1)] * 2

PROG_UNDERTOW = [("Ebmaj9", 2), ("Bbmaj9", 2), ("Gm9", 2), ("A7sus4", 2)]

PROG_BLAZE_A = [("Dm9", 1), ("G13", 1), ("Bbmaj9", 1), ("Cadd9", 1)] * 4

PROG_BLAZE_B = [("Bbmaj9", 2), ("Cadd9", 2), ("Dm9", 2), ("Dm9", 2),
                ("Bbmaj9", 2), ("Cadd9", 2), ("Dm9", 2), ("A7b9", 2)]

PROG_EMBERS = [("Dm9", 2), ("Bbmaj9", 2), ("Gm9", 2), ("Dm_F", 2),
               ("Ebmaj9", 2), ("Dm9", 2), ("Dmaj9", 2), ("Dmaj9", 2)]


def roots(prog):
    """把进行表拆成 [(根音名, 起点小节, 小节数), ...]。"""
    out, b = [], 0
    for cname, n in prog:
        v = VOICINGS[cname][0]
        out.append((v, b, n))
        b += n
    return out


#  低音一律取和弦根音下方八度，不做「折叠进某个频率窗口」的处理。
#  根音横跨 F2-B♭3，减八度落在 44-78 Hz，本来就是合理的 sub 音域；而把低于
#  某个阈值的音折上去，会让 Dm9 -> G13 这条下行四度的低音线变成上行五度——
#  低音走向被搞反，比多几赫兹的「不可闻低频」糟得多。


def chord_at(prog, bar):
    """第 bar 小节上是哪个和弦（进行表里各和弦占的小节数可以不一样）。"""
    b = 0
    for cname, n in prog:
        if bar < b + n:
            return cname
        b += n
    return prog[-1][0]


def up_oct(name):
    return name[:-1] + str(int(name[-1]) + 1)


def mid_voices(cname, num=3):
    """取和弦排列里靠上的 num 个音。

    它们一定都是和弦音，而且因为 VOICINGS 已经做过声部连接，听起来是连着的。
    人声垫、琶音这类「别弹错音」的声部一律从这里取，不要手写音名。
    """
    return VOICINGS[cname][-num:]


def add_choir(s, prog, gain_db, vowel="ah", bright=2200, a=2.0, r=1.6,
              rev=0.9, start_bar=0.0, num=3):
    """按进行表铺人声垫：每个和弦唱它自己的上方声部。"""
    b = start_bar
    for cname, n in prog:
        s.add(E.choir(chord(mid_voices(cname, num)), n * BAR + 0.3,
                      vowel=vowel, bright=bright, a=a, r=r),
              gain_db, 0.0, rev, name="choir", offset_bars=b)
        b += n


def arp_events(prog, bars, step, start_bar=0.0, note_dur=None, cycle_beats=1.0):
    """按进行表铺琶音：每 `cycle_beats` 拍在和弦的上方声部里换一个音。

    音高永远来自当前和弦，所以不会跟和声打架。
    """
    out = []
    if note_dur is None:
        note_dur = step * BEAT * 1.4
    k = 0
    beat = start_bar * 4
    end = (start_bar + bars) * 4
    while beat < end:
        vo = mid_voices(chord_at(prog, int(beat // 4)), 3)
        hi = [up_oct(x) for x in vo]
        nm = hi[int((beat - start_bar * 4) // cycle_beats) % len(hi)]
        out.append((beat * BEAT, (lambda n=nm: E.glass(note(n), note_dur))))
        beat += step
        k += 1
    return out


# ---------------------------------------------------------------------------
#  三度下方 / 上方：按字母级数走，落在 D 自然小调上
# ---------------------------------------------------------------------------

_SCALE_ACC = {"C": "", "D": "", "E": "", "F": "", "G": "", "A": "", "B": "b"}
_LETTERS = "CDEFGAB"


def shift_degree(name: str, steps: int) -> str:
    """把一个音名按字母级数移动 steps 级（-2 = 三度下方），落在 D 小调上。

    走的是「字母 + 音级」而不是半音数，所以得到的永远是和声音程：
    D 往下三度是 B♭（不是 B），B♮ 往下三度是 G，C♯ 往下三度是 A。
    """
    letter, acc, octv = name[0].upper(), name[1:-1], int(name[-1])
    i = _LETTERS.index(letter)
    j = i + steps
    octv += j // 7
    j %= 7
    return _LETTERS[j] + _SCALE_ACC[_LETTERS[j]] + str(octv)


def third_below(spec):
    return [(None if n is None else shift_degree(n, -2), b) for n, b in spec]


def sixth_below(spec):
    return [(None if n is None else shift_degree(n, -5), b) for n, b in spec]


# =============================================================================
#  主题（写成数据；每小节必须正好 4 拍）
# =============================================================================

#  主题 A：叹息式的动机（D–B♭–A），第 5 小节靠 B♮ 转到多利亚的暖色
THEME_A = [
    ("D5", 1), ("Bb4", 0.5), ("A4", 0.5), ("G4", 2),
    ("A4", 4),
    ("D5", 1), ("C5", 0.5), ("Bb4", 0.5), ("A4", 2),
    ("F4", 2), ("E4", 2),
    ("F4", 1), ("G4", 0.5), ("A4", 0.5), ("B4", 2),
    ("C5", 4),
    ("D5", 0.5), ("C5", 0.5), ("A4", 1), ("G4", 2),
    ("F4", 2), ("D4", 2),
]

#  主题 A 的前半句（1-4 小节，正好 16 拍）和后半句（5-8 小节）
THEME_A_FIRST = THEME_A[:11]
THEME_A2 = THEME_A[11:]

#  主题 B：燎原段，音区更高、更冲，第 4 小节冲到 G5
THEME_B = [
    ("A4", 0.5), ("Bb4", 0.5), ("D5", 1), ("F5", 2),
    ("E5", 2), ("D5", 2),
    ("C5", 1), ("D5", 0.5), ("E5", 0.5), ("F5", 2),
    ("G5", 4),
    ("F5", 0.5), ("E5", 0.5), ("D5", 1), ("C5", 2),
    #  这里必须落在 B♮（G13 的大三度）而不是 B♭，否则跟 G13 打小二度
    ("D5", 2), ("B4", 2),
    ("C5", 1), ("D5", 1), ("F5", 2),
    ("D5", 4),
]

#  主题 C：♭VI–♭VII–i 上的圣歌，两小节一个和弦
THEME_C = [
    ("F5", 2), ("D5", 1), ("E5", 1), ("F5", 4),
    ("E5", 2), ("G5", 2), ("F5", 4),
    ("F5", 2), ("E5", 2), ("D5", 4),
    ("F5", 1), ("E5", 1), ("D5", 2), ("A4", 4),
    ("D5", 2), ("F5", 2), ("E5", 4),
    ("G5", 2), ("F5", 2), ("E5", 4),
    ("F5", 2), ("D5", 2), ("A4", 4),
    ("C#5", 2), ("E5", 2), ("A4", 4),
]


# =============================================================================
#  编曲小工具
# =============================================================================


def ev(positions, make):
    """把「第几拍」的列表变成 (起始秒, 声音工厂) 事件列表。

    make(i) 收到第几个命中，返回音频；用它可以做重音 / 左右摆位的变化。
    """
    return [(p * BEAT, (lambda i=i: make(i))) for i, p in enumerate(positions)]


def melody(spec, voice, start_bar=0.0, dur_extra=0.0, dur_scale=1.0, **vkw):
    """(音名, 拍数) 的列表 -> 事件列表。None 表示休止。"""
    out = []
    t = (start_bar * 4) * BEAT
    for name, b in spec:
        d = b * BEAT
        if name is not None:
            out.append((t, (lambda n=name, dd=d: voice(
                note(n), dd * dur_scale + dur_extra, **vkw))))
        t += d
    return out


def breath(dur, cutoff=900.0, curve=1.2):
    """一段呼吸般的滤波噪声，氛围段用。"""
    n = E.nsamp(dur)
    return E.bpf(E.white(n), cutoff, 0.7) * E.swell(n, curve)


def floor_beats(bars, hits_per_bar=4):
    """每小节 hits_per_bar 次。4 = 四踩底鼓，2 = 半拍（落在 1、3 拍）。

    注意默认值是 4：这里原来写成 `hits=1.0`，于是「四踩底鼓」实际每小节只有
    一脚——III 段和 V 段的律动整个垮掉。是分轨波形图把这个错照出来的
    （Kick 那一格在应该满的地方是一条直线）。
    """
    step = 4.0 / hits_per_bar
    return [b * 4 + k * step
            for b in range(bars) for k in range(int(hits_per_bar))]


def backbeat_beats(bars):
    """二、四拍。"""
    return [b * 4 + k for b in range(bars) for k in (1, 3)]


def offbeat_beats(bars):
    """反拍八分（每拍的「和」）。"""
    return [b * 4 + k + 0.5 for b in range(bars) for k in range(4)]


def sixteenth_beats(bars):
    return [b * 4 + k * 0.25 for b in range(bars) for k in range(16)]


#  303 的十六分型： (小节内第几拍, 相对根音的半音数 或 None, 重音, 是否滑音)
ACID_PAT = [
    (0.00, 0, 1.0, False), (0.25, 0, 0.0, False), (0.50, None, 0.0, False),
    (0.75, 0, 0.55, False), (1.00, 12, 0.0, False), (1.25, None, 0.0, False),
    (1.50, 0, 0.75, False), (1.75, 0, 0.0, True),
    (2.00, 7, 0.55, False), (2.25, None, 0.0, False), (2.50, 0, 0.0, False),
    (2.75, 12, 0.45, False), (3.00, 0, 0.85, False), (3.25, 0, 0.0, True),
    (3.50, 12, 0.55, False), (3.75, 7, 0.35, False),
]


def acid_events(prog, start_bar=0.0, cut=False):
    """按进行表铺 303 低音：每小节一个根音，十六分型，重音 + 滑音。"""
    out = []
    for root_name, b, n in roots(prog):
        rf = note(root_name)
        for k in range(n):
            base = (start_bar + b + k) * 4 * BEAT
            prev = None
            for off, semi, accent, glide in ACID_PAT:
                if semi is None:
                    prev = None
                    continue
                f = rf * 2.0 ** (semi / 12.0)
                sf = prev if (glide and prev) else None
                out.append((base + off * BEAT,
                            (lambda ff=f, aa=accent, ss=sf: E.acid(
                                ff, BEAT * 0.27, base=190.0, env_amt=3.1 + aa,
                                q=7.5, decay=0.13, accent=aa, slide_from=ss,
                                res_glide=0.35))))
                prev = f
    return out


# =============================================================================
#  I. 灰烬 —— 只有空气、寒气和一个很远的声音
# =============================================================================


def build_ashes():
    s = E.Section("I.灰烬", 16, level_db=-5.0)
    prog = PROG_ASHES

    # 铺底：两小节一换，起落很慢
    b = 0
    for cname, n in prog:
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.4, bright=1050,
                    voices=4, drift=0.7, a=2.2, r=1.8, sub_oct=0.2),
              -9.0, 0.0, 1.0, name="pad", offset_bars=b)
        b += n

    # 人声垫从第 5 小节浮上来，唱的是每个和弦自己的上方声部
    add_choir(s, PROG_ASHES[1:], -20.0, vowel="ooh", bright=1850,
              a=2.6, r=2.0, start_bar=4)

    # 低音：只给根音，很慢的起音，像从很远的地方传过来
    for root_name, b, n in roots(prog):
        s.add(E.sub(note(root_name) / 2, n * BAR * 0.92, a=1.4, r=0.9),
              -16.0, 0.0, 0.15, name="sub", offset_bars=b)

    # 钟：稀疏，左右错开，长混响。音高都核对过当下的和弦
    bells = [(0, "D5", -0.35), (2.5, "A4", 0.4), (4, "Eb5", -0.25),
             (5.5, "Bb4", 0.3), (9, "D5", 0.35), (11, "A4", -0.4),
             (12.5, "D5", 0.25), (15, "C#5", -0.3)]
    for bar_pos, name, pan in bells:
        s.add(E.fm_bell(note(name), BAR * 2.6, index=4.5, tau=0.5,
                        amp_tau=2.4), -14.0, pan, 0.95, name="bell",
              offset_bars=bar_pos)

    # 玻璃琶音：第 9 小节探头，只在缝隙里响；音高永远落在这个和弦里
    for k, (t0, a) in enumerate(arp_events(prog, 8, 1.5, start_bar=8)):
        s.add(a(), -23.0, 0.45 * (-1) ** k, 0.8, dly=0.5, name="arp",
              offset_bars=t0 / BAR)

    # 空气：几段呼吸噪声
    for bp in (0, 6, 12):
        s.add(breath(BAR * 2.5, 700), -24.0, 0.0, 0.6, name="breath",
              offset_bars=bp)
    s.add(E.impact(BAR * 1.2, freq=38.0), -18.0, 0.0, 0.4, offset_bars=0)
    s.add(E.subdrop(note("D3"), note("D2"), BAR * 2), -17.0, 0.0, 0.3,
          offset_bars=14)
    return s.finish()


# =============================================================================
#  II. 余温 —— 心跳一样的半拍底鼓，暖色第一次出现
# =============================================================================


def build_warm():
    s = E.Section("II.余温", 16, level_db=-3.5)
    prog = PROG_WARM

    b = 0
    for cname, n in prog:
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.3, bright=1450,
                    voices=5, drift=0.5, a=1.1, r=1.0, sub_oct=0.22),
              -9.5, 0.0, 0.85, name="pad", offset_bars=b)
        b += n

    # 半拍底鼓（1、3 拍），比四踩松弛
    s.add_seq(ev([b * 4 + k for b in range(16) for k in (0, 2)],
                 lambda i: E.kick(52.0, pitch_amt=4.6, amp_tau=0.28)),
              -11.5, 0.0, 0.05, name="kick")

    # 低音跟着根音走，比 pad 稍短（下沉一个八度，落在 44-78 Hz）
    for root_name, b, n in roots(prog):
        s.add(E.sub(note(root_name) / 2, n * BAR * 0.96, a=0.05, r=0.2),
              -16.5, 0.0, 0.0, name="sub", offset_bars=b)

    # 反拍闭镲，左右轻微摆动
    s.add_seq(ev(offbeat_beats(16),
                 lambda i: E.hat(tau=0.018) * (1.0 if i % 2 == 0 else 0.7)),
              -13.0, 0.28, 0.18, name="hat")

    # 边击：每小节第 3 拍，8 小节之后才有
    s.add_seq(ev([b * 4 + 2 for b in range(8, 16)], lambda i: E.rim()),
              -21.0, -0.3, 0.3, name="rim")

    # 玻璃琶音：十六分分解和弦，音高按当下的和弦取
    s.add_seq(arp_events(prog, 16, 0.25, cycle_beats=1.0), -22.0, 0.0, 0.5,
              dly=0.45, name="arp")

    # 后半段浮出人声垫，把「暖」推出来
    add_choir(s, PROG_WARM[4:], -22.0, vowel="ooh", bright=2000,
              a=2.2, r=1.8, start_bar=8)

    # 主题 A 的前半句，用柔和的拨弦音色，从第 9 小节起
    s.add_seq(melody(THEME_A_FIRST, E.saw_pluck, start_bar=8,
                     dur_extra=BEAT * 0.5, base=1100, env_amt=2.4, q=2.0,
                     decay=0.35, amp_tau=0.9, detune=0.006),
              -15.0, 0.15, 0.75, dly=0.35, name="lead")

    s.add(E.fm_bell(note("D5"), BAR * 3, index=3.5, amp_tau=2.8), -18.0, 0.3,
          0.9, name="bell", offset_bars=8)
    s.add(E.fm_bell(note("A4"), BAR * 3, index=3.5, amp_tau=2.8), -20.0, -0.35,
          0.9, name="bell", offset_bars=12)
    s.add(E.crash(2.2, tau=0.7), -26.0, -0.2, 0.6, name="crash", offset_bars=0)
    s.add(E.reverse_swell(BAR * 2, 4200), -20.0, 0.0, 0.5, name="rev",
          offset_bars=14)
    return s.finish()


# =============================================================================
#  III. 复燃 —— 四踩、303、十六分钉钉，主题 A 全奏
# =============================================================================


def build_rekindle():
    s = E.Section("III.复燃", 16, level_db=0.0)
    prog = PROG_REKINDLE

    duck = E.duck_env(16, 4, depth=0.42, release=0.11)

    # 铺底：一小节一个和弦，跟着底鼓呼吸
    b = 0
    for cname, n in prog:
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.2, bright=1700,
                    voices=5, drift=0.4, a=0.5, r=0.5, sub_oct=0.3),
              -13.0, 0.0, 0.7, name="pad", offset_bars=b, duck=duck)
        b += n

    s.add_seq(ev(floor_beats(16, 4),
                 lambda i: E.kick(55.0, dur=0.36, pitch_amt=4.8, click=0.5, amp_tau=0.12)),
              -13.5, 0.0, 0.0, name="kick")
    s.add_seq(ev(backbeat_beats(16), lambda i: E.clap()), -14.0, 0.0, 0.25,
              name="clap")

    # 十六分闭镲，重音落在反拍上（律动的「推」就在这里）
    s.add_seq(ev(sixteenth_beats(16),
                 lambda i: E.hat(tau=0.016) * (1.0 if i % 4 == 2 else 0.55)),
              -15.0, 0.2, 0.15, name="hat")
    # 开镲：每小节第 4 拍的反拍
    s.add_seq(ev([b * 4 + 3.5 for b in range(16)],
                 lambda i: E.hat(0.34, tau=0.15)), -21.0, -0.25, 0.25,
              name="openhat")

    # 低音 + 酸性低音线
    for root_name, b, n in roots(prog):
        s.add(E.sub(note(root_name) / 2, n * BAR * 0.96, a=0.02, r=0.14),
              -13.5, 0.0, 0.0, duck=duck, name="sub", offset_bars=b)
    s.add_seq(acid_events(prog), -15.0, 0.0, 0.12, name="acid", duck=duck)

    # 和弦 stab 落在反拍，给一点延迟
    stab_pos = [b * 4 + k for b in range(16) for k in (0.75, 2.75)]
    for i, p in enumerate(stab_pos):
        cname = chord_at(prog, int(p // 4))
        s.add(E.stab(chord(VOICINGS[cname]), BEAT * 0.45, bright=2600),
              -22.0, 0.35 * (-1) ** i, 0.4, dly=0.5, name="stab",
              offset_bars=p / 4)

    # 主音：主题 A 两遍，第二遍加厚
    s.add_seq(melody(THEME_A, E.supersaw, start_bar=0, dur_extra=BEAT * 0.25,
                     voices=7, detune=0.014, bright=5200),
              -15.0, 0.0, 0.55, dly=0.45, name="lead")
    s.add_seq(melody(THEME_A, E.supersaw, start_bar=8, dur_extra=BEAT * 0.25,
                     voices=7, detune=0.02, bright=6200),
              -14.0, 0.0, 0.55, dly=0.45, name="lead")
    s.add_seq(melody(third_below(THEME_A), E.supersaw, start_bar=8,
                     dur_extra=BEAT * 0.25, voices=5, detune=0.012,
                     bright=4200),
              -19.0, -0.25, 0.6, dly=0.4, name="harmony")

    s.add(E.crash(2.4, tau=0.75), -18.0, -0.15, 0.6, name="crash", offset_bars=0)
    s.add(E.crash(2.4, tau=0.75), -19.0, 0.25, 0.6, name="crash", offset_bars=8)
    s.add_seq(ev([15 * 4 + k * 0.25 for k in range(16)],
                 lambda i: E.snare(200 + i * 8, 0.2) * (0.4 + 0.6 * i / 15)),
              -24.0, 0.0, 0.4, name="roll")
    return s.finish()


# =============================================================================
#  IV. 暗涌 —— 寒气回来，张力拉到顶
# =============================================================================


def build_undertow():
    s = E.Section("IV.暗涌", 8, level_db=-3.5)
    prog = PROG_UNDERTOW

    b = 0
    for cname, n in prog:
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.3, bright=1250,
                    voices=5, drift=0.8, a=1.6, r=1.4, sub_oct=0.18),
              -8.5, 0.0, 1.0, name="pad", offset_bars=b)
        b += n

    add_choir(s, PROG_UNDERTOW, -15.5, vowel="ah", bright=2500, a=2.2, r=1.8)

    # 低频：一个慢慢涨上来的 D 踏板
    s.add(E.sub(note("D2"), BAR * 8, a=3.0, r=1.5, harm=0.2), -16.0, 0.0, 0.2,
          name="sub")
    s.add(E.sub(note("Eb2"), BAR * 4, a=1.0, r=1.0), -20.0, 0.0, 0.2,
          name="sub", offset_bars=0)

    # 心跳底鼓：只有第一拍
    s.add_seq(ev([b * 4 for b in range(8)],
                 lambda i: E.kick(52.0, pitch_amt=4.0, amp_tau=0.35)), -14.5, 0.0, 0.0,
              name="kick")

    # 钟 + 玻璃点缀
    for bp, nm, pan in ((0, "Eb5", 0.35), (2, "D5", -0.3), (4, "A4", 0.4),
                        (5.5, "C5", -0.35)):
        s.add(E.fm_bell(note(nm), BAR * 3, index=4.0, amp_tau=2.6), -15.5, pan,
              0.95, name="bell", offset_bars=bp)

    # riser：最后两小节，把水重新拉高
    s.add(E.noise_riser(BAR * 2, 300, 13000, q=1.2, curve=1.5), -17.0, 0.0,
          0.5, name="riser", offset_bars=6)
    s.add(E.reverse_swell(BAR * 2, 6000), -16.0, 0.35, 0.4, name="rev",
          offset_bars=6)
    # 军鼓滚奏：从十六分加密
    roll_pos = []
    for k in range(16):
        roll_pos.append(6 * 4 + k * 0.25)
    roll_pos += [7 * 4 + k * 0.125 for k in range(32)]
    s.add_seq(ev(roll_pos, lambda i: E.snare(210, 0.16) * (0.35 + 0.65 * i / len(roll_pos))),
              -22.0, 0.0, 0.45, name="roll")

    s.add(E.impact(BAR * 1.5, freq=40.0), -17.0, 0.0, 0.35, name="impact",
          offset_bars=6)
    s.add(E.subdrop(note("A2"), note("A1"), BAR * 2), -18.0, 0.0, 0.3,
          name="subdrop", offset_bars=6)
    return s.finish()


# =============================================================================
#  V. 燎原 —— 全奏。前 16 小节多利亚 vamp，后 16 小节 ♭VI–♭VII–i
# =============================================================================


def build_blaze():
    s = E.Section("V.燎原", 32, level_db=3.0)
    pa, pb = PROG_BLAZE_A, PROG_BLAZE_B
    allprog = pa + pb
    duck = E.duck_env(32, 4, depth=0.4, release=0.1)

    # 铺底：整段一小节一个和弦，四个一循环，重复四次
    cycle = [E.pad(chord(VOICINGS[c]), BAR + 0.15, bright=2100, voices=5,
                   drift=0.35, a=0.28, r=0.28, sub_oct=0.35) for c, _ in pa]
    for k in range(4):
        s.add_seq([((k * 4 + i) * BAR, cycle[i]) for i in range(4)],
                  -12.0, 0.0, 0.6, name="pad", duck=duck)
    b = 0
    for cname, n in pb:
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.15, bright=2200,
                    voices=5, drift=0.3, a=0.4, r=0.4, sub_oct=0.35),
              -11.5, 0.0, 0.6, name="pad", offset_bars=16 + b, duck=duck)
        b += n

    # 鼓组
    s.add_seq(ev(floor_beats(32, 4),
                 lambda i: E.kick(55.0, dur=0.34, pitch_amt=5.0, click=0.6, amp_tau=0.11)),
              -12.0, 0.0, 0.0, name="kick")
    s.add_seq(ev(backbeat_beats(32), lambda i: E.clap()), -11.0, 0.0, 0.2,
              name="clap")
    s.add_seq(ev(sixteenth_beats(32),
                 lambda i: E.hat(tau=0.014) * (1.0 if i % 4 == 2 else 0.5)),
              -12.0, 0.25, 0.12, name="hat")
    s.add_seq(ev([b * 4 + 3.5 for b in range(32)],
                 lambda i: E.hat(0.3, tau=0.13)), -19.0, -0.3, 0.2, name="openhat")
    # 后半段加 ride 八分
    s.add_seq(ev([16 * 4 + k * 0.5 for k in range(128)],
                 lambda i: E.ride(1.1, tau=0.42) * (0.8 if i % 2 else 1.0)),
              -21.0, 0.3, 0.25, name="ride")

    # 低音
    for root_name, b, n in roots(allprog):
        s.add(E.sub(note(root_name) / 2, n * BAR * 0.96, a=0.015, r=0.14),
              -13.0, 0.0, 0.0, name="sub", offset_bars=b, duck=duck)
    s.add_seq(acid_events(allprog), -12.0, 0.0, 0.1, name="acid", duck=duck)

    # 主音与和声
    s.add_seq(melody(THEME_B, E.supersaw, start_bar=0, dur_extra=BEAT * 0.2,
                     voices=7, detune=0.018, bright=6000),
              -11.0, 0.0, 0.5, dly=0.4, name="lead")
    s.add_seq(melody(third_below(THEME_B), E.supersaw, start_bar=8,
                     dur_extra=BEAT * 0.2, voices=5, detune=0.013, bright=4600),
              -17.0, -0.3, 0.55, dly=0.35, name="harmony")
    s.add_seq(melody(THEME_C, E.supersaw, start_bar=16, dur_extra=BEAT * 0.25,
                     voices=7, detune=0.02, bright=6600),
              -10.0, 0.0, 0.5, dly=0.4, name="lead")
    s.add_seq(melody(third_below(THEME_C), E.supersaw, start_bar=16,
                     dur_extra=BEAT * 0.25, voices=5, detune=0.014,
                     bright=5200),
              -16.0, -0.3, 0.55, dly=0.35, name="harmony")
    # 钟在高八度跟着圣歌，把顶端照亮
    s.add_seq(melody(THEME_C, E.fm_bell, start_bar=16, dur_extra=BEAT * 1.2,
                     index=2.6, tau=0.35, amp_tau=1.6),
              -23.0, 0.4, 0.9, dly=0.5, name="bell")

    # 人声垫在后半段加入，唱每个和弦的上方声部
    add_choir(s, pb, -17.0, vowel="ah", bright=2600, a=1.4, r=1.2,
              start_bar=16)

    # 和弦 stab 落在每小节的反拍
    for i, p in enumerate([b * 4 + k for b in range(32) for k in (0.75, 2.75)]):
        cname = chord_at(allprog, int(p // 4))
        s.add(E.stab(chord(VOICINGS[cname]), BEAT * 0.4, bright=3000),
              -20.0, 0.4 * (-1) ** i, 0.35, dly=0.45, name="stab",
              offset_bars=p / 4)

    # 镲与过门
    for bar_pos in (0, 8, 16, 24):
        s.add(E.crash(2.6, tau=0.8), -13.0, -0.2 + 0.4 * (bar_pos % 16 == 0),
              0.6, name="crash", offset_bars=bar_pos)
    for bar_pos in (7, 15):
        s.add_seq(ev([bar_pos * 4 + k * 0.25 for k in range(16)],
                     lambda i: E.snare(210, 0.16) * (0.4 + 0.6 * i / 15)),
                  -20.0, 0.0, 0.4, name="roll")
    # 后半段收尾的嗵鼓过门
    for i, p in enumerate([30 * 4 + 0, 30 * 4 + 1, 30 * 4 + 2, 30 * 4 + 3,
                           31 * 4 + 0, 31 * 4 + 0.5, 31 * 4 + 1, 31 * 4 + 1.5,
                           31 * 4 + 2, 31 * 4 + 2.5, 31 * 4 + 3, 31 * 4 + 3.5]):
        s.add(E.tom(note("D2") * (1.5 ** (i % 4)), 0.4), -18.0,
              0.4 * (-1) ** i, 0.4, name="tom", offset_bars=p / 4)
    s.add(E.impact(BAR * 1.5, freq=45.0), -12.0, 0.0, 0.3, name="impact",
          offset_bars=16)
    return s.finish()


# =============================================================================
#  VI. 归寂 —— 灰里透出光（Picardy）
# =============================================================================


def build_embers():
    s = E.Section("VI.归寂", 16, level_db=-4.5)
    prog = PROG_EMBERS

    b = 0
    for cname, n in prog:
        bright = 1500 if cname != "Dmaj9" else 2100
        s.add(E.pad(chord(VOICINGS[cname]), n * BAR + 0.5, bright=bright,
                    voices=5, drift=0.6, a=1.8, r=1.8, sub_oct=0.2),
              -11.0, 0.0, 1.0, name="pad", offset_bars=b)
        b += n

    add_choir(s, PROG_EMBERS, -18.0, vowel="ooh", bright=2100, a=2.4, r=2.0,
              start_bar=2)

    for root_name, b, n in roots(prog):
        s.add(E.sub(note(root_name) / 2, n * BAR * 0.9, a=1.2, r=1.0),
              -16.0, 0.0, 0.2, name="sub", offset_bars=b)

    # 主题 A 的后半句，从第 1 小节起（Dm9 / B♭maj9 上面正好合）
    s.add_seq(melody(THEME_A2, E.saw_pluck, start_bar=0, dur_extra=BEAT,
                     base=900, env_amt=2.6, q=2.4, decay=0.5, amp_tau=1.4,
                     detune=0.005),
              -16.0, 0.0, 0.8, dly=0.4, name="lead")
    # 钟把主题第一句的影子散在左右
    for bp, nm, pan in ((0, "D5", -0.35), (1.5, "A4", 0.3), (3, "F5", -0.3),
                        (4.5, "E5", 0.35), (6, "D5", -0.25), (9, "C5", 0.3),
                        (11, "A4", -0.3), (13, "F#5", 0.3), (14.5, "D5", -0.2)):
        s.add(E.fm_bell(note(nm), BAR * 3.2, index=4.0, tau=0.5, amp_tau=2.8),
              -15.0, pan, 1.0, name="bell", offset_bars=bp)

    s.add_seq(arp_events(prog, 4, 0.5, start_bar=8, cycle_beats=1.0),
              -22.0, 0.35, 0.7, dly=0.5, name="arp")
    s.add(E.impact(BAR * 1.5, freq=36.0), -20.0, 0.0, 0.35, name="impact",
          offset_bars=0)
    return s.finish()


# =============================================================================
#  渲染
# =============================================================================


#  分轨分组：把 22 个声部名归成 13 组，喂给 stems.py / visualize.py。
#  约束是可视化那 3x5 的网格——13 格分轨 + 2 格合并的 Master。
#  归组的依据是「看波形的时候你想区分什么」：音色层（铺底/人声/钟/琶音）、
#  节奏层（底鼓/军鼓/拍手/钉钉/镲）、低音层（sub/acid）、主音层、音效层。
STEM_GROUPS = {
    "Pad":       ("pad",),
    "Choir":     ("choir",),
    "Sub Bass":  ("sub",),
    "Acid Bass": ("acid", "stab"),
    "Kick":      ("kick",),
    "Snare":     ("roll", "rim"),
    "Clap":      ("clap",),
    "Hats":      ("hat", "openhat"),
    "Cymbals":   ("crash", "ride"),
    "Bells":     ("bell",),
    "Arp":       ("arp",),
    "Lead":      ("lead", "harmony"),
    "FX":        ("impact", "subdrop", "riser", "rev", "tom", "breath"),
}
#  顺序就是可视化网格里从左到右、从上到下的顺序
STEM_ORDER = ["Pad", "Choir", "Sub Bass", "Acid Bass", "Kick", "Snare", "Clap",
              "Hats", "Cymbals", "Bells", "Arp", "Lead", "FX"]


def build_parts():
    """按顺序构造六个段落。

    渲染（main）和分轨导出（stems.py）都走这里，保证两边是同一份编曲、
    同一个随机数消耗顺序，所以分轨和成品严格对得上。
    """
    return [build_ashes(), build_warm(), build_rekindle(), build_undertow(),
            build_blaze(), build_embers()]


def main():
    parts = build_parts()

    print("段落：" + " | ".join(f"{p.name} {p.bars:.0f}小节" for p in parts))
    total = sum(p.bars for p in parts)
    print(f"合计 {total:.0f} 小节，{total * BAR:.1f} 秒（不含混响尾巴）")

    mix, _ = E.mixdown(parts, tail_bars=5.0, rev_amt=0.75, dly_amt=0.6,
                       delay_beat=0.75, delay_fb=0.42, delay_mix=0.5,
                       master_hpf=28.0, master_air=(3000.0, 3.5),
                       master_low=(65.0, -1.5), master_width=1.18, master_clip=0.72,
                       comp_thresh=-18.0, comp_ratio=2.0, comp_makeup=2.5,
                       fade_in_bars=0.5, fade_out_bars=3.0)

    E.write_wav("embers.wav", mix)
    dur = mix.shape[1] / SR
    print(f"\n已写出 embers.wav：{dur:.1f} 秒 / {SR} Hz / 立体声")
    print("峰 {:.4f}  RMS {:.4f}  波峰因数 {:.1f} dB  左右相关 {:.3f}  削顶 {}".format(
        E.peak(mix), E.rms(mix), E.crest_db(mix), E.correlation(mix),
        int(np.sum(np.abs(mix) > 0.999))))
    return mix


if __name__ == "__main__":
    main()
