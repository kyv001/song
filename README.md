# song

用 numpy 和 Python 自带的 `wave` 库写歌的小程序。合成器、效果器、编曲全在
[`song/song.py`](song/song.py) 一个文件里，编曲靠硬编码，渲染是纯 CPU 的，
两分钟的曲子大约要跑几分钟。

> **给后来的 agent**
>
> **这是个 playground。** 整个仓库没有任何生产用途，唯一的目标是把曲子写得更好听。
> 引擎、音色、效果、依赖、文件结构——**全都可以改**：删掉不用的函数、
> 把单文件拆成模块、换掉整个合成器架构、上 numba 把那些逐采样循环干掉，
> 都随便。不用管向后兼容、不用管可维护性、不用管别人的代码风格，
> 也不用为了「尊重原作者」而绕路（原作者就是当年写它的那个小孩，他只会高兴）。
> 改坏了 `git checkout` 就回来了。
>
> 所以下面这份文档的定位是**地图**——「哪里有什么、哪里会塌」，
> 不是「什么不许动」。尤其 **「一、会咬人的地方」**：那是当前实现的真实行为
> （我自己踩过的坑），不是设计规范。想改掉它们完全可以，只要知道改了会发生什么。
>
> - 想换一首曲子 → 整个编曲层（`compile_section` 那段）删掉重写
> - 想加音色 → 写个 `相位 -> 波形` 的函数传给 `build_note`
> - 想动引擎底层 → 请便，「会咬人的地方」那节就是你的变更影响清单

---

## 目录

- [零、怎么跑](#零怎么跑)
- [一、会咬人的地方](#一会咬人的地方)
- [二、记谱：`note()` 整体高了三个半音](#二记谱note-整体高了三个半音)
- [三、四个概念](#三四个概念)
- [四、音色设计](#四音色设计)
- [五、编曲工作流](#五编曲工作流)
- [六、混音：用 RMS 说话](#六混音用-rms-说话)
- [七、渲染与验证](#七渲染与验证)
- [八、故障排查](#八故障排查)
- [九、现有作品](#九现有作品)

---

## 零、怎么跑

```bash
cd song
python song.py            # 默认渲染左声道 -> L.wav
SIDE=R python song.py     # 右声道 -> R.wav
python merge.py           # 合成立体声 song.wav
python plot.py            # 出 tide.png（波形 / 频谱 / 高潮放大）
```

- 必须在 `song/` 目录里跑，代码里用的是相对路径（`IR.wav`、`./long_noise_L.wav`）。
- 立体声是「渲染两遍」做的：`side` 决定读哪一路噪声和 IR，所以噪声类打击乐和
  混响在左右声道之间真正去相关。**改了编曲要两遍都重渲**。
- 依赖 numpy / scipy / matplotlib / moviepy。headless 环境设 `MPLBACKEND=Agg`。

---

## 一、会咬人的地方

下面每一条都是**当前代码的真实行为**，不是纪律。写了「必须」「别」的地方，
意思都是「这样做了会炸 / 会难听」，而不是「作者不同意」。想绕开、想重构、想改掉，
都行——知道代价就好。每一条都标了炸的方式，方便你决定值不值得修。

1. **一段里所有轨必须等长。** `compile_tracks` 用 `song += track` 累加，长度不一致
   直接 broadcast 报错。用 `mk_track(items, bars)` 把每条轨补齐/截断到
   `round(bars * bar * rate)`，别手工数。

2. ~~**FFT 滤波必须放在 `reverb` 之前。**~~ —— **这条已经作废了。**
   它曾经成立：`highpass/lowpass/…` 返回的是复数数组，一旦喂给它们 `reverb` 的
   输出，信息就落在虚部、再被 `astype(np.float16)` 丢掉。现在滤波器出口都取
   `.real`，两种顺序实测完全等价（峰值都是 0.724），随便排。
   原作者的链条仍是「先滤波 → 混响 → 增益/limiter」，照着写不会错，但不再是硬性要求。
   （这也是本仓库的典型情况：**文档里的「必须」往往是某个 bug 的影子，把 bug 修掉
   「必须」就没了**，别把它当祖训。）

3. **`compile_tracks` 从列表尾部取基准轨**（`tracks_l.pop()` / `volumes.pop()` /
   `effects.pop()`），所以 `tracks`、`volumes`、`effects` 三个列表必须严格平行、
   长度一致。

4. **`master` 之后有一道自动保险**：`if max(abs(song)) > 1: song = limiter(song)`，
   而 `limiter` 结尾会 `maximize`，也就是把整段顶到峰值 1.0。想让段落保持你指定的
   电平，就让 master 输出的峰值 **小于 1**。

5. **`maximize(arr)` 是原地修改**（`arr /= max(abs(arr))`），会改掉传进去的数组。
   别把同一个数组同时交给两个地方。

6. **`bpm` 是所有时值的基准**，且 `kick` / `psy_punch` / `psy_tail` 的长度都等于
   一个十六分音符。改 `bpm` 不用改别处（`short_noise` 已经跟着走了），但把
   `bpm` 改回 150 以上之外的值时要留意 `long_noise` 够不够长。

7. **`empty()` 返回的是 int64 全零**（`build_note` 里 `volume == 0` 走的是提前返回
   分支），不是 float。拼接时 numpy 会自动提升，但别依赖它的 dtype。

8. **这些函数是逐采样的 Python 循环**：`sawtooth`、`square`、`triangle`、
   `distortion`、`limiter`、`slide`、`scratch`、`declick`。给整段（几百万采样）
   套一个 `slide` 会卡到怀疑人生——包络请用 numpy 写（见 `swell()`）。

9. **`reverb` 会把湿信号单独 `maximize` 到峰值 1.0**，再按 `(1 - dry)` 混进来。
   也就是说不管输入多轻，混响的水位是固定的：安静的音色（琶音、分解和弦）
   会被混响淹掉。这类轨用 `dry=0.9` 左右，或者干脆不过混响。

10. **`build_note` 的相位是 `linspace(0, freq*2π*duration, length)`。** 如果
    `freq × duration` 不是整数，音符首尾对不上，会产生咔哒声——长音尤其明显。
    引擎里的 `declick()` 就是为了擦这个，但它本身也很慢。

11. **引擎自带的 `hihat()` 基本没声音**（实测峰值 0.046），`crash()` 衰减又太快
    （0.2 秒就没了）。编曲层的 `hat()` / `crash_wash()` 是替代品，用那两个最省事；
    想把引擎里那两个直接改好当然更彻底。

12. `_er()` 里递归调用的是 `_reverb`（疑似笔误），只有 `no_convolve=True` 才会走到；
    `fnoise()`、`scratch()`、`comb_filter()` 目前没人用。

---

## 二、记谱：`note()` 整体高了三个半音

```python
note("C5")  # 622.25 Hz  —— 听起来是 E♭5
note("A5")  # 1046.5 Hz  —— 听起来是 C6
note("E5")  # 784.0  Hz  —— 听起来是 G5
note("E2")  # 98.0   Hz  —— 听起来是 G2
note("E1")  # 49.0   Hz  —— 听起来是 G1
```

公式是 `440 × 2^((音级 + (八度-5)×12 + 3) / 12)`，那个 `+3` 是写死的
（源码注释写着「D#调」）。

> **听到的音 = 写下的音 + 3 个半音。**

所以想写 G 小调就记 **E 小调**，想写 C 小调就记 **A 小调**，以此类推。
《潮汐》就是记 E 小调、实际听感 G 小调。

常用音名对照（写 → 听）：

| 写 | Hz | 听 | 写 | Hz | 听 |
|---|---|---|---|---|---|
| `E1` | 49.0 | G1 | `C2` | 77.8 | E♭2 |
| `D2` | 87.3 | F2 | `E2` | 98.0 | G2 |
| `G2` | 116.5 | B♭2 | `E3` | 196.0 | G3 |
| `B3` | 293.7 | D4 | `E4` | 392.0 | G4 |
| `B4` | 587.3 | D5 | `E5` | 784.0 | G5 |
| `A5` | 1046.5 | C6 | `B5` | 1174.7 | D6 |

`note()` 只认「音名 + 一位八度」（`oct_ = int(n[-1]) - 5`）：

- 升降号要紧跟音名，且是倒数第二个字符：`"F#4"` ✓、`"Bb3"` ✓
- 八度必须是单个数字；`"E10"` 会被解析成八度 1 + 多余字符，算错
- 没有 `Cb` / `E#` / 微分音，别写

---

## 三、四个概念

### 1. 相位数组

`build_note(freq, duration, func, volume)` 负责造相位，然后 `func(相位) × volume`：

- `freq` 是**标量** → `linspace(0, freq·2π·duration, round(duration·rate))`
- `freq` 是**数组** → 逐采样累加 `freq[i]·2π/rate`，用来做滑音、音高包络
  （注意这条路径是 Python 循环，很慢）

所以在音色函数里，`x` 是**弧度制相位**，2π 一个周期：

```python
sin(x)                 # 正弦
x / (2 * np.pi)        # 第几个周期（浮点）
x - np.floor(x)        # 0-1 的相位，锯齿波的原料
sin(x * 2)             # 高八度
sin(x) + sin(x * 2)    # 叠一个八度
```

### 2. 音色函数

签名是 `f(相位数组) -> 波形数组`：长度必须和输入一致，幅度大致在 [-1, 1]。
它是纯函数，可以直接喂给 `build_note`，也可以喂给 `build_chord` 叠和弦。

### 3. 轨

一串首尾相接的波形数组。`mk_track(items, bars)` 负责拼成整段长度
（不足补零、超出截断）——**并且它顺便把 `np.append` 的 O(n²) 干掉了**：
先在 `mk_track` 里 `np.concatenate` 拼好，再以 `[[t] for t in tracks]` 的形式
交给 `compile_tracks`，每条轨只 append 一次。

### 4. 段

```python
compile_section(name, bars, tracks, volumes, effects, level)
```

若干条等长轨 → 各过效果链 → 乘音量 → 相加 → master（压限 + 归一）→ **再按 RMS
定到 `level`**。最后那一步是《潮汐》加的，见「六、混音」。

---

## 四、音色设计

### 六种做法

1. **加法（带限波形）** — 叠正弦谐波，天然不混叠。
   `lp_saw` 叠 30 次谐波、`lp_square` 只叠奇次、`lp_saw_nosub` 从 2 次开始
   （去掉基频，专门给失谐叠加用）：

   ```python
   def lp_saw(array_in):
       a = array_in * 0
       for _ in range(1, 31):
           a += sin(array_in * _) / _ * ((32 - _) / 31)   # 幅度按 1/k 衰减
       return maximize(a)
   ```

2. **波形整形（无带限）** — 直接对相位取模，混叠是音色的一部分：
   `sawtooth`（`x - int(x)`）、`square`（模 1 过半就翻转）、`triangle`。

3. **减法（砖墙滤波）** — `highpass / lowpass / highgain / lowgain / bandgain`
   都是「FFT → 把某些频点乘 0 或乘系数 → ifft」，没有共振、没有斜率，就是硬切。

   `bandgain(arr, freq_h, freq_l, times)` 的语义要读代码才明白：它先给所有
   `< freq_h` 的频点乘 `times`，再给所有 `< freq_l` 的频点除 `times`。
   当 `freq_l < freq_h`（多数调用）时净效果是**把 `[freq_l, freq_h]` 这一段
   提升 `times` 倍**，也就是一个窄带共振峰——`freq_h` 要写高频那个数。
   参数写反了就成了「挖掉一段」（`crash()` 里就是这么用的）。
   另外所有滤波器出口都带 `if max(abs(arr)) > 1: maximize(arr)`，
   所以 `times` 很大的调用（kick 里用到 300）实际是「窄带提取 + 归一到峰值 1」，
   而不是简单放大。

4. **失真（波形整形）** — `distortion(a, x, y)` 把 `[0, x]` 映到 `[0, y]`、
   `[x, 1]` 映到 `[y, 1]`，负半轴对称。**`x` 越小越脏**：
   `distortion(a, 0.9, 0.95)` 几乎透明，`distortion(a, 0.1, 0.8)` 是硬削波。
   它是逐采样循环，别用在整段上。

5. **失谐叠加** — 同一个音色按 ×1.01 / ×1.00 / ×0.99 复制几份相加，得到 supersaw
   那种厚度。`unison_saw`、`reese`、`strings` 都是这个套路：

   ```python
   a  = lp_saw_nosub(phase * 1.01) * 0.3
   a += lp_saw_nosub(phase * 1.00) * 0.3
   a += lp_saw_nosub(phase * 0.99) * 0.3
   ```

6. **噪声与包络** — `noise(x)` 返回白噪声（只看长度，不看内容）。
   包络优先用 numpy：

   ```python
   def swell(n, curve=1.0):                       # 两头归零，不爆音
       return np.sin(np.linspace(0, np.pi, n)) ** curve
   np.exp(-np.linspace(0, 1, len(a)) * 30)        # 打击乐的快速衰减
   ```

   引擎里的 `slide(start, end, length, rate)` 是「指数逼近」包络，rate 越大越快，
   但它是 Python 循环——短音符可以用，整段别用。

音高包络则是把**数组**当 `freq` 传给 `build_note`，比如 kick：

```python
slide(freq * 8, freq, round(beat / 4 * rate), 600)   # 600 采样时间常数的下滑
```

### 怎么快速试一个音色

引擎和声部定义都在编曲之前，用这个配方把它们单独 exec 出来（存成
`song/bench.py` 就能反复用）：

```python
import re, time
import numpy as np

src = open("song.py").read()
engine = re.split(r"\n\w+ = compile_section\(", src)[0]   # 砍掉编曲层
ns = {"__name__": "bench"}
exec(compile(engine, "song.py", "exec"), ns)

def rms(a):
    a = np.real(np.asarray(a, dtype=np.float64))
    print("%.3f 峰 %.4f RMS" % (np.max(np.abs(a)), np.sqrt(np.mean(a ** 2))))

t = time.time()
rms(ns["build_note"](ns["note"]("E5"), ns["beat"], ns["hardlead"]))
print("耗时 %.0f ms" % ((time.time() - t) * 1000))
```

**在 `song/` 目录里跑**（要读 `IR.wav`）。判断标准：
峰值落在 0.5~1.0 之间最省事，太低说明滤波或包络把电平吃掉了。

### 已有音色速查表

峰/RMS 是在 140 BPM、单位音量下实测的，用来反推混音音量。

| 音色 | 峰 | RMS | 说明 | 耗时 |
|---|---|---|---|---|
| `kick(freq)` | 1.000 | 0.491 | 音高包络正弦 + 噪声 click，长一个十六分 | 快 |
| `psy_punch(freq)` | 1.000 | 0.348 | psy 低音的重拍：click + sub + 共振峰 | 中 |
| `psy_tail(freq)` | 1.000 | 0.671 | psy 低音的滚动：三个衰减的十六分 | 中 |
| `snare(freq, dur)` | 0.955 | 0.304 | 军鼓，`freq` 决定鼓皮音高 | 中 |
| `hat(dur)` | 0.875 | 0.055 | **编曲层自制**，高通噪声 + 极快衰减 | 快 |
| `crash_wash(dur)` | 1.157 | 0.159 | **编曲层自制**，长尾镲 | 快 |
| `hihat(dur)` | 0.046 | 0.008 | 引擎自带，起音太慢，**基本听不见** | 快 |
| `crash(dur)` | 0.906 | 0.065 | 引擎自带，0.2 秒就衰减完 | 中 |
| `impact(dur)` | 0.685 | 0.073 | 低频轰鸣 | 中 |
| `hardlead(相位)` | 0.918 | 0.316 | 六层 supersaw + 失真，主力主音 | 慢 |
| `hardchord(相位)` | 0.588 | 0.132 | 和弦版 hardlead（三音），**约 1.4 秒/拍** | 很慢 |
| `unison_saw(相位)` | 1.000 | 0.274 | 五路失谐锯齿，柔和主音 | 中 |
| `strings(相位)` | 0.843 | 0.201 | 五路失谐 + 慢起慢落，铺底 | 慢 |
| `pluck(相位)` | 1.780 | 0.850 | 两个带音高弯折的正弦，拨弦感 | 很快 |
| `reese(相位)` | 1.000 | 0.446 | 三路无基频锯齿 + 正弦，经典 reese | 中 |
| `distorted_reese(相位)` | 1.000 | 0.678 | reese 过失真，更脏 | 中 |
| `subdrop(freq)` | 1.000 | 0.707 | 两拍的下滑正弦 | 快 |
| `sweep_up(freq)` | 1.000 | 0.287 | 十六拍的噪声 riser | 慢 |
| `build_note(f, d, sin)` | 1.000 | 0.707 | 纯正弦，当 sub 用 | 很快 |
| `build_chord(…, lp_saw)` | 0.994 | 0.362 | 和弦铺底（三音，四小节） | 快~中 |

其他没测但能用的：`raw_kick` / `raw_kick2` / `raw_kick3`（更硬的 kick）、
`punch1` / `punch3` / `punch_sub`（kick 的零件）、`laser_kick`、`frenchcore_kick`、
`bass_808`、`FM_growl` / `FM_growl_oneshot`、`screech(f0, f1, dur)`、
`lp_saw` / `lp_saw2` / `lp_saw_nosub` / `lp_square` / `lp_square2`、
`bass2`、`unison_saw2` / `unison_saw_dirty` / `unison_square`、
`lead_saw1` / `lead_saw2` / `lead_pulse1`（hardlead 的零件）、
`raw_tail` / `raw_tail2` / `raw_tail3` / `lp_square_tail`（低音尾巴）。

**耗时等级**（140 BPM 下的大致量级）：快 ≈ 几毫秒，
中 ≈ 十~几十毫秒，慢 ≈ 百毫秒（都是逐采样 Python 循环的锅）。
一整首两分钟的曲子，两个声道各约 3~6 分钟。

---

## 五、编曲工作流

### 起点：拍和小节

```python
beat = 60 / bpm        # 一拍
bar = beat * 4         # 一小节
```

所有时值都写成 `beat * n`，别写秒——改 bpm 时才不用重算。

### 和声写成数据

```python
PROG = [("E", ["E","G","B"]), ("C", ["C","E","G"]),
        ("G", ["G","B","D"]), ("D", ["D","F#","A"])]

def chord_notes(ci, oct_=3):        # 三和弦 + 高八度根音
    root, tones = PROG[ci % len(PROG)]
    return [note(t + str(oct_)) for t in tones] + [note(root + str(oct_ + 1))]
```

### 旋律也写成数据

```python
THEME = [("B4", 0.5), ("E5", 0.5), ("G5", 1), ("F#5", 0.5), ...]   # (音名, 拍数)

lead_track(bars, THEME, start_bar=8, voice=hardlead)
```

一条八小节乐句就是一个 35 项的列表，改旋律 = 改数据，不用碰代码结构。
**写完记得数一遍每小节是不是正好 4 拍**——这是最容易犯又最难查的错。

### 和声声部可以算出来

不用手抄两遍旋律，用音阶度数生成三度下方：

```python
SCALE = ["E", "F#", "G", "A", "B", "C", "D"]      # 自然小调

def third_below(name):
    letter, octv = name[:-1], int(name[-1])
    j = SCALE.index(letter) - 2                   # 往下数两个音级 = 三度
    if j < 0:
        j += 7; octv -= 1
    return SCALE[j] + str(octv)
```

### 几个现成的轨构造器

| 函数 | 作用 |
|---|---|
| `four_floor(bars)` | 四踩底鼓（鼓音高固定在主音上） |
| `rolling_track(bars)` | psy 的 rolling bass：每拍一个 punch + 三个十六分 tail |
| `hat_track(bars, sixteenth=)` | 反拍八分 / 十六分 hi-hat |
| `backbeat(bars)` | 二、四拍军鼓 |
| `roll_bar()` | 一小节的军鼓渐密滚奏（4→8→16 分） |
| `arp_track(bars, div=)` | 和弦琶音，上下往返 |
| `pad_track(bars, seq=)` | 弦乐铺底，两小节一个和弦 |
| `sub_track(bars)` | 每两小节一个根音的纯正弦低音 |
| `reese_track(bars, hold=)` | 每 hold 小节换一次的 growl 低音 |
| `lead_track(bars, spec, start_bar=)` | 把旋律数据变成轨 |
| `harmony_track(bars, spec, start_bar=)` | 同上，但走三度下方 |
| `grid(bars, [(小节号, 数组), …])` | 把 crash / riser / 滚奏摆到小节网格上，允许重叠相加 |
| `duck(bars)` | 整段的侧链包络，让 pad / 琶音 / 主音跟着底鼓呼吸 |
| `mk_track(items, bars)` | 拼轨并强制等长 |

**侧链**用 `eff(times, duck(bars))` 挂在效果链最后：

```python
eff_chain(eff(highpass, 200), eff(reverb, 0.85), eff(times, duck(24)))
```

`duck()` 铺的是引擎里那个 `sidechain` 全局，形状是**一拍长**的包络：
拍头为 0，到该拍 1/4 处升到 1，之后一直是 1。也就是说它把每拍**第一个十六分**
让给底鼓，正是侧链该有的样子（而底鼓就长一个十六分，对得严丝合缝）。

> 这个形状来自 `build_note(1/(60/bpm), 60/bpm)` **省略了 `func` 参数**，
> 于是走了默认的 `sawtooth` —— 锯齿从 -1 线性升到 +1，`+1`、`×2` 之后是
> 0 到 4 的斜坡，再被 `limit(…, 1, 0)` 在 1/4 处削平。
> 所以改了 `build_note` 的默认 `func` 就会顺手改掉整条侧链的形状——
> 知道这回事就行，真想换默认值的话，把 `sidechain` 那两行显式写成 `sawtooth` 即可。

### 段落

一段一个 `compile_section`，最后 `np.concatenate` 起来：

```python
rise = compile_section("I. 潮起", 16, [轨…], [音量…], [效果…], level=0.085)
...
song = np.concatenate([rise, surge, theme, ebb, climax, afterglow])
```

段落之间要有对比：**抽掉鼓组、换音色、换和声**比单纯加音量有效得多。
《潮汐》的退潮段就是把 kick 和 bass 全撤了，只留 pad + reese + 远处的主音。

---

## 六、混音：用 RMS 说话

**别凭感觉拧音量。** 每个乐器在单位音量下的电平差着 20 dB 以上
（`pluck` 的 RMS 是 `hihat` 的 100 倍），凭感觉一定会糊。

### 做法

1. 用第四节那张实测表，反推每条轨的音量：`音量 = 想要的 RMS / 实测 RMS`。
2. 段落强弱用 **RMS** 而不是峰值来定。同样的峰值下，密集段落听起来响得多——
   《潮汐》第一版每段都顶到峰值，结果高潮的波峰因数只有 7.6 dB，一首糊墙。
3. 峰值留给最后统一归一化，段间比例不会被破坏。

```python
def compile_section(name, bars, tracks, volumes, effects, level):
    sec = compile_tracks([[t] for t in tracks], volumes, effects,
                         eff_chain(limiter, eff(times, 0.9)))   # 先压限 + 归一
    rms0 = np.sqrt(np.mean(sec.astype(np.float64) ** 2))
    return (sec.astype(np.float64) * (level / rms0)).astype(np.float32)   # 再定 RMS
```

最后整首 `song / max(abs(song)) * 0.98`，段间比例原样保留。

### 参考值

| | 潮起 | 潮涌 | 主题 | 退潮 | 高潮 | 余波 |
|---|---|---|---|---|---|---|
| 相对高潮 (RMS) | −10.9 dB | −5.4 | −3.1 | −7.7 | **0** | −12.0 |
| 波峰因数 | 10.6 dB | 14.3 | 13.3 | 9.9 | 12.9 | 11.8 |

### 验收指标

- **波峰因数**（峰值/RMS）：整首 12~16 dB 比较健康。
  低于 8 dB 基本就是糊墙，先怀疑低音过量和压限过狠。
- **频段能量**：高潮段 60–150 Hz 占 45% 左右、0.6–3 kHz 占 25% 左右、
  3 kHz 以上占 10% 左右比较像样。**低频占 80% 以上 = 和声和主音被埋了。**
  安静段落（前奏、间奏）低频占比天然会高，别拿高潮的标准去套。
- **削顶**：`np.sum(np.abs(x) > 0.999)` 必须是 0。
- **立体声**：左右相关 0.6~0.8 比较宽而不散；接近 1.0 说明是假立体声。

`python plot.py` 会把上面几项都打出来，顺便出图。

---

## 七、渲染与验证

```bash
cd song
SIDE=L python song.py     # 左声道
SIDE=R python song.py     # 右声道
python merge.py           # song.wav
python plot.py            # tide.png + 指标
```

渲染日志会按段打印 `RMS / 峰值 / 波峰因数`，拼装时再打印一遍段间比例——
**先看日志再看图**，绝大多数问题在数字里就能发现。

改完编曲记得**两个声道都重渲**，否则 `merge.py` 会把新旧两版拼在一起。

环境备注：仓库里的 `.venv` 在某些沙箱里 `uv run` 会因为缓存目录只读而失败，
可以直接用解释器加 `PYTHONPATH`：

```bash
PYTHONPATH=.venv/lib/python3.13/site-packages \
  ~/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13 -u song.py
```

---

## 八、故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `Cannot cast ufunc 'add' output from complex128 to float64` | 某个 FFT 滤波器返回了复数数组 | 滤波器出口要取 `.real`；拼轨时用 `np.real()` 兜底 |
| `operands could not be broadcast together` | 同一段里轨长不一致 | 全部过 `mk_track(items, bars)` |
| `kick` 里 `short_noise * slide(...)` 长度不匹配 | `bpm` 不是 `short_noise` 当初对应的速度 | `short_noise` 必须取 `round(60/bpm/4*rate)` |
| 某段整体声音很小 | master 输出的峰值 < 1，没触发自动归一化，段间比例被拉开 | 这是**故意的**；要改就改 `level` |
| 某段突然变响且变糊 | master 输出峰值 > 1，触发了自动 `limiter`（归一化到 1.0） | 让 master 输出峰值 < 1，或自己写 master |
| 全是 `nan` | 对一个全零数组调了 `maximize`（除零） | 别给空轨 / 静音轨套 `maximize`、`reverb` |
| 安静段落有底噪 | 混音中间用了 float16（11 位尾数） | 改成 float32；改回 float16 就会重新有底噪 |
| 长音首尾有咔哒声 | 相位不整周期 | 用 `declick()`，或让 `freq × duration` 取整 |
| 混响把安静音色淹了 | `reverb` 会把湿信号归一化到峰值 1.0 | 提高 `dry`（0.9+）或不过混响 |
| 渲染慢得离谱 | 整段套了 `slide` / `distortion` / `limiter` | 包络改 numpy；逐采样效果只用在音符上 |
| 音色比预期暗很多 | 高通/低通是硬切，且 `bandgain` 参数名有坑 | 看第四节第 3 条 |

---

## 九、现有作品

### 《潮汐》 / Tide

140 BPM · 记谱 E 小调（听感 G 小调）· 80 小节 · 2 分 17 秒。

和声两小节一个和弦、八小节一循环：记谱 `Em C G D`（i–VI–III–VII），
实际 `Gm E♭ B♭ F`。

| 段 | 小节 | 秒 | 内容 |
|---|---|---|---|
| I 潮起 | 16 | 27.4 | 纯正弦低音 + 弦乐铺底，琶音在第 9 小节探头，末尾 riser |
| II 潮涌 | 8 | 13.7 | 四踩底鼓 + rolling bass 进入，反拍 hi-hat |
| III 主题 | 16 | 27.4 | 主题 A 走两遍（`hardlead`），第二遍加军鼓 |
| IV 退潮 | 8 | 13.7 | 抽掉鼓组，reese 托和声，主音退到 `unison_saw`，末尾军鼓渐密 |
| V 高潮 | 24 | 41.1 | 主题 A → B → A，加三度下方和声，十六分 hi-hat，全奏 |
| VI 余波 | 8 | 13.7 | 回到 `Em C Em Em`，琶音放慢，淡出 |

主题 A / B 是 E 自然小调上的八小节乐句，和声线由 `third_below()` 生成。
成品：`song/song.wav`（立体声，峰值 0.980，波峰因数 16.3 dB，削顶 0）。
更早那首收在 git 历史里：`git show HEAD:song/song.py`。

### 配套工具

| 文件 | 作用 |
|---|---|
| [`song/song.py`](song/song.py) | 引擎 + 编曲，唯一需要懂的文件 |
| [`song/merge.py`](song/merge.py) | 把 L.wav / R.wav 交叉成 `song.wav`（跨平台，替代 `add.bat`） |
| [`song/plot.py`](song/plot.py) | 波形 / 频谱 / 高潮放大 + 客观指标 |
| [`song/add.bat`](song/add.bat) | Windows 上的 ffmpeg 合并（保留给老环境） |
| [`song/tide.png`](song/tide.png) | 《潮汐》的分析图 |
