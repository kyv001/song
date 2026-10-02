# song

用 numpy 和 Python 自带的 `wave` 库写歌的小程序，编曲靠硬编码，渲染是纯 CPU 的。

现在有两套引擎、两首曲子：

| | 引擎 | 曲子 | 渲染 |
|---|---|---|---|
| 老 | [`song/song.py`](song/song.py)，单文件 | 《潮汐》 | `SIDE=L` / `SIDE=R` 各渲一遍再拼，几分钟 × 2 |
| 新 | [`song/engine.py`](song/engine.py) | 《余烬》[`song/embers.py`](song/embers.py) | 一次出真立体声，约 2~3 分钟 |

《余烬》另有分轨导出和分轨波形可视化视频，见[第十一节](#十一分轨与可视化视频)。

> **给后来的 agent**
>
> **这是个 playground。** 整个仓库没有任何生产用途，唯一的目标是把曲子写得更好听。
> 引擎、音色、效果、依赖、文件结构**全都可以改**：删掉不用的函数、拆成模块、
> 换掉合成器架构、上 numba 干掉那些逐采样循环，都随便。不用管向后兼容、
> 可维护性和代码风格，也不用怕改坏——`git checkout` 就回来了。
>
> 下面这份文档是**地图**（哪里有什么、哪里会塌），不是「什么不许动」。
> 尤其 **「一、会咬人的地方」**：那是当前实现的行为，不是设计规范。
>
> - 想换一首曲子 → 整个编曲层（`compile_section` 那段）删掉重写
> - 想加音色 → 写个 `相位 -> 波形` 的函数传给 `build_note`
> - 想动引擎底层 → 请便，「会咬人的地方」那节就是你的变更影响清单
>
> **注意第一到八节讲的都是老引擎 `song.py`。** 新引擎 `engine.py` 换掉了里面
> 大部分设计（真立体声、全向量化、`note()` 不再跑调、真滤波器），它的用法和
> 坑在[第十节](#十新引擎-enginepy-与余烬)。

---

## 目录

**老引擎（`song/song.py` / 《潮汐》）**

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

**新引擎（`song/engine.py` / 《余烬》）**

- [十、新引擎 `engine.py` 与《余烬》](#十新引擎-enginepy-与余烬)
- [十一、分轨与可视化视频](#十一分轨与可视化视频)

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

下面每一条都是当前代码的真实行为，不是纪律——每条都写了违反之后会怎么炸。

1. **一段里所有轨必须等长。** `compile_tracks` 用 `song += track` 累加，长度不一致
   直接 broadcast 报错。用 `mk_track(items, bars)` 把每条轨补齐/截断到
   `round(bars * bar * rate)`，别手工数。

2. **`compile_tracks` 从列表尾部取基准轨**（`tracks_l.pop()` / `volumes.pop()` /
   `effects.pop()`），所以 `tracks`、`volumes`、`effects` 三个列表必须严格平行、
   长度一致。

3. **`master` 之后有一道自动保险**：`if max(abs(song)) > 1: song = limiter(song)`，
   而 `limiter` 结尾会 `maximize`，也就是把整段顶到峰值 1.0。想让段落保持你指定的
   电平，就让 master 输出的峰值 **小于 1**。

4. **`maximize(arr)` 是原地修改**（`arr /= max(abs(arr))`），会改掉传进去的数组。
   别把同一个数组同时交给两个地方。

5. **`bpm` 是所有时值的基准。** `kick` / `psy_punch` / `psy_tail` 的长度都等于一个
   十六分音符，`short_noise` 也按 `round(60 / bpm / 4 * rate)` 取，改 bpm 只改
   这一个变量。

6. **`empty()` 返回的是 int64 全零**（`build_note` 里 `volume == 0` 走的是提前返回
   分支），不是 float。拼接时 numpy 会自动提升，但别依赖它的 dtype。

7. **这些函数是逐采样的 Python 循环**：`sawtooth`、`square`、`triangle`、
   `distortion`、`limiter`、`slide`、`scratch`、`declick`。给整段（几百万采样）
   套一个 `slide` 会卡到怀疑人生——包络请用 numpy 写（见 `swell()`）。

8. **`reverb` 会把湿信号单独 `maximize` 到峰值 1.0**，再按 `(1 - dry)` 混进来。
   也就是说不管输入多轻，混响的水位是固定的：安静的音色（琶音、分解和弦）
   会被混响淹掉。这类轨用 `dry=0.9` 左右，或者干脆不过混响。

9. **`build_note` 的相位是 `linspace(0, freq*2π*duration, length)`。** 如果
    `freq × duration` 不是整数，音符首尾对不上，会产生咔哒声——长音尤其明显。
    引擎里的 `declick()` 就是为了擦这个，但它本身也很慢。

10. **引擎自带的 `hihat()` 基本没声音**（峰值 0.046），`crash()` 0.2 秒就衰减完。
    编曲层的 `hat()` / `crash_wash()` 是替代品。

11. `_er()` 里递归调用的是 `_reverb`（疑似笔误），只有 `no_convolve=True` 才会走到；
    `fnoise()`、`scratch()`、`comb_filter()` 目前没人用。

---

## 二、记谱：`note()` 整体高了三个半音

> **这一条只对老引擎 `song/song.py` 成立。** `engine.py` 里的 `note()` 是标准音高
> （A4 = 440，写 `D5` 就是 D5），第十节写曲子时不要再套这个 +3 的偏移。

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

一串首尾相接的波形数组。`mk_track(items, bars)` 负责拼成整段长度（不足补零、
超出截断），并且用 `np.concatenate` 一次拼好；再以 `[[t] for t in tracks]` 的
形式交给 `compile_tracks`，每条轨只会被 append 一次，避开逐音符 `np.append`
的 O(n²)。

### 4. 段

```python
compile_section(name, bars, tracks, volumes, effects, level)
```

若干条等长轨 → 各过效果链 → 乘音量 → 相加 → master（压限 + 归一）→ **再按 RMS
定到 `level`**。最后那一步见「六、混音」。

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

   自己写 FFT 类效果时记得出口取 `.real`：`ifft` 回来的是复数，直接往后传会在
   转 float32 时报 `ComplexWarning`，拼轨时也会因为复数没法累加而报错。

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
   只按峰值对齐的话，高潮会糊成一堵墙（波峰因数掉到 8 dB 以下）。
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

**渲染不是确定性的。** `noise()` 返回随机数，`reese()` / `strings()` 还会用
`np.random.rand()` 给每一路失谐抽相位偏移，所以同一份代码重渲，波形和峰值都
不一样（段落 RMS 基本一致，因为按 RMS 定级）。**别拿峰值或波形图去比对两次
渲染。** 想复现得自己在开头固定随机种子，目前没做——仓库里那张 `tide.png`
也只是一次取样的结果。

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
| `short_noise` 与 `slide(...)` 长度不匹配 | `bpm` 改了，`short_noise` 没跟着 | `short_noise` 取 `round(60/bpm/4*rate)` |
| 某段整体声音很小 | master 输出的峰值 < 1，没触发自动归一化，段间比例被拉开 | 这是**故意的**；要改就改 `level` |
| 某段突然变响且变糊 | master 输出峰值 > 1，触发了自动 `limiter`（归一化到 1.0） | 让 master 输出峰值 < 1，或自己写 master |
| 全是 `nan` | 对一个全零数组调了 `maximize`（除零） | 别给空轨 / 静音轨套 `maximize`、`reverb` |
| 安静段落有底噪 | 混音路径用了 float16（11 位尾数） | 保持 float32 |
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
上一版（原曲）在 git 历史里：`git show 8f4d3de:song/song.py`。

### 配套工具

| 文件 | 作用 |
|---|---|
| [`song/song.py`](song/song.py) | 引擎 + 编曲，唯一需要懂的文件 |
| [`song/merge.py`](song/merge.py) | 把 L.wav / R.wav 交叉成 `song.wav`，只用标准库 |
| [`song/plot.py`](song/plot.py) | 波形 / 频谱 / 高潮放大 + 客观指标 |
| [`song/add.bat`](song/add.bat) | 同样的事，走 ffmpeg，Windows 用 |
| [`song/tide.png`](song/tide.png) | 《潮汐》的分析图 |

---

## 十、新引擎 `engine.py` 与《余烬》

### 怎么跑

```bash
cd song
python embers.py               # 渲染 -> embers.wav（立体声，一次成型，约 2~3 分钟）
python analyze.py embers.wav   # 客观指标 + embers_analysis.png
python engine.py               # 音色试听台：每个音色渲一小段，打印峰值 / RMS / 耗时
```

只依赖 numpy / scipy / matplotlib，**不需要 moviepy**（`engine.py` 用
`scipy.io.wavfile` 直接读 `IR.wav`，能读 32bit float 的那种）。

### 跟老引擎比换掉了什么

| | `song/song.py`（老） | `engine.py`（新） |
|---|---|---|
| 立体声 | `SIDE=L/R` 渲两遍再拼，本质是「单声道 + 假宽度」 | 一次出真立体声，失谐的每一路各自摆位 |
| 逐采样循环 | `sawtooth`/`square`/`triangle`/`distortion`/`limiter`/`slide`/`declick` 全是 `for` | 全部 numpy，`python engine.py` 跑完整套音色只要 2 秒 |
| 记谱 | `note()` 整体 +3 半音（源码注释写「D#调」） | 标准 A4 = 440，写什么是什么 |
| 相位 | `linspace(0, f·2π·d, n)`，不整周期就咔哒 | `TAU·f·arange(n)/SR`，天生整周期，不需要 declick |
| 滤波器 | FFT 砖墙（把频点乘 0），没有共振和斜率 | RBJ 双二阶，能扫频、能共振（303 就靠它） |
| 混响 | 每条轨各卷一遍 IR，湿信号固定 `maximize` 到 1.0，安静音色会被淹 | 全局 send bus，整首只卷一次；湿/干比由编曲给 |
| 随机 | 每次重渲都不一样，没法比对 | `SEED` 固定，同一份代码逐采样可复现 |
| 整首耗时 | 几分钟 × 2 个声道 | 约 2.5 分钟（其中约一半是铺底的加法合成） |

### 音色表（实测：单位音量、124 BPM）

| 音色 | 峰 | RMS | 耗时 | 说明 |
|---|---|---|---|---|
| `pad(3音, 2拍)` | 0.753 | 0.138 | 223 ms | 每音 5 路失谐带限锯齿，失谐量决定声像位置 |
| `choir(3音, 2拍)` | 0.572 | 0.105 | 199 ms | 共振峰合成，`vowel="ah"/"ooh"/"eh"` |
| `sub` | 0.820 | 0.466 | 1 ms | 正弦 + 一点二次谐波 |
| `fm_bell` | 0.849 | 0.481 | 4 ms | 两对算子，非整数比 = 金属感 |
| `glass` | 0.982 | 0.366 | 3 ms | 14 倍频「毛刺」快速衰减，电钢 / 玻璃拨弦 |
| `acid` | 0.850 | 0.560 | 9 ms | 303：锯齿 + 共振低通包络 + 重音 + 滑音 |
| `supersaw` | 0.882 | 0.171 | 49 ms | 7 路失谐，每路独立声像（宽度来源） |
| `saw_pluck` | 1.138 | 0.267 | 12 ms | 减法拨弦 |
| `stab` | 1.007 | 0.202 | 56 ms | 和弦短促 stab |
| `noise_riser` | 0.580 | 0.110 | 13 ms | 带通噪声上行 + 音调上行 |
| `kick` | 0.980 | 0.495 | 2 ms | 音高包络正弦 + 噪声 click + 软削波 |
| `snare` | 0.900 | 0.282 | 2 ms | 两个鼓皮音 + 高通噪声 |
| `clap` | 0.850 | 0.118 | 2 ms | 四个错开的短噪声 + 弥散尾 |
| `hat` | 0.716 | 0.125 | 0 ms | 808 味儿：6 个非谐波方波 + 高通 |
| `openhat` | 1.007 | 0.131 | 1 ms | `hat(0.34, tau=0.15)` |
| `crash` | 0.800 | 0.107 | 48 ms | 18 个非谐波正弦 + 噪声，长尾 |
| `ride` | 0.600 | 0.076 | 10 ms | 比 crash 更「有形」 |
| `tom` | 0.850 | 0.364 | 2 ms | 音高包络 + 一点噪声 |
| `rim` | 0.700 | 0.075 | 0 ms | 边击 / 木鱼 |
| `impact` | 0.900 | 0.356 | 4 ms | 低频轰鸣 |

没用上但能用：`pink`（粉噪声）、`chorus`、`exciter`（高频激励）、`wavefold`、
`bitcrush`、`compress`、`duck_env`、`reverse_swell`、`subdrop`。

**耗时等级**：`pad` / `choir` 是加法合成的，谐波数按低通截止频率反推
（`bright*2.5/f`），把 `bright` 从 1400 调到 8000 会让耗时翻好几倍；其余音色
都是几毫秒。

### 《余烬》的设计

**概念**：一堆看着已经冷掉的灰，里面有火。

调性中心是 D，但色彩换了三次，每次换的其实就是**一个音**：

| 色彩 | 特征音 | 用在哪 | 怎么用 |
|---|---|---|---|
| D 弗里吉亚 | ♭2 = E♭ | I 灰烬、IV 暗涌 | `E♭maj9`（同时含 E♭ 和 D），寒气就是这个小二度 |
| D 多利亚 | ♮6 = B♮ | II 余温、III 复燃、V 燎原 | `G13` 是多利亚的签名和弦（IV 级是大三 + 小七）；主题 A 第 5 小节的 B♮ 是全曲「解冻」的一刻 |
| ♭VI–♭VII–i | B♭ – C – D | V 燎原后半 | 最直白也最有效的史诗终止 |
| Picardy 三度 | F♯ | VI 归寂结尾 | 最后落在 `Dmaj9`，灰里透出光 |

**结构**（124 BPM，104 小节 ≈ 3 分 21 秒 + 混响尾巴）：

| 段 | 小节 | 秒 | 内容 |
|---|---|---|---|
| I 灰烬 | 16 | 31.0 | 只有铺底、人声垫、钟、低频呼吸。没有鼓 |
| II 余温 | 16 | 30.9 | 半拍底鼓、玻璃琶音、低音进来；主题 A 前半句探头 |
| III 复燃 | 16 | 31.0 | 四踩底鼓 + 303 酸性低音 + 十六分钉钉；主题 A 全奏 |
| IV 暗涌 | 8 | 15.5 | 抽掉鼓组，弗里吉亚的寒气回来，riser + 军鼓滚奏拉起张力 |
| V 燎原 | 32 | 61.9 | 全奏。前 16 小节多利亚 vamp，后 16 小节 ♭VI–♭VII–i 圣歌 |
| VI 归寂 | 16 | 31.0 | 鼓组退出，钟与琶音回响，定格在 D 大三和弦 |

主题写成 `(音名, 拍数)` 的列表（`THEME_A` / `THEME_B` / `THEME_C`），和声线用
`shift_degree(name, -2)` 按**字母级数**往下数三度——不是按半音数，所以得到的
永远是和声音程（D 往下三度是 B♭ 不是 B，B♮ 往下三度是 G，C♯ 往下三度是 A）。

人声垫和琶音的音高一律从 `VOICINGS[cname][-3:]`（和弦排列的上方声部）里取，
不手写音名，这样改进行表的时候不会留下打错的和弦外音。

### 验收数字（`python analyze.py embers.wav`）

```
时长 211.0 秒   峰值 0.985   RMS -14.9 dBFS
波峰因数 14.7 dB（目标 12~16）   削顶 0   NaN 0
左右相关 0.817   300Hz 以上相关 0.712（目标 0.6~0.8）

频段能量      整首     最响的 20 秒
  20-60       11.7%      8.5%
  60-150      30.0%     26.5%
 150-400      22.3%     21.0%
 400-900      18.0%     22.9%
   0.9-3k     11.1%     13.5%
     3-8k      5.6%      6.7%
      8k+      1.3%      0.9%
```

逐段（相对最响的 V 段）：

| 段 | I | II | III | IV | V | VI |
|---|---|---|---|---|---|---|
| RMS | −9.6 dB | −7.0 | −2.9 | −7.9 | **0** | −8.9 |
| 波峰因数 | 15.8 dB | 14.4 | 12.2 | 14.0 | 11.2 | 15.7 |

![《余烬》分析图](song/embers_analysis.png)

150 Hz 以下占 41.7%（README 给高潮段定的参考值就是 45%），400 Hz-3 kHz 占
29.1%。整首的相关度 0.817 比 0.6~0.8 略高，是因为低音和底鼓按规矩走在正中间；
**决定「宽不宽」的是 300 Hz 以上那 0.712**，`analyze.py` 会把这个数单独打出来。

调这一版踩到的五个坑，记下来省得下一个人再踩：

1. **低频一开始占了 63%。** 和弦排列的根音横跨 F2–B♭3，直接减一个八度会掉到
   44 Hz；再加上每层铺底都带一个低八度正弦、底鼓又在 49 Hz，20-100 Hz 直接
   挤爆。修法是把铺底的低八度从「给满」改成给 0.2~0.35 的电平。
2. **每段单独看才发现的。** 整首的频段占比是健康的，但 II 段 40-100 Hz 占
   60%——因为平均值被 V 段拉平了。`analyze.py` 的逐段表就是为了抓这个：
   做「安静段落」的时候，低音的比例反而最容易失控。
3. **响度是削峰给的，不是归一化给的。** 峰值顶到 0.985 之后 RMS 只有 −19 dBFS，
   因为波峰因数 18.9 dB。在母带最后加一道 `tanh` 软削峰（阈值 0.72）——
   只有瞬态会碰到它——RMS 抬到 −14.9 dBFS，波峰因数收进 14.7 dB，
   而各段的内部动态基本没动。
4. **`floor_beats()` 的默认值写错了**，见下。
5. **低音每小节掉了 0.8 秒**，见下。

后两条是**分轨波形图**照出来的，谱图上看不见：

* `floor_beats(bars, hits=1.0)` 里那个默认值让「四踩底鼓」变成每小节**一脚**——
  III 段和 V 段的律动整个垮掉，但频段能量、波峰因数全都正常，因为我一开始只是
  按这些指标调的电平。是视频里 Kick 那一格在应该满的地方是一条直线才发现的。
  现在默认 `hits_per_bar=4`。
* 低音写成 `n * BAR * 0.78`：一小节一个和弦时，每个和弦留 0.42 小节（0.8 秒）
  的空洞，等于四踩段落里低音每小节掉一次。谱图上只是低频少一点，看不出「断了」。
  现在铺到 0.96 小节。`stems.py` 现在会打一张**逐段逐轨的静音占比表**
  （自检 2）——`analyze.py` 只统计混合后的结果，看不见单条轨的断点，
  而这张表一眼就能看出「Sub Bass 在 V 段静音 21%」不对劲。

还有一条**没做**的：一开始写了个 `sub_freq()`，想把低音「折叠」进 55-110 Hz，
理由是 44 Hz 在小喇叭上放不出来。但根音横跨 F2-B♭3，折叠会让 `Dm9 → G13`
这条下行四度的低音线变成上行五度——**低音走向被搞反**，比多几赫兹的
「不可闻低频」糟得多。现在低音一律取根音下方八度（44-78 Hz）。

### 新引擎会咬人的地方

1. **`Section.finish()` 会按 RMS 把整段重新定级。** 段内各轨的绝对增益不重要，
   重要的是它们之间的比例；改一条轨的电平，整段会被缩回去。想改段间关系就改
   `level_db`。

2. **`offset_bars` 的起点落在段末之后，整条轨会被静默丢掉**（不报错）。这一条
   害我调了半天：`add()` 里 `a[:, :self.n - off]` 在 `off > self.n` 时是负索引，
   拼出来长度变成 2n，报的是莫名其妙的 broadcast 错误。现在有 guard 了，但
   编曲时事件位置和段长对不上，表现就是「这条轨没响」。用 `analyze.py` 的逐段表查。

3. **`duck` 是「整段长」的侧链包络，在摆位之后才乘。** `E.duck_env(bars, 4,
   depth=0.4)` 生成一段，`add(..., duck=duck)` / `add_seq(..., duck=duck)` 都收。

4. **混响送出量是线性的，总水位由 `mixdown(rev_amt=...)` 定。** IR 按能量归一
   （白噪过一遍输出 RMS ≈ 输入 RMS），所以 wet 的电平 ≈ send 总线的电平。人声垫
   / 钟 / 铺底给 0.8~1.0，鼓和低音给 0~0.25。

5. **母带是一条链**：总线压缩 → 28 Hz 高通 → 低架 / 高架 → 中高频加宽 →
   `tanh` 软削峰 → 限制器 → 归一化。响度是 `master_clip` 给的：这一版削峰阈值
   0.72，把波峰因数从 18.9 dB 收到 15.1 dB、RMS 从 −19.1 抬到 −15.2 dBFS。
   阈值调低会更响也更脏。

6. **段落之间没有交叉淡化**，只有每段首尾 4/8 ms 的 fade。接缝靠混响尾巴和
   同一个 downbeat 盖住。要真拼接就把 `fade()` 换成交叉淡化。

7. **`note()` / `chord()` 是普通函数**：`E.note("D5")` → 587.33 Hz。老引擎那条
   「写 E 听 G」的规矩在这里不成立。

8. **`SEED` 环境变量**控制所有随机数（失谐相位、噪声）。默认 20250902；换一个
   数就是同一首曲子的另一版渲染，`SEED=1 python embers.py` 即可。

### 配套工具（新的）

| 文件 | 作用 |
|---|---|
| [`song/engine.py`](song/engine.py) | 向量化立体声合成引擎 + 混音台，可单独 `python engine.py` 自检 |
| [`song/embers.py`](song/embers.py) | 《余烬》的编曲层，只有和声表 / 主题 / 段落 |
| [`song/analyze.py`](song/analyze.py) | 通用渲染自检：指标 + `*_analysis.png`，认 `embers.STRUCTURE` 自动分段 |
| [`song/embers.wav`](song/embers.wav) | 成品（立体声，211 秒，峰值 0.985，波峰因数 15.1 dB，削顶 0） |
| [`song/embers_analysis.png`](song/embers_analysis.png) | 《余烬》的分析图 |


---

## 十一、分轨与可视化视频

### 怎么跑

```bash
cd song
python stems.py         # 导出 13 条分轨 -> stems/*.wav（约 2 分钟）
python visualize.py     # -> embers_visual.mp4（1080p60，约 1.5 分钟）
python visualize.py --jobs 1                # 串行，约 4 分钟
python visualize.py --limit 5 --out t.mp4   # 只渲前 5 秒，调画面用
```

依赖多了 `pygame` 和 `opencv-python-headless`（已经 `uv add` 进 `pyproject.toml`），
**不需要 moviepy**。

### 画面

原版 `snippets/song_visualize.py` 的画面逻辑照搬：3 列 x 5 行网格，13 格放分轨，
右下角 2 格合并成 Master；每格横向 640 像素对应 6400 个采样点（0.145 秒），
每帧往后挪一个采样窗口，于是波形「从左往右流过」；60 fps，1920x1080。
静音的轨道是一条平直线，所以某个时刻是哪些乐器在响，一眼就能看出来——
《余烬》的六个段落在这张图上比频谱图直观得多（I 段只有 Pad / Sub / Bells / FX
四条线在动，Kick 和 Lead 是平的）。

### 性能：8 分钟 -> 1 分 26 秒

原版 37.7 ms/帧，整首 12658 帧要 8 分钟。逐阶段量下来，**69% 花在取像素上，
不是画波形**：

| 阶段 | ms/帧 | 处理 |
|---|---|---|
| 清屏 + 边框 + 标签 | 0.51 | 留着 |
| `pygame.draw.aalines` 画 14 条波形 | 4.51 | 留着（试过 numpy 光栅化，6.1 ms，更慢） |
| `surfarray.array3d` + `swapaxes` + 翻转通道 | **25.72** | 换成 `pygame.image.tostring(screen, "RGB")`，3.0 ms |
| `cv2.VideoWriter`（mp4v）编码 | 6.96 | 换成裸帧管道灌给 ffmpeg / libx264 |

关键点：`surfarray.array3d(screen)` 给的是 **(宽, 高, 3)**，要变成 cv2 要的
(高, 宽, 3) 就得转置，那是一次跨步大拷贝，只有 0.22 GB/s；
`image.tostring(screen, "RGB")` 直接给行优先字节流，2.07 GB/s（快 9.6 倍），
而且本来就是 RGB——只要让 ffmpeg 按 `-pix_fmt rgb24` 读，连通道翻转都省了。
（`surfarray.pixels3d` 只要 9 ms，但还是得转置，不够快。）

然后两层并行：画帧是 CPU 密集的 Python/SDL 调用（约 5 ms/帧），16 个核闲着浪费，
所以把帧区间切成 `--jobs` 份，每个进程画自己那段、各自起一个单线程 x264 编成
MPEG-TS，最后 concat 无损拼起来；同时 ffmpeg 在另一个进程里编码，和画帧重叠。

| 版本 | ms/帧 | 整首 12658 帧 |
|---|---|---|
| 原版（array3d + cv2） | 37.7 | 8 分 00 秒 |
| tostring + 管道（`--jobs 1`） | 19.7 | 4 分 10 秒 |
| 再并行 8 进程（默认） | **6.3** | **1 分 25 秒** |

**8 进程之后不再涨**：12 / 16 进程都是 6.0~6.3 ms/帧——16 个画帧进程 + 16 个
ffmpeg 抢的是内存带宽，不是核。

### 改编自原版的几处

| 原版 | 这里 | 为什么 |
|---|---|---|
| `moviepy.editor.AudioFileClip` | `scipy.io.wavfile` | moviepy 2.x 已经把 `.editor` 删了；而且我们全是 wav，用不着 moviepy |
| `pygame.mixer.Sound` 读主音频 | 同上 | 顺带去掉了对音频设备的依赖 |
| `surfarray` + `cv2.VideoWriter` | `image.tostring` + 裸帧管道 + libx264 | 见上表，35 ms -> 3 ms |
| `os.system(ffmpeg ... -acodec copy)` | `-c:a aac` | wav 的 PCM 封不进 mp4 |
| 播到结尾采样点不够就 `except` 跳过整格 | 补零 | 原版最后几帧会有格子突然空掉 |

另外加了 `SDL_VIDEODRIVER=dummy`（沙箱里没有显示设备，`set_mode` 会直接失败）。

**试过但没用的两条**：关掉抗锯齿（`pygame.draw.lines`）不但更难看，成片还
**更大**（27.0 MB vs 24.3 MB / 20 秒）——锯齿边的高频比平滑边更难预测；
换更慢的 x264 preset 也没用，veryfast / fast / medium / slow 在同样 CRF 下
体积只差 13%。1080p60 的 13 Mbps（整首 351 MB）就是这个画面该有的代价，
想小就降 `--crf` 或 `--fps 30`。

### 13 组是怎么从 22 个声部归出来的

`embers.py` 的 `STEM_GROUPS` 干这件事，约束是那个 3x5 网格正好 13 格：

| 组 | 来自 | 组 | 来自 |
|---|---|---|---|
| Pad | `pad` | Cymbals | `crash` `ride` |
| Choir | `choir` | Bells | `bell` |
| Sub Bass | `sub` | Arp | `arp` |
| Acid Bass | `acid` `stab` | Lead | `lead` `harmony` |
| Kick | `kick` | FX | `impact` `subdrop` `riser` `rev` `tom` `breath` |
| Snare | `roll` `rim` | Clap | `clap` |
| Hats | `hat` `openhat` | | |

归组依据是「看波形的时候你想区分什么」：音色层、节奏层、低音层、主音层、音效层。
改编曲时如果新加了声部名而忘了归组，它会静默不出现在视频里——
`stems.py` 启动时会打印每段累计了多少个声部，对不上就说明漏了。

### 会咬人的地方

1. **分轨是母带之前的干声**：没有混响总线、没有总线压缩和削峰，13 条相加
   **不等于** `embers.wav`（实测相关系数 0.872，差的正是混响和延迟的湿声）。
   画波形监视器这样最好，但别拿它当「分轨重混」的素材。

2. **每条分轨导出前单独归一化到峰值 0.95。** 调音台的真实电平是 −30 ~ −10 dB，
   不归一化的话波形图上全是直线。代价是「哪一轨更响」这个信息丢了——
   想看电平关系就看 `stems.py` 打印的 RMS 列。

3. **`stems.py` 会把整首重算一遍**（约 2 分钟）。`engine.py` 的 seed 固定，
   所以重算出来的分轨和 `embers.wav` 严格对得上；改了编曲要**两边都重跑**，
   否则视频里的分轨和音轨就对不上了。

4. **`visualize.py` 默认读 `embers.wav`**；换成别的曲子要同时给
   `--audio` 和 `--stems`，而且分轨数量必须是 13（网格是写死的 3x5）。
