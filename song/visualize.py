"""分轨波形监视器：把一首歌的分轨画成波形网格并录成视频

改编自 `/data/Programming/snippets/song_visualize.py`。画面逻辑（3x5 网格、
13 条分轨 + 右下角两格合并的 Master、波形从左往右流过、抗锯齿折线、逐帧
60fps、1920x1080）全部照搬，改的是**取像素**和**编码**这两步——原版 37 ms/帧
里有 35 ms 花在这两处。

用法：
    cd song
    python stems.py                   # 先导出分轨 -> stems/*.wav
    python visualize.py               # -> embers_visual.mp4
    python visualize.py --jobs 8      # 并行（默认 8）
    python visualize.py --jobs 1      # 串行
    python visualize.py --limit 5     # 只渲前 5 秒，调画面用

--- 原版慢在哪（实测 1920x1080，ms/帧）---

    清屏 + 边框 + 标签                  0.51
    pygame.draw.aalines 画 14 条波形     4.51
    surfarray.array3d + swapaxes + 翻转  25.72   <-- 69%
    cv2.VideoWriter(mp4v) 编码            6.96
    ------------------------------------------
    合计                                37.7     -> 整首 12660 帧 = 8 分钟

**第一处**：`pygame.surfarray.array3d(screen)` 返回的是 (宽, 高, 3)，要变成
cv2 要的 (高, 宽, 3) 就得 swapaxes、再翻转通道——那是一次跨步大拷贝，实测
只有 0.22 GB/s。换成 `pygame.image.tostring(screen, "RGB")` 直接拿行优先的
字节流（2.07 GB/s，快 9.6 倍），而且本来就是 RGB，只要让 ffmpeg 按
`-pix_fmt rgb24` 读，连通道翻转都省了。
（也试过 `surfarray.pixels3d`：9 ms，还是要转置，不够快。）

**第二处**：cv2 只能写 mp4v，1080p60 要 40 Mbps、整首 1 GB 多，而且单线程
编码 + 之后还要再跑一遍 ffmpeg 重编码。改成把裸帧直接灌进 ffmpeg 的 stdin，
让 libx264 编码——它自己会吃满多核。

**第三处**：编码和画帧本来一前一后串行，现在 ffmpeg 在另一个进程里跑，
两边重叠，瓶颈只剩画帧。

--- 并行 ---

画帧本身是 CPU 密集的 Python/SDL 调用（约 5 ms/帧），16 个核闲着浪费。所以
把帧区间切成 `--jobs` 份，每个进程画自己那段、各自起一个单线程 x264 编成
MPEG-TS，最后用 concat 解复用器无损拼起来（`-c:v copy`，不重编码）。
`--jobs 1` 就是单进程串行。

顺带修掉的小问题：原版播到结尾采样点不够时靠 try/except 跳过整格，最后几帧
会有格子突然空掉；这里改成补零。

--- 试过但没用的两条（别再走一遍）---

* **关掉抗锯齿**（`pygame.draw.lines`）：不但更难看，成片还**更大**（27.0 MB vs
  24.3 MB / 20 秒）——锯齿边的高频比平滑边更难预测。
* **换更慢的 x264 preset**：veryfast / fast / medium / slow 在同样的 CRF 下
  体积只差 13%（39.5 / 34.9 / 35.8 / 34.9 MB），这段素材本来就是高熵的
  （满屏细线），preset 救不了。1080p60 的 13 Mbps 就是这个画面该有的代价；
  想小就降 `--crf` 或者 `--fps 30`。

--- 实测 ---

    版本                      ms/帧    整首 12658 帧
    原版（array3d+cv2）        37.7     8 分 00 秒
    改 tostring + 管道         19.7     4 分 10 秒   （--jobs 1）
    再并行 8 进程               6.4     1 分 26 秒   （默认）

8 进程之后不再涨：12 / 16 进程都是 6.0~6.3 ms/帧——16 个画帧进程 + 16 个
ffmpeg 抢的是内存带宽，不是核。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

# 必须在 import pygame 之前设：沙箱里没有显示 / 音频设备
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["SDL_VIDEO_WINDOW_POS"] = "0, 0"

import numpy as np
from scipy.io import wavfile

import embers as M

#  fork 出来的子进程直接继承这些全局量，975 MB 的分轨数据不会被复制
_G: dict = {}


# ---------------------------------------------------------------------------
#  读音频
# ---------------------------------------------------------------------------


def load_mono(path: str):
    """读成 float64 单声道。整数 wav 按满量程归一，float wav 原样。"""
    sr, d = wavfile.read(path)
    if d.dtype == np.int16:
        x = d.astype(np.float64) / 32768.0
    elif d.dtype == np.int32:
        x = d.astype(np.float64) / 2147483648.0
    elif d.dtype == np.uint8:
        x = (d.astype(np.float64) - 128.0) / 128.0
    else:
        x = d.astype(np.float64)
    if x.ndim == 2:
        x = (x[:, 0] + x[:, 1]) / 2.0
    return sr, x


def surface_bytes(surf):
    """拿行优先的 RGB 字节流。pygame 2.3+ 叫 image.tobytes，老版本叫 tostring。"""
    import pygame
    fn = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    return fn(surf, "RGB")


# ---------------------------------------------------------------------------
#  一个分块的渲染 + 编码（子进程入口）
# ---------------------------------------------------------------------------


def render_chunk(job):
    """画 [f0, f1) 这些帧，直接灌给一个单线程 x264，输出 MPEG-TS 分块。"""
    idx, f0, f1, out_path, cfg = job
    import pygame

    labels = cfg["labels"]
    size = cfg["size"]
    blockw, blockh = cfg["blockw"], cfg["blockh"]
    sounds, master, sr = _G["sounds"], _G["master"], _G["sr"]
    fps, step = cfg["fps"], cfg["step"]

    pygame.init()
    font = pygame.font.SysFont(["DejaVu Sans", "Arial", "Liberation Sans"], 20,
                               True) or pygame.font.Font(None, 22)
    screen = pygame.display.set_mode(size)

    x = np.arange(0, blockw)
    x_master = np.arange(0, blockw * 2)
    label_surfs = [font.render(lb, True, (0, 0, 0), (255, 255, 255))
                   for lb in labels]
    surf_master = font.render("Master", True, (0, 0, 0), (255, 255, 255))

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", cfg["preset"], "-crf", str(cfg["crf"]),
           "-threads", str(cfg["threads"]), "-pix_fmt", "yuv420p",
           "-f", "mpegts", out_path]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
    write = enc.stdin.write

    total = len(labels)
    for i in range(f0, f1):
        screen.fill(0x000000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        t = round(i / fps * sr)
        n = 0
        for c in range(5):
            for r in range(3):
                if n < total:
                    pygame.draw.rect(screen, (255, 255, 255),
                                     (r * blockw, c * blockh, blockw, blockh),
                                     2, 2)
                    screen.blit(label_surfs[n], (r * blockw, c * blockh))
                    seg = sounds[n][t:t + blockw * step:step]
                    if len(seg) < blockw:      # 结尾不够就补零，别整格空掉
                        seg = np.concatenate([seg, np.zeros(blockw - len(seg))])
                    pts = np.empty((blockw, 2))
                    pts[:, 0] = x + r * blockw
                    pts[:, 1] = (seg * 0.4 + 0.5) * blockh + c * blockh
                    pygame.draw.aalines(screen, (255, 255, 255), False, pts)
                n += 1

        pygame.draw.rect(screen, (255, 255, 255),
                         (blockw, 4 * blockh, blockw * 2, blockh), 2, 2)
        screen.blit(surf_master, (blockw, 4 * blockh))
        seg = master[t:t + blockw * 2 * step:step]
        if len(seg) < blockw * 2:
            seg = np.concatenate([seg, np.zeros(blockw * 2 - len(seg))])
        pts = np.empty((blockw * 2, 2))
        pts[:, 0] = x_master + blockw
        pts[:, 1] = (seg * 0.4 + 0.5) * blockh + 4 * blockh
        pygame.draw.aalines(screen, (255, 255, 255), False, pts)

        write(surface_bytes(screen))

    enc.stdin.close()
    enc.wait()
    pygame.quit()
    if enc.returncode != 0:
        raise RuntimeError("分块 %d 编码失败：%s"
                           % (idx, enc.stderr.read().decode()[-800:]))
    return out_path


# ---------------------------------------------------------------------------
#  主流程
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="embers.wav", help="主音频（混好的成品）")
    ap.add_argument("--stems", default="stems", help="分轨目录")
    ap.add_argument("--out", default="embers_visual.mp4")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--step", type=int, default=10, help="1 像素 = 几个采样点")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--jobs", type=int, default=8, help="并行进程数，1 = 串行")
    ap.add_argument("--crf", type=int, default=19, help="H.264 质量，越小越清楚")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--limit", type=float, default=0.0, help="只渲前 N 秒（调试）")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("找不到 ffmpeg")

    width, height = (int(v) for v in args.size.lower().split("x"))
    size = (width, height)
    blockw, blockh = width // 3, height // 5

    labels = list(M.STEM_ORDER)
    stem_files = [os.path.join(args.stems, n.replace(" ", "_") + ".wav")
                  for n in labels]
    missing = [f for f in stem_files if not os.path.exists(f)]
    if missing:
        sys.exit("缺分轨文件：\n  " + "\n  ".join(missing) +
                 "\n先跑 python stems.py")

    print("读素材 …")
    sr, master = load_mono(args.audio)
    sounds = []
    for f in stem_files:
        s, x = load_mono(f)
        if s != sr:
            sys.exit(f"{f} 采样率 {s} 和主音频 {sr} 不一致")
        sounds.append(x)

    dur = len(master) / sr
    if args.limit:
        dur = min(dur, args.limit)
    frames = int(round(dur * args.fps))
    print(f"  主音频 {len(master) / sr:.1f} 秒 / {len(labels)} 条分轨 / "
          f"{frames} 帧 @ {args.fps}fps / {size[0]}x{size[1]} / {args.jobs} 进程")

    _G["sounds"], _G["master"], _G["sr"] = sounds, master, sr
    # 串行就让 x264 吃满多核；并行时每个分块只给 1 个线程，免得 8 个 ffmpeg
    # 互相抢核（实测 8 进程是甜点：再往上内存带宽先饱和，ms/帧反而涨回去）
    cfg = {"labels": labels, "size": size, "blockw": blockw, "blockh": blockh,
           "fps": args.fps, "step": args.step, "crf": args.crf,
           "preset": args.preset, "threads": 0 if args.jobs == 1 else 1}

    tmpdir = "_vis_tmp"
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)
    jobs = max(1, min(args.jobs, frames))
    bounds = np.linspace(0, frames, jobs + 1).astype(int)
    tasks = [(i, int(bounds[i]), int(bounds[i + 1]),
              os.path.join(tmpdir, f"part_{i:03d}.ts"), cfg)
             for i in range(jobs) if bounds[i + 1] > bounds[i]]

    t0 = time.time()
    if len(tasks) == 1:
        print("渲染（单进程）…")
        render_chunk(tasks[0])
    else:
        import multiprocessing as mp
        print(f"渲染（{len(tasks)} 进程并行）…")
        ctx = mp.get_context("fork")     # fork 才能共享那 975 MB 分轨数据
        with ctx.Pool(len(tasks)) as pool:
            done = 0
            for _ in pool.imap_unordered(render_chunk, tasks):
                done += 1
                el = time.time() - t0
                print(f"  分块 {done}/{len(tasks)} 完成，已用 {el / 60:.2f} 分钟",
                      flush=True)
    el = time.time() - t0
    print(f"画面完成：{frames} 帧，用时 {el / 60:.2f} 分钟"
          f"（{el / max(frames, 1) * 1000:.1f} ms/帧，含编码）")

    # 无损拼接 + 封装音轨（-c:v copy，不重编码）
    print("拼接 + 封装音轨 …")
    listfile = os.path.join(tmpdir, "concat.txt")
    with open(listfile, "w") as f:
        for _, _, _, p, _ in tasks:
            f.write("file '%s'\n" % os.path.abspath(p))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", listfile,
           "-i", args.audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(args.out) / 1e6
    print(f"完成 -> {args.out}（{sz:.1f} MB，{dur:.1f} 秒）")
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
