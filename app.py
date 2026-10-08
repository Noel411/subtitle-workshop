# -*- coding: utf-8 -*-
"""
字幕工坊 - 本地离线音频转字幕工具（纯提取版）
- faster-whisper 转录（支持 CPU / NVIDIA GPU 加速）
- 针对耳语/ASMR 音频优化：关闭 VAD 预过滤 + 幻觉过滤 + 碎片清理
- 日语短句断句（≤18 字，词级时间戳对时）
- 仅输出日语 SRT（翻译由 AI 助手按作品人工精翻后填入）
"""
import os
import re
import sys
import time
import queue
import threading
import traceback

# ---------------------------------------------------------------- 环境准备
def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _setup_dll_dirs():
    """GPU 模式需要 cuBLAS / cuDNN DLL；打包后位于 exe 旁的 cudnn/ cublas/ 目录。"""
    candidates = [
        os.path.join(app_dir(), "cudnn"),
        os.path.join(app_dir(), "cublas"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass


_setup_dll_dirs()

# 禁用系统代理（部分机器代理会拦截模型下载）
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

# 模型自动下载走国内镜像（已有外部设置时以外部为准）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ---------------------------------------------------------------- 管线逻辑
SUPPORTED_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma",
                 ".mp4", ".mkv", ".flv", ".mov", ".avi", ".webm", ".aac")


def log_q_put(q, msg):
    q.put(("log", msg))


def find_model_dir(model_name):
    """优先用 exe 旁的 models/，找不到则交给 faster-whisper 自行解析（缓存或下载）。"""
    local = os.path.join(app_dir(), "models", "faster-whisper-" + model_name)
    if os.path.isdir(local):
        return local
    return model_name


def transcribe(audio_path, model_name, use_gpu, q, cancel_flag):
    """转录 + 幻觉过滤 + 清理 + 断句。返回 segments 列表。"""
    from faster_whisper import WhisperModel

    device, compute = "cpu", "int8"
    if use_gpu:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() < 1:
                log_q_put(q, "⚠ 未检测到 NVIDIA CUDA 设备，自动改用 CPU")
            else:
                device, compute = "cuda", "float16"
        except Exception as e:
            log_q_put(q, "⚠ CUDA 初始化失败(%s)，自动改用 CPU" % e)

    model_dir = find_model_dir(model_name)
    log_q_put(q, "加载模型 %s (%s/%s)..." % (model_name, device, compute))
    t0 = time.time()
    try:
        model = WhisperModel(model_dir, device=device, compute_type=compute)
    except Exception as e:
        if device == "cuda":
            log_q_put(q, "⚠ GPU 加载失败(%s)，回退 CPU 重新加载" % e)
            device, compute = "cpu", "int8"
            model = WhisperModel(model_dir, device=device, compute_type=compute)
        else:
            raise
    log_q_put(q, "模型加载完成，用时 %.1f 秒（设备: %s）" % (time.time() - t0, device))

    log_q_put(q, "开始转录（耳语优化模式：关闭 VAD 预过滤）...")
    seg_iter, info = model.transcribe(
        audio_path,
        language="ja",
        vad_filter=False,                    # 关键：耳语会被 VAD 误剪
        word_timestamps=True,
        beam_size=5,
        condition_on_previous_text=False,    # 降低幻觉连锁
        no_speech_threshold=0.5,
    )
    total = info.duration
    log_q_put(q, "音频总时长: %d 分 %d 秒" % divmod(int(total), 60))

    raw = []
    last_start = -1
    for seg in seg_iter:
        if cancel_flag.is_set():
            log_q_put(q, "已取消")
            return None
        words = []
        if seg.words:
            for w in seg.words:
                wd = w.word.strip()
                if wd:
                    words.append({"start": w.start, "end": w.end, "word": wd})
        raw.append({"start": seg.start, "end": seg.end, "text": seg.text.strip(),
                    "words": words})
        # 进度按已处理到的位置估算
        q.put(("progress", seg.end / total if total else 0))
        if seg.start - last_start > 300:
            m, s = divmod(int(seg.start), 60)
            log_q_put(q, "  ...已处理至 %02d:%02d" % (m, s))
        last_start = seg.start

    q.put(("progress", 1.0))
    log_q_put(q, "原始识别 %d 条，开始过滤幻觉与噪声..." % len(raw))

    filtered = []
    for s in raw:
        dur = s["end"] - s["start"]
        text = s["text"].strip()
        if not text or text in {"-", "…", "..."}:
            continue
        # 幻觉核心规则：语速合理性（正常 ≤ 1.5 字/秒，留 2 倍余量）
        max_dur = max(12.0, len(text) / 1.5 * 2.5)
        if dur > max_dur and len(text) < 60:
            continue
        # 纯喘息/单字符碎片
        if re.fullmatch(r"[ぁ-んァ-ヶa-zA-Z-―ー、。！？\s]{1,2}", text):
            continue
        filtered.append(s)

    # 去重（相邻同文）与重叠修正
    cleaned = []
    for s in filtered:
        if cleaned:
            prev = cleaned[-1]
            if s["text"] == prev["text"] and s["start"] - prev["end"] < 5:
                continue
            if s["start"] < prev["end"]:
                if s["end"] <= prev["end"]:
                    continue  # 完全被覆盖
                s = dict(s, start=prev["end"])
                if s["end"] - s["start"] < 0.3:
                    continue
        cleaned.append(s)

    log_q_put(q, "过滤后保留 %d 条有效字幕" % len(cleaned))
    if not cleaned:
        log_q_put(q, "未识别到有效语音。")
        return cleaned

    cleaned = split_long_segments(cleaned, q)
    return cleaned


# ---------------------------------------------------------------- 断句层
MAX_LINE_LEN = 18

_BREAK_AFTER = "、。！？…!?!"      # 保留在这些标点之后断开
_SOFT_BREAK = "はがをにでとやもねよわか"  # 无标点长句的软断点


def _split_clauses(text):
    """把一段日文按标点切成短句，短句合并、长句在助词处二次切断。"""
    text = text.replace(" ", "")
    parts = re.split(r"(?<=[、。！？…!?])", text)
    parts = [p for p in parts if p]
    merged = []
    for p in parts:
        if merged and (len(merged[-1]) < 4 or len(p) < 2):
            merged[-1] += p
        else:
            merged.append(p)
    out = []
    for p in merged:
        while len(p) > MAX_LINE_LEN:
            cut = MAX_LINE_LEN
            for i in range(MAX_LINE_LEN, MAX_LINE_LEN - 8, -1):
                if 0 < i < len(p) and p[i - 1] in _SOFT_BREAK:
                    cut = i
                    break
            out.append(p[:cut])
            p = p[cut:]
        if p:
            out.append(p)
    return out


def split_long_segments(segs, q):
    """把超长 segment 切成 <=18 字的短行，时间用词级时间戳精确分摊。"""
    out = []
    n_split = 0
    for s in segs:
        clauses = _split_clauses(s["text"])
        if len(clauses) <= 1:
            out.append(s)
            continue
        words = s.get("words") or []
        if words:
            idx = 0
            ok = True
            pieces = []
            for c in clauses:
                need = len(c)
                w_start = words[idx]["start"] if idx < len(words) else s["end"]
                acc = 0
                w_end = w_start
                while idx < len(words) and acc < need:
                    acc += len(words[idx]["word"])
                    w_end = words[idx]["end"]
                    idx += 1
                if acc < need:          # 词表覆盖不全，退回比例分摊
                    ok = False
                    break
                pieces.append([w_start, max(w_end, w_start + 0.5), c])
            if ok:
                for st, en, c in pieces:
                    out.append({"start": st, "end": min(en, s["end"]), "text": c,
                                "words": []})
                    n_split += 1
                continue
        # 比例分摊兜底
        total = len("".join(clauses))
        dur = s["end"] - s["start"]
        pos = s["start"]
        for c in clauses:
            d = dur * len(c) / total
            out.append({"start": pos, "end": pos + d, "text": c, "words": []})
            pos += d
            n_split += 1
    if n_split:
        log_q_put(q, "长句切分: %d 条 -> %d 条（按标点/字数，词级对时）"
                  % (len(segs), len(out)))
    return out


def fmt_ts(sec):
    sec = max(0.0, sec)
    ms = int(round(sec * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def write_srt(segs, out_dir, stem):
    """仅输出日语 SRT（时间轴 + 日文短句），供后续人工精翻。"""
    out = []
    for n, t in enumerate(segs, 1):
        out.append(str(n))
        out.append("%s --> %s" % (fmt_ts(t["start"]), fmt_ts(t["end"])))
        out.append(t.get("text", ""))
        out.append("")
    p = os.path.join(out_dir, stem + "_日文字幕.srt")
    with open(p, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(out))
    return p


def run_pipeline(audio_path, model_name, use_gpu, out_dir, q, cancel_flag):
    try:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        segs = transcribe(audio_path, model_name, use_gpu, q, cancel_flag)
        if segs is None:
            return
        path = write_srt(segs, out_dir, stem)
        q.put(("done", [path]))
    except Exception as e:
        traceback.print_exc()
        q.put(("error", "%s\n%s" % (e, traceback.format_exc())))


# ---------------------------------------------------------------- GUI
class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        root.title("字幕工坊 - 本地离线音频转字幕（提取版）")
        root.geometry("720x560")
        root.minsize(640, 500)

        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(root)
        frm.pack(fill="both", expand=True, **pad)

        # 文件选择
        row1 = ttk.Frame(frm); row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="音频/视频文件:").pack(side="left")
        self.var_file = tk.StringVar()
        ttk.Entry(row1, textvariable=self.var_file).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row1, text="浏览...", command=self.pick_file).pack(side="left")

        # 模型 + GPU
        row2 = ttk.Frame(frm); row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="识别模型:").pack(side="left")
        self.var_model = tk.StringVar(value="medium")
        ttk.Combobox(row2, textvariable=self.var_model, state="readonly",
                     values=["medium", "small"], width=10).pack(side="left", padx=6)
        ttk.Label(row2, text="(medium 精度高 / small 速度快)").pack(side="left")

        self.var_gpu = tk.BooleanVar(value=True)
        self.chk_gpu = ttk.Checkbutton(
            frm, text="使用 NVIDIA 显卡加速 (CUDA) —— 无显卡或加载失败时自动回退 CPU",
            variable=self.var_gpu)
        self.chk_gpu.pack(anchor="w", pady=2)

        # 输出目录
        row3 = ttk.Frame(frm); row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="输出目录:").pack(side="left")
        self.var_out = tk.StringVar(value="")
        ttk.Entry(row3, textvariable=self.var_out).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row3, text="浏览...", command=self.pick_out).pack(side="left")

        # 开始按钮 + 进度
        row4 = ttk.Frame(frm); row4.pack(fill="x", pady=8)
        self.btn = ttk.Button(row4, text="开始生成字幕", command=self.start)
        self.btn.pack(side="left")
        ttk.Button(row4, text="取消", command=self.cancel_run).pack(side="left", padx=8)
        self.var_prog = tk.DoubleVar(value=0)
        self.pb = ttk.Progressbar(row4, variable=self.var_prog, maximum=1.0)
        self.pb.pack(side="left", fill="x", expand=True, padx=10)

        # 日志
        ttk.Label(frm, text="运行日志:").pack(anchor="w")
        self.txt = tk.Text(frm, height=16, state="disabled", wrap="word",
                           font=("Microsoft YaHei UI", 9))
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_config("warn", foreground="#b06000")

        ttk.Label(frm, foreground="#888",
                  text="输出：<音频名>_日文字幕.srt（日语短句 + 精确时间轴，B 站可直接上传）\n"
                       "本工具只做语音识别提取，不做翻译；把生成的日文 SRT 发给 AI 助手精翻中文即可。"
                  ).pack(anchor="w", pady=(6, 0))

        self.root.after(100, self.poll_queue)

    # ---- 交互
    def pick_file(self):
        p = filedialog.askopenfilename(
            title="选择音频或视频文件",
            filetypes=[("媒体文件", " ".join("*" + e for e in SUPPORTED_EXT)),
                       ("所有文件", "*.*")])
        if p:
            self.var_file.set(p)
            if not self.var_out.get():
                self.var_out.set(os.path.dirname(p))

    def pick_out(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.var_out.set(d)

    def start(self):
        path = self.var_file.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showwarning("提示", "请先选择音频/视频文件")
            return
        if not path.lower().endswith(SUPPORTED_EXT):
            messagebox.showwarning("提示", "不支持的文件格式")
            return
        out = self.var_out.get().strip() or os.path.dirname(path)
        os.makedirs(out, exist_ok=True)
        self.btn.config(state="disabled")
        self.cancel.clear()
        self.var_prog.set(0)
        self.log("========== 开始 ==========")
        self.worker = threading.Thread(
            target=run_pipeline,
            args=(path, self.var_model.get(), self.var_gpu.get(), out, self.q, self.cancel),
            daemon=True)
        self.worker.start()

    def cancel_run(self):
        self.cancel.set()

    def log(self, msg, warn=False):
        self.txt.config(state="normal")
        self.txt.insert("end", msg + "\n", "warn" if warn else ())
        self.txt.see("end")
        self.txt.config(state="disabled")

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    self.var_prog.set(payload)
                elif kind == "done":
                    self.btn.config(state="normal")
                    self.var_prog.set(1.0)
                    self.log("========== 完成 ==========")
                    for p in payload:
                        self.log("  生成: " + p)
                    if messagebox.askyesno("完成", "字幕已生成，是否打开输出目录？"):
                        os.startfile(self.var_out.get().strip() or os.path.dirname(payload[0]))
                elif kind == "error":
                    self.btn.config(state="normal")
                    self.log("发生错误: " + payload, warn=True)
                    messagebox.showerror("错误", payload.splitlines()[0])
        except queue.Empty:
            pass
        self.root.after(100, self.poll_queue)


def main():
    # CLI 模式：SubtitleTool.exe <音频文件> [输出目录] [--cpu]
    args = [a for a in sys.argv[1:]]
    if args:
        # Windows GBK 控制台打印日文会崩溃，强制 UTF-8 并容错
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        cli = [a for a in args if not a.startswith("--")]
        use_gpu_cli = "--cpu" not in args
        path = cli[0]
        out = cli[1] if len(cli) > 1 else os.path.dirname(os.path.abspath(path))
        os.makedirs(out, exist_ok=True)
        cancel = threading.Event()
        qq = queue.Queue()

        def drain():
            while True:
                try:
                    kind, payload = qq.get(timeout=1)
                    if kind == "log":
                        try:
                            print(payload, flush=True)
                        except Exception:
                            print(payload.encode("ascii", "replace").decode(), flush=True)
                    elif kind == "progress":
                        try:
                            print("进度: %d%%" % int(payload * 100), flush=True)
                        except Exception:
                            pass
                    elif kind == "done":
                        print("完成！")
                        for p in payload:
                            print("  " + p)
                        return
                    elif kind == "error":
                        print("错误: " + payload, flush=True)
                        sys.exit(1)
                except queue.Empty:
                    cancel_t = threading.current_thread()
                    if not cancel_t.is_alive():
                        return

        t = threading.Thread(target=run_pipeline,
                             args=(path, "medium", use_gpu_cli, out, qq, cancel),
                             daemon=True)
        t.start()
        drain()
        t.join()
        return

    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
