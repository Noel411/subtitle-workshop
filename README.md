# 字幕工坊 SubtitleTool

本地离线的音频/视频 → 日语 SRT 字幕提取工具，专为**耳语 / ASMR 音频**调优。

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（medium 模型），支持 NVIDIA 显卡加速，全程本地运行、不联网上传任何数据。

```
音频文件 ──► faster-whisper 转录 ──► 幻觉过滤/去重 ──► ≤18字短句断句 ──► 日语 SRT
```

## 它解决什么问题

直接用 Whisper 给耳语音频出字幕，通常会遇到三大痛点，本工具逐一处理：

| 痛点 | 处理方式 |
|---|---|
| 耳语、气音片段被 VAD 大量漏识别 | **默认关闭 VAD 预过滤**，宁可多识别再过滤 |
| 静音段幻觉输出（复读歌词/广告词） | 语速合理性检测 + 单字符碎片清理 + 相邻重复去重 |
| 长句字幕挂屏、断句糟糕 | ≤18 字短句断句，标点优先、助词软断点，词级时间戳精确分摊 |

输出为标准 SRT（UTF-8 with BOM），B 站等平台可直接上传。

**注意：本工具只做"提取"，不做翻译。** 机翻 ASMR 口语质量很差，推荐的工作流是：本工具生成日文 SRT → 交给 AI 助手（如 WorkBuddy/ChatGPT 等）逐条精翻中文。

## 下载使用（推荐，无需装环境）

到 [Releases](../../releases) 下载完整离线包（内置 medium 模型，下载解压即用）：

- 完整包体积约 4GB，为满足 GitHub 单文件 2GB 限制做了分卷：`SubtitleTool-win64.zip.001` / `.002`
- **合并方法（Windows）**：把两个分卷放到同一目录，命令行执行
  ```
  copy /b SubtitleTool-win64.zip.001 + SubtitleTool-win64.zip.002 SubtitleTool-win64.zip
  ```
  然后解压 `SubtitleTool-win64.zip`

### 图形界面

双击 `SubtitleTool.exe`：选择音频 → 勾选"使用 NVIDIA 显卡加速" → 开始。

### 命令行

```
SubtitleTool.exe <音频/视频文件> [输出目录] [--cpu]
```

`--cpu` 强制用 CPU。实测参考：3 小时音频 GPU 约十几分钟，纯 CPU 数倍于此。

### 显卡要求

- NVIDIA 显卡即可自动启用 CUDA 加速（加载失败会自动回退 CPU，不用改设置）
- 完整包已内置 cuDNN / cuBLAS 运行库，无需另装 CUDA Toolkit
- RTX 50 系（Blackwell）需较新的驱动；完整包内置的运行库已适配

## 从源码运行

```bash
pip install -r requirements.txt
python app.py            # GUI
python app.py 某音频.mp3 输出目录   # CLI
```

- 仓库不含模型文件。首次运行会自动从 **hf-mirror.com**（国内镜像，已内置）下载 medium 模型，也可自行下载后放到程序旁 `models/faster-whisper-medium/`
- GPU 加速需额外安装运行库：
  ```bash
  pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
  ```

## 自己打包 exe

```bash
pip install -r requirements.txt pyinstaller
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12   # 可选，打包 GPU 运行库
bash build.sh
```

产物在 `dist/SubtitleTool/`。

## 项目结构

```
app.py        全部逻辑（GUI + 转录 + 过滤 + 断句 + SRT 输出），单文件
build.sh      PyInstaller 打包脚本
```

## 已知局限

- 识别语言固定为日语（`language="ja"`），这是目标场景
- 幻觉过滤依赖语速启发式，极慢速的清晰语音可能被误删（概率很低，删掉的那几条通常也无意义）
- 断句规则针对日语助词设计，换语言需要调整 `_SOFT_BREAK`

## 免责声明

本工具仅进行本地语音识别，不包含、不分发任何音频内容。请仅对您拥有相应权利的音频使用；请支持正版（如 DLsite 等平台的作品）。

## License

[MIT](LICENSE)
