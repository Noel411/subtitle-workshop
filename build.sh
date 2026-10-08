#!/bin/bash
# 打包脚本：生成 dist/SubtitleTool/ 完整离线发行包
# 用法: bash build.sh
# 依赖: Python 3.10+，并已执行 pip install -r requirements.txt pyinstaller
set -e
cd "$(dirname "$0")"
PY=python

# 1. PyInstaller 打包（windowed GUI exe）
"$PY" -m PyInstaller \
  --noconfirm \
  --clean \
  --name SubtitleTool \
  --noconsole \
  --collect-all faster_whisper \
  --collect-all ctranslate2 \
  --collect-all tokenizers \
  --collect-all av \
  --hidden-import huggingface_hub \
  app.py

DIST=dist/SubtitleTool
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")

# 2. 复制模型（离线运行的关键）
#    models/ 下没有模型也能跑：首次运行会自动从 hf-mirror.com 下载
rm -rf "$DIST/models"
mkdir -p "$DIST/models"
for m in faster-whisper-medium faster-whisper-small; do
  if [ -d "models/$m" ]; then
    cp -r "models/$m" "$DIST/models/"
  fi
done

# 3. 复制 CUDA 运行库（GPU 加速）
#    需要 pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
if [ -d "$SP/nvidia" ]; then
  rm -rf "$DIST/cudnn" "$DIST/cublas"
  mkdir -p "$DIST/cudnn" "$DIST/cublas"
  cp -L "$SP"/nvidia/cudnn/bin/*.dll "$DIST/cudnn/" 2>/dev/null || true
  cp -L "$SP"/nvidia/cublas/bin/*.dll "$DIST/cublas/" 2>/dev/null || true
  # cudnn_adv64_9.dll (271MB) 经实测非必需，剔除
  rm -f "$DIST/cudnn/cudnn_adv64_9.dll"
fi

# 4. 清理构建中间产物，保留 dist
rm -rf build SubtitleTool.spec

echo "=== 打包完成 ==="
du -sh "$DIST"
ls "$DIST"
