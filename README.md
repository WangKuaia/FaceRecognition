# FaceRecognition

快速开始：用 Swin-Large + ArcFace 做 embedding 抽取与人脸相似度比较。

注意：PyTorch 的安装因 CUDA 版本而异，推荐根据你的平台使用官方安装命令（https://pytorch.org/）。下面的步骤假设你已经安装好合适的 torch + torchvision（或使用 CPU 版本）。

## 环境依赖
先安装系统依赖（可选）：

- Python 3.9+（推荐 3.10/3.11）
- pip

安装 Python 包（注意 PyTorch）：

1) 首先根据你的 CUDA/CPU 在 https://pytorch.org/ 选择并安装 PyTorch + torchvision。

2) 安装其余依赖：

```bash
pip install -r requirements.txt
```

（requirements.txt 中包含大部分常用依赖；如果你在安装 torch 时遇到问题，请先按照 PyTorch 官网说明安装匹配的 wheel，再运行上面命令）

## 下载模型权重（示例）
仓库包含 scripts/download_checkpoint.py，支持从 Hugging Face、Google Drive 或任意 URL 下载检查点文件。

示例：

- 从 Hugging Face 仓库下载：

```bash
python scripts/download_checkpoint.py --hf <hf_repo_id> <filename.pth> --out-dir checkpoints
```

- 从 Google Drive 下载：

```bash
python scripts/download_checkpoint.py --gdrive <gdrive_id_or_url> --out checkpoints/model.pth
```

- 直接从 URL 下载：

```bash
python scripts/download_checkpoint.py --url https://example.com/model.pth --out checkpoints/model.pth
```

（把实际可用的 checkpoint 链接/ID 写到这里，或将 checkpoints/ 下放入你自己的权重文件）

## 快速推理示例
仓库提供两个推理脚本：

1) 使用 InsightFace 的人脸 embedding 并比较两张图像（无需本仓库的模型）：

```bash
# 安装 insightface 等
pip install insightface opencv-python numpy
python scripts/infer_insightface.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg --device cpu
```

2) 使用本仓库的 Swin-Large backbone 抽取 embedding（需要 checkpoint）：

```bash
# 假设你已经有 checkpoints/model.pth
python scripts/infer_swin_arcface.py --checkpoint checkpoints/model.pth --img path/to/image.jpg --device cpu

# 或批量处理目录并保存 embedding
python scripts/infer_swin_arcface.py --checkpoint checkpoints/model.pth --img-dir data/faces/ --out-dir out/embs --device cpu
```

��本说明：
- scripts/infer_swin_arcface.py 会构建 backbone（不从 timm 下载预训练权重），并尝试从你的 checkpoint 中加载 backbone 的 state_dict。脚本会输出 L2 归一化 embedding 并将其保存为 .npy 文件（或打印到 stdout）。
- 若你的 checkpoint 格式与脚本不匹配（不同的 key 前缀或嵌套结构），可以手动提取 backbone 的 state_dict：

```python
# 保存 backbone state dict 示例
torch.save(backbone.state_dict(), 'backbone.pth')
```

## 常见问题
- PyTorch 版本/CUDA：请优先按 https://pytorch.org/ 的安装说明安装匹配系统的 PyTorch wheel（例如你有 CUDA 12.1/11.8 等）。
- checkpoint 加载失败：查看 checkpoint 的 keys（torch.load(checkpoint).keys()），并调整脚本中加载逻辑或先把 state_dict 导出为纯净格式。

## 下一步建议
- 在 README 中写入你要使用的 checkpoint 链接或 Hugging Face repo id
- 补充对训练脚本、数据预处理、评估指标的文档

