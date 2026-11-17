# SCW-Pose

SCW-Pose estimates two walnut endpoints and their undirected axis using semantic
endpoint refinement and contour-guided coordinate correction.

This repository contains the complete inference model, its trained weights,
the **118-image test set (239 walnut instances)**, and evaluation scripts.

## Install

Tested with **Python 3.13**, PyTorch 2.9.1, torchvision 0.24.1, and
Ultralytics 8.4.188. The pinned dependencies require Python 3.12 or newer.

```bash
git clone --depth 1 https://github.com/dreamerYCJ/scwpose.git
cd scwpose
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

For an NVIDIA GPU, install the tested CUDA 12.8 PyTorch wheels first:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

For CPU inference, install the dependencies directly:

```bash
python -m pip install -r requirements.txt
```

The weights and test images are included in the repository. No additional
checkpoint download is required. The checkout is approximately 400 MB.

## Evaluate the test set

```bash
python evaluate.py --device auto
```

`auto` uses the first CUDA GPU when available and otherwise uses the CPU.
Use `--device cpu` or `--device cuda:0` to select a device explicitly.
Results are printed and saved to `runs/evaluation.json`.

The default evaluation uses an input size of 1280, box confidence 0.30,
NMS IoU 0.55, and one-to-one greedy matching at IoU 0.50. The endpoint
correspondence with the smaller total distance is used. Endpoint distances
are normalized by the annotated bounding-box diagonal; angular errors treat
directions separated by 180 degrees as equivalent. There is no additional
keypoint-confidence filter. Unmatched instances are reported and excluded
from localization-error statistics.

Expected results with the supplied checkpoint:

| Metric | Value |
| --- | ---: |
| Matched instances | 239 / 239 |
| Axis MAE | approximately 1.466 degrees |
| Endpoint NME | approximately 1.558% |
| PCK@0.05 | 95.40% |

Small floating-point differences may occur across devices. The script also
reports angular RMSE and P95, and the endpoint and axis evaluation counts.

## Predict on an image

```bash
python predict.py --source data/test/images/lab1_016.png --device auto
```

An annotated PNG and a JSON file containing boxes, scores, and endpoint
coordinates are saved to `runs/predictions/`. `--source` also accepts an image
directory. Output endpoint coordinates are in original-image pixels; `p1`
and `p2` identify the two prediction channels, and their axis is undirected.

For another YOLO-format test set:

```bash
python evaluate.py --data path/to/test --weights weights/scwpose.pt --device auto
```

The data directory must contain `images/` and `labels/`. See
[the annotation format](data/test/README.md).

## Files

```text
scwpose/             Model definition and inference
weights/scwpose.pt   Complete trained SCW-Pose checkpoint
data/test/           Test images and endpoint annotations
evaluate.py          Endpoint and axis evaluation
predict.py           Image inference and visualization
requirements.txt     Runtime dependencies
```

## License

The code is released under [AGPL-3.0](LICENSE). SCW-Pose uses Ultralytics;
see [third-party notices](THIRD_PARTY_NOTICES.md).
