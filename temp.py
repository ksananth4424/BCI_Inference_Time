import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

python - <<'PY'
import urllib.request
from pathlib import Path
from bci_src.data.benchmarks.registry import get_benchmark_adapter

N = 500   # smoke first
SEED = 42
OUT = Path("/data1/DATA/cs22btech11030/datasets/coco/val2014")
OUT.mkdir(parents=True, exist_ok=True)

def is_valid_jpeg(path: Path) -> bool:
    try:
        if path.stat().st_size < 1024:
            return False
        with open(path, "rb") as f:
            return f.read(2) == b"\xff\xd8"
    except Exception:
        return False

adapter = get_benchmark_adapter("pope")
samples = adapter.load_samples(n=N, seed=SEED)
ids = sorted({int(s.image_id) for s in samples if str(s.image_id).isdigit()})
print("samples:", len(samples), "unique_images:", len(ids))

downloaded = 0
skipped_valid = 0
failed = 0

for i in ids:
    fn = f"COCO_val2014_{i:012d}.jpg"
    dst = OUT / fn
    if dst.exists() and is_valid_jpeg(dst):
        skipped_valid += 1
        continue

    url = f"http://images.cocodataset.org/val2014/{fn}"
    tmp = dst.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            if r.status != 200:
                failed += 1
                continue
            data = r.read()

        if len(data) < 1024 or data[:2] != b"\xff\xd8":
            failed += 1
            continue

        with open(tmp, "wb") as f:
            f.write(data)
        tmp.replace(dst)
        downloaded += 1
    except Exception:
        failed += 1
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

print("downloaded:", downloaded, "skipped_valid:", skipped_valid, "failed:", failed)
PY

# rerun smoke
python scripts/run_phase2_experiment.py \
  --exp E12b \
  --config configs/phase2/e12b_pope_confidence_gated_e1_tuned.yaml \
  --output-dir results/pope_e12b_popular \
  --device cuda:3 \
  --n-samples 500