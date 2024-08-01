import subprocess
import time
from pathlib import Path


def run(commands):
    print("-"*40)
    print(" ".join(commands))
    print("-"*40)
    subprocess.run(commands)
    time.sleep(4)


# run(["python", "scripts/trainer.py", "fit", "-c", "configs/keypoints.yaml"])
# ckpt_path = next(Path("lightning_logs/version_0").rglob("*.ckpt"))
# run([
#     "python", "scripts/trainer.py",
#     "predict", "-c", "configs/keypoints.yaml",
#     "--ckpt_path", f"{ckpt_path}",
#     "--data.batch_size", "64"
# ])


# run(["python", "scripts/trainer.py", "fit", "-c", "configs/levels.yaml"])
# ckpt_path = next(Path("lightning_logs/version_2").rglob("*.ckpt"))
# run([
#     "python", "scripts/trainer.py",
#     "predict", "-c", "configs/levels.yaml",
#     "--ckpt_path", f"{ckpt_path}",
#     "--data.batch_size", "64"
# ])

keypoints_path = next(Path("lightning_logs/version_1").rglob("*.parquet"))
levels_path = next(Path("lightning_logs/version_3").rglob("*.parquet"))
for n_layers in (6, ):
    for emb_dim in (256, 512):
        for n_heads in (4, 8):
            subprocess.run(
                [
                    "python", "scripts/trainer.py", "fit",
                    "-c", "configs/patchseq.yaml",
                    "--model.n_layers", f"{n_layers}",
                    "--model.emb_dim", f"{emb_dim}",
                    "--model.emb_dim", f"{emb_dim}",
                    "--model.n_heads", f"{n_heads}",
                    "--data.keypoints_path", f"{keypoints_path}",
                    "--data.levels_path", f"{levels_path}"
                ]
            )
