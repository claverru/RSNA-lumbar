import subprocess
import time
from pathlib import Path


def run(commands):
    print("-" * 40)
    print(" ".join(commands))
    print("-" * 40)
    subprocess.run(commands)
    time.sleep(4)


# train
ckpt_paths = {}
for i, model_type in enumerate(("keypoints", "levels", "patch")):
    run(["python", "scripts/trainer.py", "fit", f"-c=configs/{model_type}.yaml"])
    ckpt_paths[model_type] = next(Path(f"lightning_logs/version_{i}").rglob("*.ckpt"))


# predict
for model_type in ("keypoints", "levels"):
    run(
        [
            "python",
            "scripts/trainer.py",
            "predict",
            f"-c=configs/{model_type}.yaml",
            f"--ckpt_path={ckpt_paths[model_type]}",
            "--data.batch_size=64",
        ]
    )


# train sequence
keypoints_path = next(Path("lightning_logs/version_3").rglob("*.parquet"))
levels_path = next(Path("lightning_logs/version_4").rglob("*.parquet"))
run(
    [
        "python",
        "scripts/trainer.py",
        "fit",
        "-c=configs/sequence.yaml",
        f"--data.keypoints_path={keypoints_path}",
        f"--data.levels_path={levels_path}",
        f"--model.image.ckpt_path={ckpt_paths['patch']}",
    ]
)


# finetune sequence
sequence_ckpt_path = next(Path("lightning_logs/version_5").rglob("*.ckpt"))
run(
    [
        "python",
        "scripts/trainer.py",
        "fit",
        "-c=configs/finetune_sequence.yaml",
        f"--data.keypoints_path={keypoints_path}",
        f"--data.levels_path={levels_path}",
        f"--model.ckpt_path={sequence_ckpt_path}",
    ]
)
