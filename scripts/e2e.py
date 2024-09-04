import subprocess
import time
from pathlib import Path


def run(commands):
    print("-" * 40)
    print(" ".join(commands))
    print("-" * 40)
    subprocess.run(commands)
    time.sleep(4)


def get_checkpoint(i):
    return next(Path(f"lightning_logs/version_{i}").rglob("*.ckpt"))


def get_preds(i):
    return next(Path(f"lightning_logs/version_{i}").rglob("*.parquet"))


i = 0
for this_split in range(1, 2):
    # train keypoints, levels
    ckpt_paths = {}
    for model_type in ("keypoints", "levels"):
        run(
            [
                "python",
                "scripts/trainer.py",
                "fit",
                f"-c=configs/{model_type}.yaml",
                f"--data.this_split={this_split}",
            ]
        )
        ckpt_paths[model_type] = get_checkpoint(i)
        i += 1

    # predict keypoints, levels
    preds_paths = {}
    for model_type in ("keypoints", "levels"):
        run(
            [
                "python",
                "scripts/trainer.py",
                "predict",
                f"-c=configs/{model_type}.yaml",
                f"--ckpt_path={ckpt_paths[model_type]}",
                f"--data.this_split={this_split}",
            ]
        )
        preds_paths[model_type] = get_preds(i)
        i += 1

    # train patch
    run(
        [
            "python",
            "scripts/trainer.py",
            "fit",
            "-c=configs/patch.yaml",
            f"--data.keypoints_path={preds_paths['keypoints']}",
            f"--data.this_split={this_split}",
        ]
    )
    ckpt_paths["patch"] = get_checkpoint(i)
    i += 1

    # train sequence
    run(
        [
            "python",
            "scripts/trainer.py",
            "fit",
            "-c=configs/sequence.yaml",
            f"--data.keypoints_path={preds_paths['keypoints']}",
            f"--data.levels_path={preds_paths['levels']}",
            f"--model.backbone.ckpt_path={ckpt_paths['patch']}",
            f"--data.this_split={this_split}",
        ]
    )
    ckpt_paths["sequence"] = get_checkpoint(i)
    i += 1

    # finetune sequence
    run(
        [
            "python",
            "scripts/trainer.py",
            "fit",
            "-c=configs/sequence.yaml",
            "-c=configs/finetune_sequence.yaml",
            f"--data.keypoints_path={preds_paths['keypoints']}",
            f"--data.levels_path={preds_paths['levels']}",
            f"--model.ckpt_path={ckpt_paths['sequence']}",
            f"--data.this_split={this_split}",
            "--trainer.max_epochs=3",
        ]
    )
    ckpt_paths["finetune_sequence"] = get_checkpoint(i)
    i += 1

    for k, ckpt_path in ckpt_paths.items():
        print(k, ckpt_path)
