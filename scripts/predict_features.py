from pathlib import Path
import subprocess
import tyro


def main(ckpts_dir: Path):
    for ckpt_dir in sorted(ckpts_dir.glob("*"), key=lambda x: int(x.stem.split("_")[1])):
        print(f"Predicting from {ckpt_dir}")
        config_path = next(ckpt_dir.rglob("*config.yaml"))
        ckpt_path = next(ckpt_dir.rglob("*.ckpt"))
        subprocess.run([
            "python", "scripts/trainer.py", "predict",
            "-c", config_path,
            "--ckpt_path", ckpt_path,
            "--return_predictions", "false"
        ])

if __name__ == "__main__":
    tyro.cli(main)
