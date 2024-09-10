import subprocess
import time
from pathlib import Path
import json
import os
import sys

def run(commands):
    print("-" * 40)
    print(" ".join(commands))
    print("-" * 40)
    result = subprocess.run(commands, text=True, check=False)
    time.sleep(4)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}: {' '.join(commands)}")
        return False
    return True

def get_latest_file(model_type, split, file_type):
    if "ckpt" in file_type:
        pattern = f"{model_type}_split{split}/version_*/checkpoints/*.{file_type}"
    else:
        pattern = f"{model_type}_split{split}/version_*/*.{file_type}"
    files = list(Path("lightning_logs").rglob(pattern))
    return max(files, key=os.path.getctime) if files else None

def save_progress(progress):
    with open("e2e_progress.json", "w") as f:
        json.dump(progress, f, indent=4)

def load_progress():
    if os.path.exists("e2e_progress.json"):
        with open("e2e_progress.json", "r") as f:
            return json.load(f)
    return {"last_completed_split": -1, "completed_steps": {}, "ckpt_paths": {}, "preds_paths": {}}


def generate_command(action, model_type, split, extra_args=None, config_name=None):
    config = f"-c=configs/{model_type}.yaml" if config_name is None else f"-c=configs/{config_name}.yaml"
    cmd = [
        "python", "scripts/trainer.py", action,
        config,
        f"--data.this_split={split}",
        f"--trainer.default_root_dir=lightning_logs/{model_type}_split{split}",
        f"--trainer.logger.save_dir=lightning_logs/{model_type}_split{split}"
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd

def execute_step(progress, step_name, command):
    if step_name not in progress["completed_steps"]:
        success = run(command)
        if not success:
            print(f"Step '{step_name}' failed. Stopping execution.")
            return False
        progress["completed_steps"][step_name] = True
        save_progress(progress)
    return True

def process_split(this_split, progress):
    ckpt_paths = progress.get("ckpt_paths", {})
    preds_paths = progress.get("preds_paths", {})
    
    # Train and predict for keypoints and levels
    for model_type in ("keypoints", "levels"):
        if not execute_step(progress, f"{model_type}_train_{this_split}", 
                     generate_command("fit", model_type, this_split)):
            return None
        
        ckpt_paths[f"{model_type}_{this_split}"] = str(get_latest_file(model_type, this_split, "ckpt"))
        
        if not execute_step(progress, f"{model_type}_predict_{this_split}", 
                     generate_command("predict", model_type, this_split, 
                                      [f"--ckpt_path={ckpt_paths[f'{model_type}_{this_split}']}"])):
            return None
        
        preds_paths[f"{model_type}_{this_split}"] = str(get_latest_file(model_type, this_split, "parquet"))

    # Train patch
    if not execute_step(progress, f"patch_train_{this_split}", 
                 generate_command("fit", "patch", this_split, 
                                  [f"--data.keypoints_path={preds_paths[f'keypoints_{this_split}']}"])):
        return None
    
    ckpt_paths[f"patch_{this_split}"] = str(get_latest_file("patch", this_split, "ckpt"))

    # Train sequence
    if not execute_step(progress, f"sequence_train_{this_split}", 
                 generate_command("fit", "sequence", this_split, [
                     f"--data.keypoints_path={preds_paths[f'keypoints_{this_split}']}",
                     f"--data.levels_path={preds_paths[f'levels_{this_split}']}",
                     f"--model.backbone.ckpt_path={ckpt_paths[f'patch_{this_split}']}"
                 ])):
        return None
    
    ckpt_paths[f"sequence_{this_split}"] = str(get_latest_file("sequence", this_split, "ckpt"))

    # Finetune sequence
    if not execute_step(progress, f"finetune_sequence_{this_split}", 
                 generate_command("fit", "finetune_sequence", this_split, [
                     f"-c=configs/finetune_sequence.yaml",
                     f"--data.keypoints_path={preds_paths[f'keypoints_{this_split}']}",
                     f"--data.levels_path={preds_paths[f'levels_{this_split}']}",
                     f"--model.ckpt_path={ckpt_paths[f'sequence_{this_split}']}",
                     "--trainer.max_epochs=2"
                 ], "sequence")):  # keeping this a ssequence which is overwritten by finetune_sequence
        return None
    
    ckpt_paths[f"finetune_sequence_{this_split}"] = str(get_latest_file("finetune_sequence", this_split, "ckpt"))

    progress["ckpt_paths"] = ckpt_paths
    progress["preds_paths"] = preds_paths
    save_progress(progress)

    return ckpt_paths


def main():
    progress = load_progress()
    last_completed_split = progress["last_completed_split"]

    for this_split in range(last_completed_split + 1, 5):
        ckpt_paths = process_split(this_split, progress)
        
        if ckpt_paths is None:
            print(f"Execution stopped due to an error in split {this_split}")
            sys.exit(1)
        
        progress["last_completed_split"] = this_split
        save_progress(progress)

        for k, ckpt_path in ckpt_paths.items():
            print(k, ckpt_path)

    print("E2E process completed successfully.")

if __name__ == "__main__":
    main()