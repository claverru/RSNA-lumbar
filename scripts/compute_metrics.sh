python scripts/val_inference.py --ckpt-dir checkpoints/effnetb4_380_w/version_1
python scripts/val_inference.py --ckpt-dir checkpoints/effnetb4_380_w/version_2
python scripts/val_inference.py --ckpt-dir checkpoints/effnetb4_380_w/version_3
python scripts/val_inference.py --ckpt-dir checkpoints/effnetb4_380_w/version_4
python scripts/val_inference.py --ckpt-dir checkpoints/effnetb4_380_w/version_5

python scripts/compute_metrics.py --ckpt-dir checkpoints/effnetb4_380_w/version_1
python scripts/compute_metrics.py --ckpt-dir checkpoints/effnetb4_380_w/version_2
python scripts/compute_metrics.py --ckpt-dir checkpoints/effnetb4_380_w/version_3
python scripts/compute_metrics.py --ckpt-dir checkpoints/effnetb4_380_w/version_4
python scripts/compute_metrics.py --ckpt-dir checkpoints/effnetb4_380_w/version_5
