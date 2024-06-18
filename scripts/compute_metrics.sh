rm metrics.json
echo '{"mean": 0.0}' > metrics.json

python scripts/compute_metrics.py --ckpt-dir lightning_logs/version_0
python scripts/compute_metrics.py --ckpt-dir lightning_logs/version_1
python scripts/compute_metrics.py --ckpt-dir lightning_logs/version_2
python scripts/compute_metrics.py --ckpt-dir lightning_logs/version_3
python scripts/compute_metrics.py --ckpt-dir lightning_logs/version_4
