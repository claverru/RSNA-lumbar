python scripts/trainer.py fit -c configs/image.yaml --data.this_split 0 --data.n_splits 5
python scripts/trainer.py fit -c configs/image.yaml --data.this_split 1 --data.n_splits 5
python scripts/trainer.py fit -c configs/image.yaml --data.this_split 2 --data.n_splits 5
python scripts/trainer.py fit -c configs/image.yaml --data.this_split 3 --data.n_splits 5
python scripts/trainer.py fit -c configs/image.yaml --data.this_split 4 --data.n_splits 5

python scripts/image/predict_features.py --ckpts-dir lightning_logs
python scripts/image/merge_features.py --ckpts-dir lightning_logs

python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 0 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 1 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 2 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 3 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 4 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 5 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 6 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 7 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 8 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
python scripts/trainer.py fit -c configs/sequence.yaml --data.this_split 9 --data.n_splits 10 --data.feats_path lightning_logs/feats.parquet
