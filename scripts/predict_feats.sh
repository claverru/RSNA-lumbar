python scripts/train.py predict \
    -c checkpoints/effnetb4_380_w/version_1/config.yaml \
    --ckpt_path checkpoints/effnetb4_380_w/version_1/checkpoints/epoch=3_val_loss=12.8702.ckpt \
    --return_predictions false

python scripts/train.py predict \
    -c checkpoints/effnetb4_380_w/version_2/config.yaml \
    --ckpt_path checkpoints/effnetb4_380_w/version_2/checkpoints/epoch=3_val_loss=12.3755.ckpt \
    --return_predictions false

python scripts/train.py predict \
    -c checkpoints/effnetb4_380_w/version_3/config.yaml \
    --ckpt_path checkpoints/effnetb4_380_w/version_3/checkpoints/epoch=2_val_loss=11.8393.ckpt \
    --return_predictions false

python scripts/train.py predict \
    -c checkpoints/effnetb4_380_w/version_4/config.yaml \
    --ckpt_path checkpoints/effnetb4_380_w/version_4/checkpoints/epoch=3_val_loss=12.4429.ckpt \
    --return_predictions false

python scripts/train.py predict \
    -c checkpoints/effnetb4_380_w/version_5/config.yaml \
    --ckpt_path checkpoints/effnetb4_380_w/version_5/checkpoints/epoch=3_val_loss=12.4136.ckpt \
    --return_predictions false
