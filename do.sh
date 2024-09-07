# data
python scripts/parse_images.py --in-dir data/train_images --out-dir data/train_images_clip
python scripts/create_metadata_df.py --out-path data/metadata.csv --no-normalize --no-fillna
python scripts/create_spinal_new_coordnates.py


# mount images
mkdir mount
sh mount.sh


# train everything
python scripts/e2e.py