import os
from pathlib import Path

import pandas as pd



if __name__ == "__main__":
    DATA_DIR = "../SONICS"
    real_dir = Path(DATA_DIR) / "real_songs"
    fake_dir = Path(DATA_DIR) / "fake_songs"

    # Build sets of available filenames (without extension) from disk
    real_files = {
        p.stem for p in real_dir.glob("*.wav")
    }
    fake_files = {
        p.stem for p in fake_dir.glob("*.mp3")
    }

    real_df = pd.read_csv(f"{DATA_DIR}/real_songs.csv", low_memory=False)
    real_df["filepath"] = f"{DATA_DIR}/real_songs/" + real_df.filename + ".wav"
    real_df["target"] = 0
    real_df = real_df[real_df.filename.isin(real_files)]
    real_df = real_df[real_df.filepath.map(os.path.exists)]

    fake_df = pd.read_csv(f"{DATA_DIR}/fake_songs.csv", low_memory=False)
    fake_df["filepath"] = f"{DATA_DIR}/fake_songs/" + fake_df.filename + ".mp3"
    fake_df["target"] = 1
    fake_df = fake_df[fake_df.filename.isin(fake_files)]
    fake_df = fake_df[fake_df.filepath.map(os.path.exists)]

    df = pd.concat([real_df, fake_df])
    df = df[(df.duration >= 30) & (df.no_vocal == False)]

    train_df = df[df.split == 'train']
    train_df.to_csv("train.csv",index=False)

    valid_df = df[df.split == 'valid']
    valid_df.to_csv("valid.csv",index=False)

    test_df = df[df.split == 'test']
    test_df.to_csv("test.csv",index=False)
