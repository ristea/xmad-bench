import torch
import pandas as pd
import os
import logging

from TTS.api import TTS
from tqdm import tqdm
from utils import resample_clip_to_16khz


def main(data_path, device):
    temp_wav_path = "temp.wav"

    # TSV files with metadata
    # These files were obtained from original dataset splits by filtering clips that failed other generation methods
    df_train = pd.read_csv(os.path.join(data_path, "train_detection.tsv"), sep="\t")
    df_dev = pd.read_csv(os.path.join(data_path, "dev_detection.tsv"), sep="\t")
    df_test = pd.read_csv(os.path.join(data_path, "test_detection.tsv"), sep="\t")

    logging.info("Starting the generation!")

    vc_freevc = TTS("voice_conversion_models/multilingual/vctk/freevc24").to(device)
    model_name = "tts_models/ro/cv/vits"
    tts = TTS(model_name=model_name, progress_bar=False).to(device)

    for df in [df_train, df_dev, df_test]:
        logging.info("\n Starting a new dataframe! \n")

        for index, row in tqdm(df.iterrows()):
            try:
                wav_path = os.path.join(data_path, "clips-wav-16khz", row["path"])
                fake_path = os.path.join(data_path, "fake_vits_freevc", row["path"])
                if os.path.exists(fake_path):
                    continue

                # use TTS model with downsample
                tts.tts_to_file(row["sentence"], file_path=temp_wav_path)
                resample_clip_to_16khz(temp_wav_path, temp_wav_path)

                # use freevc voice conversion model
                vc_freevc.voice_conversion_to_file(
                    source_wav=temp_wav_path,
                    target_wav=wav_path,
                    file_path=fake_path
                )
                resample_clip_to_16khz(fake_path, fake_path)
                os.remove(temp_wav_path)

            except Exception:
                logging.info(f"Some error for file: {row["path"]}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_path = "../../../datasets/cv-corpus-20.0-2024-12-06-ro"

    main(data_path=ar_path, device=device)

