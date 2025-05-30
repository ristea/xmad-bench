import logging
import os

import pandas as pd
import torch
import torchaudio
from TTS.api import TTS
from tqdm import tqdm

from utils import gather_speaker_references, resample_clip_to_16khz


def main(data_path, device):
    temp_wav_path = "temp.wav"

    # TSV files with metadata
    df_train = pd.read_csv(os.path.join(data_path, "train.tsv"), sep="\t")
    df_dev = pd.read_csv(os.path.join(data_path, "dev.tsv"), sep="\t")
    df_test = pd.read_csv(os.path.join(data_path, "test.tsv"), sep="\t")

    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    model_name = "tts_models/ara/fairseq/vits"
    tts = TTS(model_name= model_name, progress_bar=False).to(device)
    refs_cache = {} #cache of audio references for speakers

    for df in [df_train, df_dev, df_test]:
        logging.info("\n Starting a new dataframe! \n")

        for index, row in tqdm(df.iterrows()):
            try:
                fake_path = os.path.join(data_path, "fake_fairseq_knnvc", row["path"])

                if os.path.exists(fake_path):
                    continue

                # gather 30 reference samples for the speaker. KNN-VC needs at least 30sec to do the conversion
                no_ref_samples = 30
                references_filenames = gather_speaker_references(df_train, df_dev, row["client_id"],
                                                                 refs_cache, number=no_ref_samples)
                if len(references_filenames) < no_ref_samples:
                    logging.info(f"Not enough references for {row['path']} voice : {len(references_filenames)} samples")
                    continue

                if row["path"] not in references_filenames:
                    references_filenames.append(row["path"])

                for filename in references_filenames:
                    clip_path = os.path.join(data_path, "clips-wav-16khz", filename)
                    resample_clip_to_16khz(clip_path, clip_path)

                ref_wav_paths = [os.path.join(data_path, "clips-wav-16khz", filename) for filename in references_filenames]

                # use TTS model with downsample
                tts.tts_to_file(row["sentence"], file_path=temp_wav_path)
                resample_clip_to_16khz(temp_wav_path, temp_wav_path)

                # use knn-vc model
                query_seq = knn_vc.get_features(temp_wav_path)
                matching_set = knn_vc.get_matching_set(ref_wav_paths)
                out_wav = knn_vc.match(query_seq, matching_set, topk=4)
                out_wav = out_wav.unsqueeze(0)
                torchaudio.save(fake_path, src=out_wav, sample_rate=16000)

            except Exception as e:
                logging.info(f"Unknown exception on file {row['path']} : {e}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_path = "../../../datasets/cv-corpus-20.0-2024-12-06-ar/cv-corpus-20.0-2024-12-06/ar"

    main(data_path=ar_path, device=device)