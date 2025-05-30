import torch
import os
from TTS.api import TTS
import pandas as pd
from pydub import AudioSegment
import webvtt
from utils import time_to_seconds, resample_clip_to_16khz
import logging
from tqdm import tqdm


def write_to_detection_file(row, clip_id, text, start, end, columns_det, detection_file_path):
    row['text'] = text
    row['id'] = clip_id
    row['start'] = start
    row['end'] = end
    data = {col: [row[col]] for col in columns_det}
    new_df = pd.DataFrame(data)
    new_df.to_csv(detection_file_path, mode="a", index=False, header=False, sep="\t")


def main(data_path, device, language):
    # CSV files with metadata (filtered for music content)
    df_train = pd.read_csv(os.path.join(data_path, "subsets/clean_train_meta_filtered.csv"))
    df_dev = pd.read_csv(os.path.join(data_path, "subsets/clean_dev_meta_filtered.csv"))
    df_test = pd.read_csv(os.path.join(data_path, "subsets/clean_test_meta_filtered.csv"))

    logging.info("Starting the generation!")


    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name=model_name, progress_bar=False).to(device)

    # Every video is split into clips for every caption.
    # Up to 7 minutes of audio from each video is used.
    # Then, fake audio is generated for each clip.
    def generate_df(df, split_name='train'):
        logging.info("\n Starting a new dataframe! \n")

        # Create {split}_detection.tsv file
        detection_file_path = os.path.join(data_path, f'subsets/{split_name}_detection.tsv')

        # start and end timestamps in original video
        columns_det = ['id', 'text', 'channel_id', 'video_id', 'category', 'country',
                       'dialect', 'gender', 'start', 'end']

        df_det = pd.DataFrame(columns=columns_det)
        df_det.to_csv(detection_file_path, sep="\t", index=False)

        # Take every video, split into clips and generate fake audio
        for index, row in tqdm(df.iterrows()):
            video_id = row['video_id']
            video_wav_filename = video_id + '.wav'
            video_path = os.path.join(data_path, f'audios/{video_wav_filename}')
            vtt_file_path = os.path.join(data_path, f'subtitles/{video_id}.ar.vtt')

            try:
                audio = AudioSegment.from_wav(video_path)
                clip_count = 0
                # take up to 7 minutes from every video
                for caption in webvtt.read(vtt_file_path).iter_slice(start='00:00:00.000', end='00:07:00.000'):
                    start_vtt = time_to_seconds(caption.start)
                    end_vtt = time_to_seconds(caption.end)
                    duration = end_vtt - start_vtt
                    text = caption.text

                    if duration < 4 or len(text) > 160:
                        continue
                    else:
                        clip_count +=1

                    audio_segment = audio[start_vtt * 1000:end_vtt * 1000]  # Convert to milliseconds
                    clip_id = f'{video_id}_{clip_count}'
                    clip_wav_filename = clip_id + '.wav'
                    clip_wav_path = os.path.join(data_path, 'real', clip_wav_filename)
                    audio_segment.export(clip_wav_path, format='wav')
                    fake_path = os.path.join(data_path, 'fake-xttsv2', clip_wav_filename)
                    if os.path.exists(fake_path):
                        continue

                    # use TTS model with downsample
                    tts.tts_to_file(text, speaker_wav=clip_wav_path, language=language, file_path=fake_path)
                    resample_clip_to_16khz(fake_path, fake_path)
                    write_to_detection_file(row, clip_id, text, start=start_vtt, end=end_vtt,
                                            columns_det=columns_det, detection_file_path=detection_file_path)
            except Exception:
                logging.info(f"Some error for file: {row["path"]}")

    generate_df(df_train, split_name='train')
    generate_df(df_dev, split_name='dev')
    generate_df(df_test, split_name='test')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_path = "../../../datasets/masc"
    language = 'ar'

    main(data_path=ar_path, device=device, language=language)