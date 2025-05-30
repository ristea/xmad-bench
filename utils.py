import torch
import random
import librosa
import torchaudio


# Convert vtt caption time to seconds
def time_to_seconds(vtt_time):
    hours, minutes, seconds = vtt_time.split(":")
    seconds, milliseconds = map(int, seconds.split("."))
    return int(hours) * 3600 + int(minutes) * 60 + seconds + milliseconds / 1000

def resample_clip_to_16khz(orig_path, target_path):
    tensor, old_sr = torchaudio.load(orig_path)
    if old_sr == 16_000:
        return

    np_arr = tensor.numpy()
    resampled_np = librosa.resample(np_arr, orig_sr=old_sr, target_sr=16000, res_type="soxr_vhq")
    resampled_tensor = torch.from_numpy(resampled_np)
    torchaudio.save(target_path, src=resampled_tensor, sample_rate=16000)

#take random reference samples from speaker from validated dataset, and from other if not enough
def gather_speaker_references(df_validated, df_other, client_id, refs_cache, number=15):
    if client_id in refs_cache:
        validated = refs_cache[client_id]["validated"]
        other = refs_cache[client_id]["other"]
    else:
        validated = []
        other = []
        for index, row in df_validated.iterrows():
            if row["client_id"] == client_id:
                validated.append(row["path"])
        #if there are not enough samples, search also in df_other
        if len(validated) < number:
            for index, row in df_other.iterrows():
                if row["client_id"] == client_id:
                    other.append(row["path"])
        refs_cache[client_id] = {"validated":validated, "other":other}

    if len(validated) < number :
        #take random samples also from df_other
        res = validated + random.sample(other, min(number-len(validated), len(other)))
    else:
        res = random.sample(validated, number)
    return res
