import os
import argparse
import torch
import torchaudio
from TTS.api import TTS
from utils import resample_clip_to_16khz


def generate_one_sample(sentence, reference_paths, output_path, device):
    temp_wav_path = "temp_demo.wav"

    # Load KNN-VC and TTS models
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True)
    tts = TTS(model_name="tts_models/ro/cv/vits", progress_bar=True).to(device)

    # Resample reference files if needed
    for ref_path in reference_paths:
        resample_clip_to_16khz(ref_path, ref_path)

    # Generate synthetic voice with TTS
    tts.tts_to_file(sentence, file_path=temp_wav_path)
    resample_clip_to_16khz(temp_wav_path, temp_wav_path)

    # Apply KNN-VC to convert TTS to target speaker
    query_seq = knn_vc.get_features(temp_wav_path)
    matching_set = knn_vc.get_matching_set(reference_paths)
    out_wav = knn_vc.match(query_seq, matching_set, topk=4)
    out_wav = out_wav.unsqueeze(0)

    # Save the result
    torchaudio.save(output_path, src=out_wav, sample_rate=16000)
    print(f"Saved generated sample to {output_path}")
    os.remove(temp_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate one voice clone sample using KNN-VC and Arabic TTS")
    parser.add_argument("--sentence", required=True, help="Sentence to synthesize")
    parser.add_argument("--refs", nargs='+', required=True, help="List of reference .wav files (at least 30sec total). "
                                                                 "The voice that we want to clone.")
    parser.add_argument("--output", required=True, help="Path to save the generated audio")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generate_one_sample(
        sentence=args.sentence,
        reference_paths=args.refs,
        output_path=args.output,
        device=device
    )
