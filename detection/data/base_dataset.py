import os
import random

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data
from pedalboard import Gain, Clipping, HighShelfFilter, LowShelfFilter, PitchShift, \
    Reverb, PeakFilter
from transformers import ASTFeatureExtractor


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode="train", ast_proc=False):
        self.ast_processor = None if ast_proc is False else ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        self.config = config['dataset']
        self.data = None
        self.training = mode == "train"
        self.multilingual = self.config['multilingual']
        if self.multilingual:
            self._process_multilingual_dataset()
        else:
            self._process_dataset()
        self.FIXED_LENGTH = 5  # in seconds
        self.window_size, self.overlap = 5, 0  # in seconds

        self.fq_win_size = 320
        self.hop_size = self.fq_win_size // 2

    def _process_dataset(self):
        dataset = self.config['dataset_train']
        if self.training:
            meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "train"]
            meta["dir"] = meta['is_fake'].apply(lambda x: f'{dataset}/real' if x == 0 else f'{dataset}/fake')
        else:
            meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            meta = meta[meta['split'] == "val"]
            meta["dir"] = meta['is_fake'].apply(lambda x: f'{dataset}/real' if x == 0 else f'{dataset}/fake')

        self.data = meta


    def _process_multilingual_dataset(self):
        metas = []
        for dataset in self.config['train_datasets']:
            print(dataset)
            if dataset not in ["commonvoice-ru", "commonvoice-en"]:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            else:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"), sep="\t")
            if self.training:
                meta_aux = meta_aux[meta_aux['split'] == 'train']
            else:
                meta_aux = meta_aux[meta_aux['split'] == 'val']
            meta_aux["dir"] = meta_aux['is_fake'].apply(lambda x: f'{dataset}/real' if x == 0 else f'{dataset}/fake')

            if self.config["extract_samples"]:
                meta_real = meta_aux[meta_aux['is_fake']==0]
                sample_names = meta_real["sample_name"].sample(self.config["num_samples"] // 2)
                meta_aux = meta_aux[meta_aux["sample_name"].isin(sample_names)]

            metas.append(meta_aux)
        self.data = pd.concat(metas, ignore_index=True)


    def __getitem__(self, index):
        meta = self.data.iloc[index]
        label = meta['is_fake']
        sample_path = os.path.join(self.config['root_path'], meta['dir'], meta['sample_name'])

        time_domain, fs = sf.read(sample_path)
        time_domain = self.get_fix_length(time_domain, fs)

        if random.uniform(0, 1) < self.config['augment_chance']:
            time_domain = self.augment_signal(time_domain, index, fs)

        # time_domain = self._split_array(time_domain, fs * self.window_size, self.overlap)
        if self.ast_processor is not None:
            freq_domain = self.ast_processor(time_domain, sampling_rate=fs, return_tensors="np").data['input_values']
        else:
            freq_domain = self._get_freq_features(time_domain)[0]
        return freq_domain, label

    def _split_array(self, x, window_size, overlap):
        num_win = (len(x) - window_size) // (window_size - overlap) + 1
        step = window_size - overlap

        window = np.zeros((num_win, window_size), dtype=x.dtype)
        for i in range(num_win):
            start = i * step
            end = start + window_size
            window[i] = x[start:end]
        return window

    def _get_freq_features(self, x, compress_factor=0.3):
        x = torch.tensor(x).unsqueeze(0)
        x_fq = torch.stft(x, self.fq_win_size, hop_length=self.hop_size,
                          window=torch.hann_window(self.fq_win_size),
                          center=False, onesided=True, return_complex=True).transpose(1, 2)

        x_fq = torch.view_as_real(x_fq)
        x_fq2 = torch.clamp(torch.sum(x_fq * x_fq, dim=-1), min=1e-12)
        x_fq2_u = x_fq2.unsqueeze(-1)

        x_fq = torch.pow(x_fq2_u, (compress_factor-1) / 2.0) * x_fq
        mag = torch.pow(x_fq2_u, compress_factor / 2.0)

        features = torch.cat((mag, x_fq), dim=-1).permute(0, 3, 1, 2)
        return features

    def augment_signal(self, raw_signal, index, fs):
        if self.config['noise_augment'] and self.training:
            if random.uniform(0, 1) < self.config['noise_augment_chance']:
                raw_signal = self.noise_augmentation(raw_signal)

        if self.config['time_augment'] and self.training:
            if random.uniform(0, 1) < self.config['time_augment_chance']:
                raw_signal = np.roll(raw_signal, random.randint(0, int(self.FIXED_LENGTH / 15)))

        if self.config['speed_augment'] and self.training:
            if random.uniform(0, 1) < self.config['speed_augment_chance']:
                raw_signal = self.speed_augment(raw_signal, fs)

        if self.config['mixup_augment'] and self.training:
            if random.uniform(0, 1) < self.config['mixup_augment_chance']:
                same_class_mask = np.array(self.labels) == self.labels[index]
                path = np.random.choice(np.array(self.data)[same_class_mask], 1)[0]
                alpha = np.random.randint(1, 70) / 100.
                raw_signal_mix, fs_ = sf.read(path)
                if fs_ == fs:
                    raw_signal_mix = self.get_fix_length(raw_signal_mix, fs)
                    raw_signal = raw_signal + alpha*raw_signal_mix

        if self.config['volume_augment'] and self.training:
            if random.uniform(0, 1) < self.config['volume_augment_chance']:
                raw_signal = Gain(random.randint(-6, 6))(raw_signal, fs)

        if self.config['clipping'] and self.training:
            if random.uniform(0, 1) < self.config['clipping_augment_chance']:
                raw_signal = Clipping(random.randint(-6, 0))(raw_signal, fs)

        if self.config['reverb'] and self.training:
            if random.uniform(0, 1) < self.config['reverb_augment_chance']:
                raw_signal = Reverb(room_size=random.randint(1, 40) / 100.)(raw_signal, fs)

        if self.config['spectral_shift'] and self.training:
            if random.uniform(0, 1) < self.config['spectral_shift_augment_chance_HP']:
                raw_signal = HighShelfFilter(cutoff_frequency_hz=random.randint(500, 2000),
                                             gain_db=random.randint(5, 20))(raw_signal, fs)
            if random.uniform(0, 1) < self.config['spectral_shift_augment_chance_LP']:
                raw_signal = LowShelfFilter(cutoff_frequency_hz=random.randint(200, 1000),
                                            gain_db=random.randint(5, 20))(raw_signal, fs)
            if random.uniform(0, 1) < self.config['spectral_shift_augment_chance_peak']:
                raw_signal = PeakFilter(cutoff_frequency_hz=random.randint(500, 1500),
                                        gain_db=random.randint(5, 20))(raw_signal, fs)

        if self.config['pitch'] and self.training:
            if random.uniform(0, 1) < self.config['pitch_augment_chance']:
                raw_signal = PitchShift(semitones=random.randint(-40, 40) / 100.)(raw_signal, fs)

        return raw_signal

    def spec_augment(self, spectrogram):
        chance = random.uniform(0, 1)
        if chance < 0.33:
            # Temporal augmentation
            temp_length = np.random.randint(0, spectrogram.shape[1] // 10)
            start_index = np.random.randint(0, spectrogram.shape[1] - temp_length - 1)
            spectrogram[start_index:(start_index + temp_length), :] = 0

        elif chance > 0.66:
            # Frecv augmentation
            freq_length = np.random.randint(0, spectrogram.shape[0] // 10)
            start_index = np.random.randint(0, spectrogram.shape[0] - freq_length - 1)
            spectrogram[:, start_index:(start_index + freq_length)] = 0
        else:
            # Both augmentations
            temp_length = np.random.randint(0, spectrogram.shape[1] // 10)
            start_index = np.random.randint(0, spectrogram.shape[1] - temp_length - 1)
            spectrogram[start_index:(start_index + temp_length), :] = 0
            freq_length = np.random.randint(0, spectrogram.shape[0] // 10)
            start_index = np.random.randint(0, spectrogram.shape[0] - freq_length - 1)
            spectrogram[:, start_index:(start_index + freq_length)] = 0

        return spectrogram

    def noise_augmentation(self, raw_signal):
        if random.uniform(0, 1) < self.config['gaussian_noise_augment_chance']:
            # Set a target SNR
            target_snr_db = np.random.randint(10, 60)
            # Calculate signal power and convert to dB
            sig_avg_watts = np.mean(raw_signal ** 2)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), raw_signal.shape)
            # Noise up the original signal
            raw_signal = raw_signal + noise
        else:
            noise, _ = sf.read(self.back_ground_noise[np.random.randint(0, len(self.back_ground_noise))])
            start_idx = np.random.randint(0, len(noise) - self.FIXED_LENGTH - 1)
            noise = noise[start_idx:start_idx + self.FIXED_LENGTH]

            target_snr_db = np.random.randint(0, 60)
            snr_noise_coeff = (raw_signal ** 2).mean() / ((noise ** 2).mean() * np.power(10, target_snr_db / 20))
            noise = np.sqrt(snr_noise_coeff) * noise
            raw_signal = raw_signal + noise

        return raw_signal

    def speed_augment(self, data, fs):
        speed_factor = np.random.uniform(0.85, 1.15, 1)[0]
        data = librosa.effects.time_stretch(data, rate=speed_factor)
        return self.get_fix_length(data, fs)

    def get_fix_length(self, raw_signal, fs):
        fixed_length = fs * self.FIXED_LENGTH
        if len(raw_signal) == fixed_length:
            return raw_signal

        if len(raw_signal) < fixed_length:
            padding_left = (fixed_length - len(raw_signal)) // 2
            padding_right = fixed_length - len(raw_signal) - padding_left
            raw_signal = np.pad(raw_signal, (padding_left, padding_right))
        elif len(raw_signal) > fixed_length:
            len_difference = np.abs(len(raw_signal) - fixed_length)
            rand_idx = np.random.randint(0, len_difference)
            raw_signal = raw_signal[rand_idx:rand_idx+fixed_length]

        return raw_signal

    def __len__(self):
        return len(self.data)
