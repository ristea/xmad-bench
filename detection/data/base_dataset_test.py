import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data
from transformers import ASTFeatureExtractor


class BaseDatasetTest(torch.utils.data.Dataset):
    def __init__(self, config, ast_proc=False):
        self.ast_processor = None if ast_proc is False else ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.config = config['dataset']
        self.data = None
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
        dataset = self.config['dataset_test']
        meta = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
        meta["dir"] = meta['is_fake'].apply(lambda x: f'{dataset}/real' if x == 0 else f'{dataset}/fake')
        self.data = meta


    def _process_multilingual_dataset(self):
        metas = []
        for dataset in self.config['test_datasets']:
            if dataset not in ["mailabs-ru", "mailabs-en"]:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"))
            else:
                meta_aux = pd.read_csv(os.path.join(self.config['root_path'], dataset, "meta.csv"), sep="\t")
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
