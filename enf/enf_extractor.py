import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Any, Mapping

import numpy as np
from scipy.signal import butter, filtfilt, get_window
from numpy.fft import rfft, rfftfreq


# -----------------------------
# Config
# -----------------------------

@dataclass
class ENFConfig:
    """
    Configuration for ENF extraction.
    Can be constructed directly, from a dict, or from JSON/YAML files.
    """
    nominal_freq: float = 50.0          # 50 or 60 Hz
    harmonic: int = 2                   # which harmonic to use (1 = fundamental, 2 = 2nd harmonic, etc.)
    band_margin_hz: float = 5.0         # +/- around harmonic center for band-pass
    search_width_hz: float = 1.0        # +/- around harmonic center for peak search
    window_sec: float = 5.0             # STFT window length in seconds
    hop_sec: float = 1.0                # hop between successive windows in seconds
    bp_order: int = 4                   # band-pass filter order
    smooth_window: int = 5              # moving average window for smoothing (0 or 1 to disable)
    outlier_threshold_hz: float = 0.5   # discard values too far from nominal if > 0 (0 to disable)

    # ---------- dict / JSON / YAML helpers ----------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ENFConfig":
        """
        Create ENFConfig from a plain dict. Extra keys are ignored.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- JSON ----

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ENFConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json_file(self, path: str | Path, indent: int = 2) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    # ---- YAML (optional, requires pyyaml) ----

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "ENFConfig":
        """
        Load config from a YAML file.

        Requires `pyyaml`:
            pip install pyyaml
        """
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML config. Install with `pip install pyyaml`."
            ) from e

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml_file(self, path: str | Path) -> None:
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML config. Install with `pip install pyyaml`."
            ) from e

        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)


# -----------------------------
# ENF Extractor
# -----------------------------

class ENFExtractor:
    """
    ENF (Electrical Network Frequency) extractor.

    Usage:
        config = ENFConfig.from_json_file("enf_config.json")
        extractor = ENFExtractor(config)
        times, enf_values = extractor.extract(signal, fs)
    """

    def __init__(self, config: Optional[ENFConfig] = None):
        self.config = config or ENFConfig()

    # ---------- Internal helpers ----------

    @staticmethod
    def _bandpass_filter(
        signal: np.ndarray,
        fs: float,
        f_low: float,
        f_high: float,
        order: int = 4,
    ) -> np.ndarray:
        """
        Apply a Butterworth band-pass filter.
        """
        nyq = 0.5 * fs
        low = f_low / nyq
        high = f_high / nyq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    @staticmethod
    def _quadratic_interpolation(
        spectrum: np.ndarray,
        freqs: np.ndarray,
        peak_index: int
    ) -> float:
        """
        Quadratic interpolation around a peak bin to estimate sub-bin peak frequency.
        """
        # if peak is at border, cannot interpolate
        if peak_index <= 0 or peak_index >= len(spectrum) - 1:
            return freqs[peak_index]

        # use log magnitude for more stable interpolation
        y1 = np.log(spectrum[peak_index - 1] + 1e-12)
        y2 = np.log(spectrum[peak_index] + 1e-12)
        y3 = np.log(spectrum[peak_index + 1] + 1e-12)

        denom = (y1 - 2 * y2 + y3)
        if abs(denom) < 1e-12:
            return freqs[peak_index]

        # offset from center bin in [-1, 1]
        p = 0.5 * (y1 - y3) / denom

        # frequency resolution
        df = freqs[1] - freqs[0]
        return freqs[peak_index] + p * df

    @staticmethod
    def _moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
        """
        Simple moving average with 'same' length output.
        """
        if len(x) < window_size:
            return x.copy()
        if window_size <= 1:
            return x.copy()
        window = np.ones(window_size) / window_size
        return np.convolve(x, window, mode="same")

    # ---------- Public API ----------

    def extract(
        self,
        signal: np.ndarray,
        fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ENF (Electrical Network Frequency) signature from an audio signal.

        Parameters
        ----------
        signal : np.ndarray
            1D audio signal (mono).
        fs : float
            Sampling rate in Hz.

        Returns
        -------
        times : np.ndarray
            Time stamps (seconds) for each ENF estimate.
        enf_values : np.ndarray
            Estimated fundamental ENF values (Hz) at the corresponding times.
        """
        cfg = self.config

        # Ensure 1D float array
        signal = np.asarray(signal, dtype=float).flatten()

        # --- 1) Band-pass filter around chosen harmonic ---
        harmonic_center = cfg.nominal_freq * cfg.harmonic
        f_low = harmonic_center - cfg.band_margin_hz
        f_high = harmonic_center + cfg.band_margin_hz
        filtered = self._bandpass_filter(
            signal, fs, f_low, f_high, order=cfg.bp_order
        )

        # --- 2) Framing ---
        win_len = int(round(cfg.window_sec * fs))
        hop_len = int(round(cfg.hop_sec * fs))
        if win_len <= 0 or hop_len <= 0:
            raise ValueError(
                "window_sec and hop_sec must result in positive window and hop lengths."
            )

        window = get_window("hann", win_len, fftbins=True)

        times: list[float] = []
        enf_values: list[float] = []

        start = 0
        while start + win_len <= len(filtered):
            frame = filtered[start:start + win_len] * window

            # --- 3) FFT and frequency search ---
            spectrum = np.abs(rfft(frame))
            freqs = rfftfreq(win_len, d=1.0 / fs)

            search_min = harmonic_center - cfg.search_width_hz
            search_max = harmonic_center + cfg.search_width_hz

            band_idx = np.where((freqs >= search_min) & (freqs <= search_max))[0]
            if band_idx.size < 3:  # need at least 3 bins for interpolation
                start += hop_len
                continue

            band_spectrum = spectrum[band_idx]
            local_peak_idx = np.argmax(band_spectrum)
            peak_idx = band_idx[local_peak_idx]

            # Sub-bin peak estimation
            peak_freq_harm = self._quadratic_interpolation(spectrum, freqs, peak_idx)

            # Map back to fundamental
            peak_freq_fund = peak_freq_harm / cfg.harmonic

            center_time = (start + win_len / 2) / fs
            times.append(center_time)
            enf_values.append(peak_freq_fund)

            start += hop_len

        if len(times) == 0:
            return np.array([]), np.array([])

        times_arr = np.array(times)
        enf_arr = np.array(enf_values)

        # --- 4) Outlier removal (optional) ---
        if cfg.outlier_threshold_hz > 0:
            mask = np.abs(enf_arr - cfg.nominal_freq) <= cfg.outlier_threshold_hz
            times_arr = times_arr[mask]
            enf_arr = enf_arr[mask]

        # --- 5) Smoothing (optional) ---
        if cfg.smooth_window > 1 and len(enf_arr) > 0:
            enf_arr = self._moving_average(enf_arr, cfg.smooth_window)

        return times_arr, enf_arr

    # Optional convenience if you like function-style usage:
    def __call__(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.extract(signal, fs)
