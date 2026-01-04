from enf.enf_extractor import *

import soundfile as sf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    Load a WAV file and extract ENF.
    """

    # Load audio (mono or stereo)
    filename = "/Users/cristea/Downloads/deepfake/real/common_voice_ro_20348817.wav"
    audio, fs = sf.read(filename)
    audio = np.concat((audio, audio, audio, audio))

    # If stereo, convert to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    config = ENFConfig.from_json_file("enf_config.json")
    extractor = ENFExtractor(config)
    times, enf = extractor(audio, fs)

    print(f"Extracted {len(enf)} ENF points")

    # Plot ENF over time
    plt.figure()
    plt.plot(times, enf, marker=".", linestyle="-")
    plt.xlabel("Time (s)")
    plt.ylabel("ENF estimate (Hz)")
    plt.title("ENF Signature")
    plt.grid(True)
    plt.show()
