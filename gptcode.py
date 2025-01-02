import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import matplotlib.pyplot as plt

# Step 1: Load the WAV file
sample_rate, audio_data = wavfile.read("cat.wav")

# Step 2: Normalize the audio data (if needed)
if audio_data.dtype == np.int16:  # Typical for WAV files
    audio_data = audio_data / 32768.0  # Normalize 16-bit PCM to range [-1, 1]
elif audio_data.dtype == np.int32:
    audio_data = audio_data / 2147483648.0  # Normalize 32-bit PCM

# If stereo, take only one channel (e.g., left)
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0]

# Step 3: Perform FFT
fft_result = fft(audio_data)
frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)

# Step 4: Visualize the FFT
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(fft_result)//2]))
plt.title("Frequency Spectrum of the File")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
