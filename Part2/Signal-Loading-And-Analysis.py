import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

def prepare_audio(filename, duration, start_time):
    # 1. Load audio file
    sample_rate, signal = wavfile.read(filename)
    
    # 2. Check if stereo -> convert to mono if needed
    if len(signal.shape) == 2:  # Stereo audio
        print("Signal is converted to mono")
        signal = signal.mean(axis=1)  # Convert to mono by averaging channels
    else:
        print("Signal was mono already")
    
    # 3. Extract segment of specified duration
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    segment = signal[start_sample:end_sample]
    
    # 4. Normalize signal to range [-1, 1]
    signal_max = np.max(np.abs(segment))
    if signal_max != 0:  # Avoid division by zero
        processed_signal = segment / signal_max
    else:
        processed_signal = segment
    print("Signal is Normalized")
    
    # 5. Create time array
    time_array = np.linspace(0, len(processed_signal) / sample_rate, len(processed_signal), endpoint=False)
    
    return processed_signal, time_array, sample_rate


def analyze_spectrum(signal, sample_rate):

    # 1. Compute the FFT
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)  # Magnitude of the FFT

    # 2. Generate the frequency array
    fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)  # Frequencies in Hz

    # 3. Extract positive frequencies
    positive_freq_indices = fft_freq >= 0
    frequencies = fft_freq[positive_freq_indices]
    positive_magnitude = fft_magnitude[positive_freq_indices]

    # 4. Convert magnitude to dB scale
    magnitude_spectrum = 20 * np.log10(positive_magnitude + 1e-10)  # Avoid log(0) by adding a small constant

    # Display the spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, magnitude_spectrum, color='blue')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()
    plt.show()

    return frequencies, magnitude_spectrum

filepath = "../skyfall_clip.wav"
duration = 2
start_time = 13
processed_signal, time_array, sample_rate = prepare_audio(filepath, duration, start_time)
frequencies, magnitude_spectrum = analyze_spectrum(processed_signal, sample_rate)