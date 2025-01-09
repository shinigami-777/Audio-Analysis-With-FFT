import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def analyze_energy_distribution(signal, time_array, sample_rate, carrier_freq):
    # Step 1: Generate carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_freq * time_array)

    # Step 2: Perform modulation
    modulated_signal = signal * carrier_wave

    # Saving the audio. Very high pitched :)
    wavfile.write("modulated.wav", sample_rate, (modulated_signal * 32767).astype(np.int16))
    print(f"Segment saved to modulated.wav")
    
    # Step 3: Analyze spectrum of modulated signal
    frequencies = np.linspace(-5000, 5000, 1000)
    spectrum =  manual_ctft(modulated_signal, time_array, frequencies)

    # Step 4: Analyze results
    plot_modulation_results(time_array, signal, modulated_signal, frequencies, spectrum, carrier_freq, (0.001, 0.0015))

    return modulated_signal, spectrum

def plot_modulation_results(time_array, signal, modulated_signal, frequencies, spectrum, carrier_freq, time_window=None):
    #time_window: tuple (start, end) in seconds for zoomed view (optional)

    plt.figure(figsize=(14, 10))

    # Original and Modulated Signals in Time Domain
    plt.subplot(3, 1, 1)
    plt.plot(time_array, signal, label="Original Signal")
    plt.plot(time_array, modulated_signal, label="Modulated Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time Domain: Original vs Modulated Signal")
    plt.legend()
    plt.grid()

    # Zoomed Time Domain View
    if time_window:
        start_idx = int(time_window[0] * len(time_array) / time_array[-1])
        end_idx = int(time_window[1] * len(time_array) / time_array[-1])
        plt.subplot(3, 1, 2)
        plt.plot(time_array[start_idx:end_idx], signal[start_idx:end_idx], label="Original Signal")
        plt.plot(time_array[start_idx:end_idx], modulated_signal[start_idx:end_idx], label="Modulated Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Zoomed View: {time_window[0]}s to {time_window[1]}s")
        plt.legend()
        plt.grid()

    # Spectrum Analysis
    plt.subplot(3, 1, 3)
    plt.plot(frequencies, np.abs(spectrum), label="Spectrum")
    plt.axvline(x=carrier_freq, color='r', linestyle='--', label=f"Carrier ({carrier_freq} Hz)")
    plt.axvline(x=-carrier_freq, color='r', linestyle='--')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain: Modulated Signal Spectrum")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


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
    
    output_path = "segment.wav"
    wavfile.write(output_path, sample_rate, (processed_signal * 32767).astype(np.int16))
    print(f"Segment saved to {output_path}")

    return sample_rate, processed_signal, time_array

def manual_ctft(signal, time_array, frequencies):
    # Step 1: Set up integration parameters
    dt = time_array[1] - time_array[0]  # Time step (assumes uniform spacing)

    # Step 2: Initialize the frequency spectrum
    frequency_spectrum = np.zeros(len(frequencies), dtype=complex)

    # Step 3: Compute the CTFT for each frequency
    for i, f in enumerate(frequencies):
        # Generate the complex exponential for this frequency
        exponential = np.exp(-2j * np.pi * f * time_array)

        # Multiply the signal with the complex exponential
        product = signal * exponential

        # Numerically integrate using the trapezoidal rule
        frequency_spectrum[i] = np.trapezoid(product, time_array)

    return frequency_spectrum


# main
sample_rate, audio_data, time_array = prepare_audio("../skyfall_clip.wav", 2, 13)
frequencies = np.linspace(-5000, 5000, 1000)

modulated_signal, spectrum = analyze_energy_distribution(audio_data, time_array, sample_rate, 500000)   # 500 KHz given as high freq
modulated_signal, spectrum = analyze_energy_distribution(audio_data, time_array, sample_rate, 500000000)   # 500 MHz given as high freq