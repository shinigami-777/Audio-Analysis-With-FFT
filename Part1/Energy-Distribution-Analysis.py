import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def analyze_energy_distribution(signal, time_array, sample_rate, carrier_freq):
    # Step 1: Generate carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_freq * time_array)

    # Step 2: Perform modulation
    modulated_signal = signal * carrier_wave

    # Step 3: Analyze spectrum of modulated signal
    frequencies = np.linspace(-5000, 5000, 1000)
    spectrum =  manual_ctft(signal, time_array, frequencies)

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
    sample_rate, audio_data = wavfile.read(filename)
    # 2. Check if stereo -> convert to mono if needed
    if len(audio_data.shape) == 2:  # Check if the audio is stereo
        print("Converting stereo to mono...")
        audio_data = np.mean(audio_data, axis=1, dtype=audio_data.dtype)
    
     # Normalize the audio data to the range [-1, 1] based on the data type
    if audio_data.dtype == np.int16:
        audio_data = audio_data / 32768.0  # 16-bit signed integer
    elif audio_data.dtype == np.int32:
        audio_data = audio_data / 2147483648.0  # 32-bit signed integer
    elif audio_data.dtype == np.uint8:
        audio_data = (audio_data - 128) / 128.0  # 8-bit unsigned integer
    else:
        print(f"Unsupported audio data type: {audio_data.dtype}")
    
    # 3. Extract segment of specified duration
    end_time = start_time+duration
    start_sample = int(start_time*sample_rate)
    end_sample = int(end_time*sample_rate)
    if start_sample < 0 or end_sample > len(audio_data):
        raise ValueError("Specified segment is out of bounds.")
    segment = audio_data[start_sample:end_sample]

    # i am just saving the shortened audio file. Play this.
    # Since i am using a virtual env and playback is not configured in it ....
    output_path = "segment.wav"
    wavfile.write(output_path, sample_rate, (segment * 32767).astype(np.int16))  # Convert back to int16
    print(f"Segment saved to {output_path}")

    # 5. Create time array
    # Extracting time array
    sample_rate, audio_data = wavfile.read("segment.wav")
    duration = len(audio_data) / sample_rate
    #print(duration)
    time_array = np.linspace(0, duration, num=len(audio_data))
    return time_array, sample_rate, audio_data

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
time_array, sample_rate, audio_data = prepare_audio("../skyfall_clip.wav", 2, 13)
frequencies = np.linspace(-5000, 5000, 1000)

modulated_signal, spectrum = analyze_energy_distribution(audio_data, time_array, sample_rate, 500000)   # 500 GHz given as high freq