import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


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


def analyze_time_reversal(signal, time_array, sample_rate):

    # Step 1: Time Domain Method
    # 1. Reverse the signal directly
    reversed_signal = signal[::-1]
    wavfile.write("reversed-audio.wav", sample_rate, (reversed_signal * 32767).astype(np.int16))
    # Saved . Now play this

    # 2. Calculate the spectrum of the reversed signal
    frequencies = np.linspace(-5000, 5000, 1000)
    reversed_spectrum = manual_ctft(reversed_signal, time_array, frequencies)

    # Step 2: Frequency Domain Method
    # 1. Calculate the spectrum of the original signal
    original_spectrum = manual_ctft(signal, time_array, frequencies)

    # 2. Take the complex conjugate of the original spectrum
    conjugate_spectrum = np.conj(original_spectrum)

    # Return the results for comparison

    time_method_results = (reversed_signal, reversed_spectrum)
    freq_method_results = (original_spectrum, conjugate_spectrum)
     

    # Plotting it to verify
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(reversed_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Reversed Spectrum CTFT (Abs)")
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(original_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Original Spectrum CTFT (Abs)")
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(conjugate_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Conjugate Spectrum CTFT (Abs)")
    plt.grid()
    plt.show()
    return time_method_results, freq_method_results

sample_rate, audio_data, time_array = prepare_audio("../skyfall_clip.wav", 2, 13)
tmr, ss = analyze_time_reversal(audio_data, time_array, sample_rate)