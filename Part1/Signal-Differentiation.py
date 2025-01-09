import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def analyze_differentiation(signal, time_array, sample_rate):
    # Time Domain Approach
    # 1. Calculate numerical gradient
    dt = time_array[1] - time_array[0]  # Time step
    time_domain_derivative = np.gradient(signal, dt)

    # 2. Compute the spectrum of the time-domain derivative
    frequencies = np.linspace(-5000, 5000, 1000)
    time_domain_spectrum = manual_ctft(time_domain_derivative, time_array, frequencies)
    print("Time domain diff done")

    # Frequency Domain Approach
    # 1. Transform the signal to the frequency domain
    frequency_domain_spectrum = manual_ctft(signal, time_array, frequencies)

    # 2. Multiply by j2πf (differentiation in the frequency domain)
    frequency_domain_derivative_spectrum = 1j * 2 * np.pi * frequencies * frequency_domain_spectrum
    print("Frequency domain  * 2pi*j part made")
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(time_domain_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Time Domain Differentiation Spectrum")
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(frequency_domain_derivative_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Freq Domain Spectrum 2*pi*j*f")
    plt.grid()
    plt.show()
    
    # Organizing results
    derivative_results = {
        "time_domain_derivative": time_domain_derivative,
        "time_domain_spectrum": time_domain_spectrum,
        "frequency_domain_derivative_spectrum": frequency_domain_derivative_spectrum
    }

    return derivative_results, frequencies


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
    print(duration)
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

ds,kk = analyze_differentiation(audio_data, time_array, sample_rate)