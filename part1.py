import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

    # i am just saving the shortened audio file
    output_path = "segment.wav"
    wavfile.write(output_path, sample_rate, (segment * 32767).astype(np.int16))  # Convert back to int16
    print(f"Segment saved to {output_path}")

    #  4. Normalize signal to range [-1, 1] done

    # 5. Create time array
    # Extracting time array
    sample_rate, audio_data = wavfile.read("segment.wav")
    duration = len(audio_data) / sample_rate
    print(duration)
    time_array = np.linspace(0, duration, num=len(audio_data))
    return time_array, sample_rate, audio_data


def manual_ctft(signal, time_array, frequencies):
    """
    Implement Continuous-Time Fourier Transform (CTFT) from first principles.

    Parameters:
    - signal: numpy array, the input time-domain signal
    - time_array: numpy array, the time values corresponding to the signal
    - frequencies: numpy array, the frequency points at which to compute the CTFT

    Returns:
    - frequency_spectrum: numpy array, the CTFT values at the specified frequencies
    """
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
    """
    Compare time and frequency domain approaches for time reversal.

    Parameters:
    - signal: numpy array, the original signal in the time domain
    - time_array: numpy array, the time values corresponding to the signal
    - sample_rate: int, the sample rate of the signal in Hz

    Returns:
    - time_method_results: tuple (reversed_signal, reversed_spectrum)
    - freq_method_results: tuple (original_spectrum, conjugate_spectrum)
    """
    # Step 1: Time Domain Method
    # 1. Reverse the signal directly
    print("started")
    reversed_signal = signal[::-1]

    print("reversed signal")
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
     

    # Plotting just cause i want to
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(reversed_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Reversed Spectrum CTFT")
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(original_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Original Spectrum CTFT")
    plt.grid()
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(conjugate_spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Conjugate Spectrum CTFT")
    plt.grid()
    plt.show()
    return time_method_results, freq_method_results

def analyze_differentiation(signal, time_array, sample_rate):
    """
    Implement and compare differentiation methods: time domain vs. frequency domain.

    Parameters:
    - signal: numpy array, the input signal in the time domain
    - time_array: numpy array, the time values corresponding to the signal
    - sample_rate: int, the sample rate of the signal in Hz

    Returns:
    - derivative_results: dict containing time-domain and frequency-domain derivatives
    """
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
    plt.title("Freq Domain Spectrum 2pijf")
    plt.grid()
    plt.show()
    

    # Organize results for return
    derivative_results = {
        "time_domain_derivative": time_domain_derivative,
        "time_domain_spectrum": time_domain_spectrum,
        "frequency_domain_derivative_spectrum": frequency_domain_derivative_spectrum
    }

    return derivative_results, frequencies


def analyze_modulation_using_theory(signal, time_array, sample_rate, carrier_freq):
    """
    Analyze amplitude modulation (AM) using the theoretical relationship.

    Parameters:
    - signal: numpy array, the input signal in the time domain
    - time_array: numpy array, the time values corresponding to the signal
    - sample_rate: int, the sample rate of the signal in Hz
    - carrier_freq: float, the carrier frequency in Hz

    Returns:
    - modulated_signal: numpy array, the modulated signal
    - spectrum: numpy array, the spectrum of the modulated signal
    """
    # Step 1: Generate carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_freq * time_array)

    # Step 2: Perform modulation
    modulated_signal = signal * carrier_wave

    # Step 3: Analyze spectrum of modulated signal
    frequencies = np.linspace(-5000, 5000, 1000)
    spectrum =  manual_ctft(signal, time_array, frequencies)

    # Step 4: Analyze theoretical result
    baseband_spectrum = np.fft.fft(signal)
    theoretical_spectrum = 0.5 * (np.roll(baseband_spectrum, int(carrier_freq * len(time_array) / sample_rate)) + np.roll(baseband_spectrum, -int(carrier_freq * len(time_array) / sample_rate)))
    
    plot_modulation_results(time_array, signal, modulated_signal, frequencies, spectrum, carrier_freq, (0.001, 0.0015))

    return modulated_signal, spectrum, theoretical_spectrum, frequencies


def plot_modulation_results(time_array, signal, modulated_signal, frequencies, spectrum, carrier_freq, time_window=None):
    """
    Plot the modulation results, including zoomed time-domain views and spectrum.

    Parameters:
    - time_window: tuple (start, end) in seconds for zoomed view (optional)
    """
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


def analyze_phase_patterns(signal, time_array, sample_rate, carrier_freq, phase_mod_index=1.0):
    """
    Investigate phase modulation effects.

    Parameters:
    - signal: numpy array, the input signal in the time domain
    - time_array: numpy array, the time values corresponding to the signal
    - sample_rate: int, the sample rate of the signal in Hz
    - carrier_freq: float, the carrier frequency in Hz
    - phase_mod_index: float, phase modulation index (default: 1.0)

    Returns:
    - modulated_signal: numpy array, the phase-modulated signal
    - spectrum: numpy array, the spectrum of the phase-modulated signal
    """
    # Step 1: Set up carrier wave
    carrier_wave = np.cos(2 * np.pi * carrier_freq * time_array)

    # Step 2: Calculate phase terms (using signal as the phase modulating term)
    phase_terms = phase_mod_index * signal

    # Step 3: Apply phase modulation
    modulated_signal = np.cos(2 * np.pi * carrier_freq * time_array + phase_terms)

    # Step 4: Analyze spectrum of the modulated signal
    frequencies = np.linspace(-5000, 5000, 1000)
    spectrum =  manual_ctft(signal, time_array, frequencies)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(spectrum))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Phase Modulated Spectrum")
    plt.grid()
    plt.show()

    return modulated_signal, spectrum, frequencies



# main
time_array, sample_rate, audio_data = prepare_audio("skyfall_clip.wav", 2, 13)
frequencies = np.linspace(-5000, 5000, 1000)
fs =  manual_ctft(audio_data, time_array, frequencies)
print(fs)  # Complex array printed
# Plot it
'''
plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fs))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Continuous-Time Fourier Transform (CTFT)")
plt.grid()
plt.show()
'''

tmr, ss = analyze_time_reversal(audio_data, time_array, sample_rate)

ds,kk = analyze_differentiation(audio_data, time_array, sample_rate)

analyze_modulation_using_theory(audio_data, time_array, sample_rate, 500000)

analyze_phase_patterns(audio_data, time_array, sample_rate, 100000000, phase_mod_index=1.0)