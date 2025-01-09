import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from scipy.io import wavfile

def prepare_audio(filename, duration, start_time):
    sample_rate, signal = wavfile.read(filename)
    if len(signal.shape) == 2:
        print("Signal is converted to mono")
        signal = signal.mean(axis=1)
    else:
        print("Signal was mono already")
    
    start_sample = int(start_time * sample_rate)
    end_sample = int((start_time + duration) * sample_rate)
    segment = signal[start_sample:end_sample]
    
    signal_max = np.max(np.abs(segment))
    if signal_max != 0:
        processed_signal = segment / signal_max
    else:
        processed_signal = segment
    print("Signal is Normalized")
    
    time_array = np.linspace(0, len(processed_signal) / sample_rate, len(processed_signal), endpoint=False)
    return processed_signal, time_array, sample_rate


def create_reverb(duration, sample_rate, decay_rate=0.5, sinc_freq=500):
    # 1. Create a time array for the kernel
    num_samples = int(duration * sample_rate)
    time_array = np.linspace(0, duration, num_samples, endpoint=False)
    
    # 2. Generate exponentially decaying sinc function
    sinc_component = np.sinc(2 * sinc_freq * (time_array - duration / 2))
    decay_envelope = np.exp(-decay_rate * time_array)
    reverb_kernel = sinc_component * decay_envelope
    
    # 3. Normalize the kernel
    reverb_kernel = reverb_kernel / np.max(np.abs(reverb_kernel))  # Normalize to range [-1, 1]

    if np.allclose(reverb_kernel, 0):
        print("The reverb kernel is silent. Check your parameters.")
        return None
    
    # Plot the reverb kernel
    plt.figure(figsize=(10, 4))
    plt.plot(time_array, reverb_kernel, color='purple')
    plt.title("Reverb Impulse Response Kernel")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


    # 4. Save the kernel as a WAV file
    reverb_kernel_int16 = (reverb_kernel * 32767).astype(np.int16)  # Convert to 16-bit PCM format
    wavfile.write("output-reverb.wav", sample_rate, reverb_kernel_int16)

    # SInce I am working inside a virtual env, the playback feature is not working
    # I am saving it as wav file to be played manually
    '''
    print("Playing the generated reverb kernel:")
    display(Audio(data=reverb_kernel, rate=sample_rate))
    '''

    return reverb_kernel

def apply_time_reverb(signal, kernel):
    # Perform convolution (output size will be len(signal) + len(kernel) - 1)
    processed_signal = np.convolve(signal, kernel, mode='full')
    
    # Handle signal boundaries by trimming or zero-padding as needed
    # Here, we trim the result to the same length as the original signal
    processed_signal = processed_signal[:len(signal)]
    
    # Normalize the output to prevent clipping
    max_val = np.max(np.abs(processed_signal))
    if max_val > 1:
        processed_signal /= max_val
    
    return processed_signal


def apply_freq_reverb(signal, kernel):
    # Zero-pad kernel to match signal length
    padded_kernel = np.pad(kernel, (0, len(signal) - len(kernel)), mode='constant')
    
    # Transform both signal and kernel to the frequency domain (FFT)
    signal_freq = np.fft.fft(signal)
    kernel_freq = np.fft.fft(padded_kernel)
    
    # Multiply the spectra
    processed_freq = signal_freq * kernel_freq
    
    # Transform back to the time domain using inverse FFT
    processed_signal = np.fft.ifft(processed_freq).real  # Take the real part of the result
    
    # Normalize the output to prevent clipping
    max_val = np.max(np.abs(processed_signal))
    if max_val > 1:
        processed_signal /= max_val
    
    return processed_signal


duration = 2.0
decay_rate = 0.8
sinc_freq = 500

# Create the reverb kernel
kernel = create_reverb(duration, sample_rate, decay_rate, sinc_freq)
filepath = "../skyfall_clip.wav"
duration = 2
start_time = 13
signal, time_array, sample_rate = prepare_audio(filepath, duration, start_time)
appl_freq_reverb = apply_freq_reverb(signal, kernel)
appl_time_reverb = apply_time_reverb(signal, kernel)


# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(signal, color='blue', label='Original')
plt.legend()

plt.subplot(3, 1, 2)
plt.title("Time Domain Reverb")
plt.plot(appl_time_reverb, color='green', label='Time Reverb')
plt.legend()

plt.subplot(3, 1, 3)
plt.title("Frequency Domain Reverb")
plt.plot(appl_freq_reverb, color='red', label='Frequency Reverb')
plt.legend()

plt.tight_layout()
plt.show()
