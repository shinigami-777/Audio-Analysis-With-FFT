import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

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


def design_lpf(cutoff_freq, sample_rate, kernel_length, window_type='hamming'):
    if kernel_length % 2 == 0:
        raise ValueError("Kernel length must be odd to center the filter at zero.")
    
    # 1. Create time array centered at zero
    half_length = (kernel_length - 1) // 2
    t = np.arange(-half_length, half_length + 1)

    # 2. Generate sinc function (ideal low-pass filter)
    normalized_cutoff = cutoff_freq / (sample_rate / 2)  # Normalize to Nyquist frequency
    sinc_func = np.sinc(2 * normalized_cutoff * t)

    # 3. Apply window function
    if window_type == 'hamming':
        window = np.hamming(kernel_length)
    elif window_type == 'hanning':
        window = np.hanning(kernel_length)
    elif window_type == 'blackman':
        window = np.blackman(kernel_length)
    else:
        raise ValueError("Unsupported window type. Choose 'hamming', 'hanning', or 'blackman'.")
    
    # Apply the window to the sinc function
    filter_kernel = sinc_func * window

    # Normalize the kernel to ensure the sum of the filter coefficients equals 1
    filter_kernel /= np.sum(filter_kernel)

    return filter_kernel


def time_domain_filter(signal, kernel):
    # 1. Perform convolution
    filtered_signal = np.convolve(signal, kernel, mode='same')  # Use 'same' to match the output size to the input size
    
    # 2. Normalize the result to prevent clipping
    max_val = np.max(np.abs(filtered_signal))
    if max_val > 1:
        filtered_signal /= max_val
    
    return filtered_signal

def frequency_domain_filter(signal, cutoff, sample_rate):
    # 1. Create frequency mask
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)  # Frequency bins
    mask = np.abs(freqs) <= cutoff  # Low-pass filter mask
    
    # 2. Transform signal to frequency domain
    signal_freq = np.fft.fft(signal)
    
    # 3. Apply the mask
    filtered_freq = signal_freq * mask
    
    # 4. Transform back to time domain
    filtered_signal = np.fft.ifft(filtered_freq).real  # Take only the real part
    
    return filtered_signal

# main
cutoff_freq = 500  # Kept this as 500 Hz as low pass filter
kernel_length = 101 # Must be odd
filepath = "../skyfall_clip.wav"
signal, time_array, sample_rate = prepare_audio(filepath, 2, 13)
lpf_kernel = design_lpf(cutoff_freq, sample_rate, kernel_length, window_type='hamming')

time_filtered_signal = time_domain_filter(signal, lpf_kernel)

freq_filtered_signal = frequency_domain_filter(signal, cutoff_freq, sample_rate)

# Plot the original and filtered signals
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original Signal")
plt.plot(time_array, signal, label="Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.title("Time-Domain Filtered Signal")
plt.plot(time_array, time_filtered_signal, label="Time-Domain Filtered", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.title("Frequency-Domain Filtered Signal")
plt.plot(time_array, freq_filtered_signal, label="Frequency-Domain Filtered", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
