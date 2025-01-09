# Laboratory Exercise: Audio Signal Processing and Frequency Analysis

## Objectives
By the end of this lab, students should be able to:
1. Implement time and frequency domain audio processing techniques
2. Design and apply digital filters
3. Understand and implement convolution effects
4. Analyze frequency spectra of audio signals

## Pre-lab Preparation

### Required Libraries
- numpy (for numerical computations)
- scipy.io.wavfile (for audio file handling)
- matplotlib.pyplot (for visualization)
- IPython.display (for audio playback)

## Part 1: Signal Loading and Analysis

### Task 1.1: Audio File Handling
1. Load the WAV file using scipy.io.wavfile
2. Handle these cases:
- Stereo to mono conversion
- Data type normalization
- Signal duration extraction

Pseudo-code structure:
```python
def prepare_audio(filename, duration, start_time):
# 1. Load audio file
# 2. Check if stereo -> convert to mono if needed
# 3. Extract segment of specified duration
# 4. Normalize signal to range [-1, 1]
# 5. Create time array
return processed_signal, time_array, sample_rate
```

### Task 1.2: Spectrum Analysis
Implement a function to compute and display the frequency spectrum:

Pseudo-code structure:
```python
def analyze_spectrum(signal, sample_rate):
# 1. Compute FFT
# 2. Generate frequency array
# 3. Extract positive frequencies
# 4. Convert magnitude to dB scale
# Hint: dB = 20 * log10(|magnitude|)
return frequencies, magnitude_spectrum
```

## Part 2: Reverb Effect Implementation

### Task 2.1: Design Reverb Kernel
Create a function that generates a reverb impulse response:

Pseudo-code structure:
```python
def create_reverb(duration, sample_rate):
# 1. Create time array for kernel
# 2. Generate exponentially decaying sinc function
# 3. Normalize kernel
# Parameters to consider:
# - Decay rate
# - Sinc frequency
# - Duration
return reverb_kernel
```

### Task 2.2: Apply Reverb
Implement both time and frequency domain approaches:

Time Domain:
```python
def apply_time_reverb(signal, kernel):
# 1. Perform convolution
# 2. Handle signal boundaries

# 3. Normalize output
return processed_signal
```

Frequency Domain:
```python
def apply_freq_reverb(signal, kernel):
# 1. Zero-pad kernel to match signal length
# 2. Transform both signals to frequency domain
# 3. Multiply spectra
# 4. Transform back to time domain
# 5. Handle real/imaginary components
return processed_signal
```

## Part 3: Low-Pass Filter Implementation

### Task 3.1: Filter Design
Create a low-pass filter kernel:

Pseudo-code structure:
```python
def design_lpf(cutoff_freq, sample_rate, kernel_length):
# 1. Create time array centered at zero
# 2. Generate sinc function
# 3. Apply window function

# Key decisions:
# - Window type selection
# - Kernel length
# - Normalization method
return filter_kernel
```

### Task 3.2: Filtering Implementation
Implement filtering in both domains:

Time Domain:
```python
def time_domain_filter(signal, kernel):
# 1. Perform convolution
# 2. Handle edge effects
# 3. Normalize result
return filtered_signal
```

Frequency Domain:
```python
def frequency_domain_filter(signal, cutoff, sample_rate):
# 1. Create frequency mask
# 2. Transform signal to frequency domain
# 3. Apply mask
# 4. Transform back to time domain

return filtered_signal
```

## Analysis Tasks

### 1. Spectrum Visualization
For each stage of processing:
- Plot time domain signal
- Calculate and plot spectrum
- Mark key frequency components

### 2. Comparative Analysis
Compare:
- Time vs frequency domain implementations
- Different parameter settings
- Processing artifacts

### 3. Parameter Exploration
Investigate effects of:
1. Reverb parameters:
- Duration
- Decay rate
- Frequency content

2. Filter parameters:
- Cutoff frequency

- Kernel length
- Window type

## Report Requirements

### Documentation
1. Implementation approach:
- Design decisions
- Algorithm explanations
- Parameter choices

2. Results analysis:
- Spectrum comparisons
- Time domain effects
- Audible differences

3. Critical evaluation:
- Implementation challenges
- Performance analysis
- Suggested improvements

### Code Organization
Structure your code with these components:
1. Signal preparation functions
2. Processing algorithms
3. Analysis tools

4. Visualization utilities

## Tips for Success
1. Test with simple signals first
2. Verify each step separately
3. Plot intermediate results
4. Compare with theoretical expectations
5. Document unexpected observations

## Common Pitfalls to Avoid
1. Incorrect normalization
2. Missing signal padding
3. Improper frequency scaling
4. Complex number handling errors
5. Edge effect oversight