import sounddevice as sd
import numpy as np
from scipy.signal import sosfilt, sosfreqz
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import sys

INPUT_DEVICE = 0   # BlackHole 2ch
OUTPUT_DEVICE = 2  # MacBook Pro Speakers

SAMPLERATE = 48000
CHANNELS = 2
BLOCKSIZE = 4096  # Increased for better FFT resolution

# Queue for passing audio data to the plotter
plot_queue = queue.Queue()


class ParametricEQBand:
    """A single parametric EQ band using a biquad peaking filter."""
    
    def __init__(self, frequency: float, gain_db: float, q: float, samplerate: float):
        """
        Initialize a parametric EQ band.
        
        Args:
            frequency: Center frequency in Hz
            gain_db: Gain in dB (positive = boost, negative = cut)
            q: Q factor (bandwidth control, higher = narrower)
            samplerate: Sample rate in Hz
        """
        self.samplerate = samplerate
        self.frequency = frequency
        self.gain_db = gain_db
        self.q = q
        
        # Calculate filter coefficients
        self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """Calculate biquad peaking EQ filter coefficients in SOS format."""
        A = 10 ** (self.gain_db / 40)  # amplitude
        omega = 2 * np.pi * self.frequency / self.samplerate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2 * self.q)
        
        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
        
        # SOS format: [b0, b1, b2, a0, a1, a2] normalized by a0
        self.sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
    
    def __repr__(self):
        return f"EQBand(freq={self.frequency:>7.1f} Hz, gain={self.gain_db:>+6.1f} dB, Q={self.q:.2f})"


class ParametricEQ:
    """Chain of parametric EQ bands using combined SOS filter for efficiency."""
    
    def __init__(self, samplerate: float, n_channels: int = 2, bands: list[dict] = None):
        """
        Initialize the parametric EQ.
        
        Args:
            samplerate: Sample rate in Hz
            n_channels: Number of audio channels
            bands: List of band configs, each a dict with 'frequency', 'gain_db', 'q'
        """
        self.samplerate = samplerate
        self.n_channels = n_channels
        self.bands: list[ParametricEQBand] = []
        self._sos = None
        self._zi = None
        
        # Initialize bands from config list
        if bands:
            for band_config in bands:
                band = ParametricEQBand(
                    band_config['frequency'],
                    band_config['gain_db'],
                    band_config['q'],
                    self.samplerate
                )
                self.bands.append(band)
        
        # Build the combined SOS
        self._rebuild_sos()
    
    def _rebuild_sos(self):
        """Combine all band SOS into a single stacked SOS array."""
        if not self.bands:
            self._sos = None
            self._zi = None
            return
        
        # Stack all SOS sections: shape (n_bands, 6)
        self._sos = np.vstack([band.sos for band in self.bands])
        # Initialize filter state: shape (n_bands, n_channels, 2)
        self._zi = np.zeros((len(self.bands), self.n_channels, 2), dtype=np.float64)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process audio through all EQ bands in a single optimized call.
        
        Args:
            data: Input audio data (frames x channels)
            
        Returns:
            Processed audio data
        """
        if self._sos is None:
            return data
        
        output, self._zi = sosfilt(self._sos, data, axis=0, zi=self._zi)
        return output
    
    def get_frequency_response(self, n_points: int = 1024):
        """
        Calculate the combined frequency response of all EQ bands.
        
        Args:
            n_points: Number of frequency points
            
        Returns:
            frequencies: Array of frequencies in Hz
            magnitude_db: Array of magnitude response in dB
        """
        if self._sos is None:
            freqs = np.linspace(20, self.samplerate / 2, n_points)
            return freqs, np.zeros(n_points)
        
        w, h = sosfreqz(self._sos, worN=n_points, fs=self.samplerate)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        return w, magnitude_db
    
    def __repr__(self):
        lines = [f"ParametricEQ ({len(self.bands)} bands):"]
        for i, band in enumerate(self.bands):
            lines.append(f"  [{i}] {band}")
        return "\n".join(lines)



# ============================================================
# EQ CONFIGURATION - Edit this list to adjust bands
# ============================================================
EQ_BANDS = [
    {'frequency': 100,  'gain_db': 12.0,  'q': 1.0},   # Bass boost
    {'frequency': 1000, 'gain_db': -12.0, 'q': 2.0},   # Mid cut
    {'frequency': 5000, 'gain_db': 12.0,  'q': 1.5},   # Treble boost
]

# Create the parametric EQ
eq = ParametricEQ(SAMPLERATE, CHANNELS, bands=EQ_BANDS)

print(eq)
print("\nStarting real-time spectrum analyzer...")

# Prepare the plot with two subplots
fig, (ax_eq, ax_spec) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [1, 2]})
x_freqs = np.fft.rfftfreq(BLOCKSIZE, 1/SAMPLERATE)

# --- Top Plot: EQ Curve ---
eq_freqs, eq_mag = eq.get_frequency_response(n_points=2048)
ax_eq.semilogx(eq_freqs, eq_mag, 'b-', linewidth=2)
ax_eq.set_ylabel('EQ Gain (dB)')
ax_eq.set_title('Parametric EQ Response')
ax_eq.set_ylim(-24, 24)
ax_eq.grid(True, which='both', linestyle='-', alpha=0.2)
ax_eq.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# --- Bottom Plot: Real-time Spectrum ---
line_spectrum, = ax_spec.semilogx(x_freqs, np.full(len(x_freqs), -100), 'g-', alpha=0.4, label='Live')
line_average, = ax_spec.semilogx(x_freqs, np.full(len(x_freqs), -100), 'y-', linewidth=2, label='Long-term Avg')
ax_spec.set_xlim(20, SAMPLERATE / 2)
ax_spec.set_ylim(-100, -40)  # Standard spectrum range
ax_spec.set_xlabel('Frequency (Hz)')
ax_spec.set_ylabel('Output Level (dB)')
ax_spec.set_title('Live Output Spectrum')
ax_spec.grid(True, which='both', linestyle='-', alpha=0.2)
ax_spec.legend(loc='upper right')

plt.tight_layout()

# FFT window and smoothing state
window = np.hanning(BLOCKSIZE)
smooth_mag = np.full(len(x_freqs), -100.0)
avg_mag = np.full(len(x_freqs), -100.0)
alpha_smooth = 0.6  # Smoothing factor for live line
alpha_avg = 0.15    # Smoothing factor for long-term average (lower = slower)

def update_plot(frame):
    """Update the spectrum lines from the queue data."""
    global line_spectrum, line_average, smooth_mag, avg_mag
    data = None
    
    # Get the latest data from the queue
    while not plot_queue.empty():
        data = plot_queue.get()
    
    if data is not None:
        # Compute FFT of the first channel
        fft_data = np.fft.rfft(data[:, 0] * window)
        
        # Magnitude calculation with window compensation
        mag = np.abs(fft_data) * 2.0 / np.sum(window)
        
        # Apply +3dB/octave compensation
        freq_compensation = np.sqrt(np.arange(len(mag)))
        mag *= freq_compensation
        
        mag_db = 20 * np.log10(mag + 1e-10)
        
        # Apply exponential smoothing to live line
        smooth_mag = (alpha_smooth * mag_db) + (1 - alpha_smooth) * smooth_mag
        line_spectrum.set_ydata(smooth_mag)
        
        # Apply very slow smoothing to average line
        avg_mag = (alpha_avg * mag_db) + (1 - alpha_avg) * avg_mag
        line_average.set_ydata(avg_mag)
    
    return line_spectrum, line_average

def callback(indata, outdata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    
    # Apply parametric EQ
    processed = eq.process(indata)
    outdata[:] = processed
    
    # Put processed data into queue for plotting
    try:
        plot_queue.put_nowait(processed.copy())
    except queue.Full:
        pass

# Start the audio stream
stream = sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SAMPLERATE,
    blocksize=BLOCKSIZE,
    channels=CHANNELS,
    dtype="float32",
    callback=callback,
)

with stream:
    # Start the animation
    ani = FuncAnimation(fig, update_plot, interval=5, blit=True, cache_frame_data=False)
    plt.show()

