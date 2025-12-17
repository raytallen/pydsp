import sounddevice as sd
import numpy as np
from scipy.signal import sosfilt, sosfreqz
import matplotlib.pyplot as plt

INPUT_DEVICE = 0   # BlackHole 2ch
OUTPUT_DEVICE = 2  # MacBook Pro Speakers

SAMPLERATE = 48000
CHANNELS = 2
BLOCKSIZE = 512


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
    
    def plot_response(self, save_path: str = None):
        """Save the EQ frequency response plot to a file."""
        freqs, magnitude_db = self.get_frequency_response(2048)
        
        plt.figure(figsize=(10, 5))
        plt.semilogx(freqs, magnitude_db, 'b-', linewidth=2)
        plt.xlim(20, self.samplerate / 2)
        plt.ylim(-24, 24)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.title('Parametric EQ Frequency Response')
        plt.grid(True, which='both', linestyle='-', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Mark the center frequencies of each band
        for band in self.bands:
            plt.axvline(x=band.frequency, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
            return save_path
        else:
            # Save to temp file
            import tempfile
            import os
            temp_path = os.path.join(tempfile.gettempdir(), 'eq_response.png')
            plt.savefig(temp_path, dpi=150)
            plt.close()
            return temp_path
    
    def __repr__(self):
        lines = [f"ParametricEQ ({len(self.bands)} bands):"]
        for i, band in enumerate(self.bands):
            lines.append(f"  [{i}] {band}")
        return "\n".join(lines)


# ============================================================
# EQ CONFIGURATION - Edit this list to adjust bands
# ============================================================
EQ_BANDS = [
    {'frequency': 200,  'gain_db': 12.0,  'q': 1.0},   # Bass boost
    {'frequency': 3000, 'gain_db': -6.0, 'q': 0.7},   # Reduce harshness
]

# Create the parametric EQ
eq = ParametricEQ(SAMPLERATE, CHANNELS, bands=EQ_BANDS)

# Print EQ configuration
print(eq)
print()

# Show the frequency response plot (opens in default image viewer)
import subprocess
plot_path = eq.plot_response()
subprocess.run(['open', plot_path])
print(f"EQ response plot saved to: {plot_path}")
print()


def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    # Apply parametric EQ
    outdata[:] = eq.process(indata)

with sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SAMPLERATE,
    blocksize=BLOCKSIZE,
    channels=CHANNELS,
    dtype="float32",
    callback=callback,
):
    print("Routing BlackHole → MacBook Speakers (Ctrl+C to stop)")
    while True:
        sd.sleep(1000)
