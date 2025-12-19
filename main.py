#!/usr/bin/env python3.13

import sounddevice as sd
import numpy as np
from scipy.signal import sosfilt, sosfreqz, butter
from dataclasses import dataclass
import queue
import sys
import rumps

INPUT_DEVICE = 0   # BlackHole 2ch
OUTPUT_DEVICE = 2  # MacBook Pro Speakers

SAMPLERATE = 48000
CHANNELS = 2
BLOCKSIZE = 32
LATENCY = 0

ENABLE_VISUALIZATION = False  # Set to False to disable the plot and save CPU

@dataclass
class LowPassFilter:
    """Configuration for a low-pass filter."""
    frequency: float
    q: float = 0.707
    steepness: int = 12

@dataclass
class HighPassFilter:
    """Configuration for a high-pass filter."""
    frequency: float
    q: float = 0.707
    steepness: int = 12

@dataclass
class LowShelf:
    """Configuration for a low-shelf filter."""
    frequency: float
    gain_db: float
    q: float = 0.707

@dataclass
class HighShelf:
    """Configuration for a high-shelf filter."""
    frequency: float
    gain_db: float
    q: float = 0.707

@dataclass
class ParametricBand:
    """Configuration for a peaking (parametric) EQ band."""
    frequency: float
    gain_db: float
    q: float = 1.0

EQ_BANDS = [
    HighPassFilter(frequency=50, q=0.707, steepness=24),
    LowShelf(frequency=150, q=0.707, gain_db=6.0),
    ParametricBand(frequency=2000, q=1.0, gain_db=-3.0),
]

class ParametricEQBand:
    """A single parametric EQ band using biquad filters."""
    
    def __init__(self, frequency: float, gain_db: float, q: float, samplerate: float, 
                 config_type: type = ParametricBand, steepness: int = 12):
        """
        Initialize a parametric EQ band.
        
        Args:
            frequency: Center frequency in Hz
            gain_db: Gain in dB (used for peaking and shelves)
            q: Q factor (bandwidth control)
            samplerate: Sample rate in Hz
            config_type: The configuration class type (LowPassConfig, etc.)
            steepness: Filter steepness in dB/octave (12, 24, 36, 48). Only for LP/HP.
        """
        self.samplerate = samplerate
        self.frequency = frequency
        self.gain_db = gain_db
        self.q = q
        self.config_type = config_type
        self.steepness = steepness
        
        # Calculate filter coefficients
        self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """Calculate biquad filter coefficients in SOS format."""
        # For LP/HP with higher steepness, use scipy.signal.butter
        if self.config_type in [LowPassFilter, HighPassFilter] and self.steepness > 12:
            order = int(self.steepness / 6)
            btype = 'lowpass' if self.config_type is LowPassFilter else 'highpass'
            self.sos = butter(order, self.frequency, btype=btype, fs=self.samplerate, output='sos')
            return

        A = 10 ** (self.gain_db / 40)
        omega = 2 * np.pi * self.frequency / self.samplerate
        sn = np.sin(omega)
        cs = np.cos(omega)
        alpha = sn / (2 * self.q)
        
        if self.config_type is ParametricBand:
            b0 = 1 + alpha * A
            b1 = -2 * cs
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cs
            a2 = 1 - alpha / A
        elif self.config_type is LowShelf:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) - (A - 1) * cs + 2 * sqrt_A * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cs)
            b2 = A * ((A + 1) - (A - 1) * cs - 2 * sqrt_A * alpha)
            a0 = (A + 1) + (A - 1) * cs + 2 * sqrt_A * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cs)
            a2 = (A + 1) + (A - 1) * cs - 2 * sqrt_A * alpha
        elif self.config_type is HighShelf:
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1) + (A - 1) * cs + 2 * sqrt_A * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cs)
            b2 = A * ((A + 1) + (A - 1) * cs - 2 * sqrt_A * alpha)
            a0 = (A + 1) - (A - 1) * cs + 2 * sqrt_A * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cs)
            a2 = (A + 1) - (A - 1) * cs - 2 * sqrt_A * alpha
        elif self.config_type is LowPassFilter:
            b0 = (1 - cs) / 2
            b1 = 1 - cs
            b2 = (1 - cs) / 2
            a0 = 1 + alpha
            a1 = -2 * cs
            a2 = 1 - alpha
        elif self.config_type is HighPassFilter:
            b0 = (1 + cs) / 2
            b1 = -(1 + cs)
            b2 = (1 + cs) / 2
            a0 = 1 + alpha
            a1 = -2 * cs
            a2 = 1 - alpha
        else:
            raise ValueError(f"Unknown config type: {self.config_type}")
        
        # SOS format: [b0, b1, b2, a0, a1, a2] normalized by a0
        self.sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)
    
    def __repr__(self):
        type_name = self.config_type.__name__.replace('Config', '')
        steep_str = f", steep={self.steepness}dB/oct" if self.config_type in [LowPassFilter, HighPassFilter] else ""
        return f"EQBand(type={type_name:<10}, freq={self.frequency:>7.1f} Hz, gain={self.gain_db:>+6.1f} dB, Q={self.q:.2f}{steep_str})"


class ParametricEQ:
    """Chain of parametric EQ bands using combined SOS filter for efficiency."""
    
    def __init__(self, samplerate: float, n_channels: int = 2, 
                 bands: list = None):
        """
        Initialize the parametric EQ.
        
        Args:
            samplerate: Sample rate in Hz
            n_channels: Number of audio channels
            bands: List of band configuration objects
        """
        self.samplerate = samplerate
        self.n_channels = n_channels
        self.bands: list[ParametricEQBand] = []
        self._sos = None
        self._zi = None
        
        # Initialize bands from config list
        if bands:
            for config in bands:
                # Extract parameters based on config type
                gain_db = getattr(config, 'gain_db', 0.0)
                steepness = getattr(config, 'steepness', 12)
                
                band = ParametricEQBand(
                    config.frequency,
                    gain_db,
                    config.q,
                    self.samplerate,
                    config_type=type(config),
                    steepness=steepness
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
        
        # Stack all SOS sections
        self._sos = np.vstack([band.sos for band in self.bands])
        # Initialize filter state: shape (n_sections, n_channels, 2)
        self._zi = np.zeros((self._sos.shape[0], self.n_channels, 2), dtype=np.float64)
    
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


class AudioProcessorApp:
    """Main application class for audio processing and visualization."""
    
    def __init__(self):
        self.eq = ParametricEQ(SAMPLERATE, CHANNELS, bands=EQ_BANDS)
        self.plot_queue = queue.Queue()

        # print setup
        self._print_io()
        print(f"Sample rate: {SAMPLERATE} Hz, Channels: {CHANNELS}, Block size: {BLOCKSIZE}")
        print(self.eq)

        if ENABLE_VISUALIZATION:
            self._setup_visualization()

    def _print_io(self):
        print("Available audio devices:")
        for i, device in enumerate(sd.query_devices()):
            in_label = f"Input channels: {device['max_input_channels']}"
            out_label = f"Output channels: {device['max_output_channels']}"
            
            # Bold the labels for the devices currently in use
            if i == INPUT_DEVICE:
                in_label = f"\033[1m{in_label}\033[0m"
            if i == OUTPUT_DEVICE:
                out_label = f"\033[1m{out_label}\033[0m"
                
            print(f"  [{i}] {device['name']} ({in_label}, {out_label})")

    def _setup_visualization(self):
        """Initialize the matplotlib plots."""

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        self.plt = plt
        self.FuncAnimation = FuncAnimation

        self.fig, (self.ax_eq, self.ax_spec) = self.plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                                               gridspec_kw={'height_ratios': [1, 2]})
        self.x_freqs = np.fft.rfftfreq(BLOCKSIZE, 1/SAMPLERATE)

        # --- Top Plot: EQ Curve ---
        eq_freqs, eq_mag = self.eq.get_frequency_response(n_points=2048)
        self.ax_eq.semilogx(eq_freqs, eq_mag, 'b-', linewidth=2)
        self.ax_eq.set_ylabel('EQ Gain (dB)')
        self.ax_eq.set_title('Parametric EQ Response')
        self.ax_eq.set_ylim(-24, 24)
        self.ax_eq.grid(True, which='both', linestyle='-', alpha=0.2)
        self.ax_eq.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # --- Bottom Plot: Real-time Spectrum ---
        self.line_spectrum, = self.ax_spec.semilogx(self.x_freqs, np.full(len(self.x_freqs), -100), 'g-', alpha=0.4, label='Live')
        self.line_average, = self.ax_spec.semilogx(self.x_freqs, np.full(len(self.x_freqs), -100), 'y-', linewidth=2, label='Long-term Avg')
        self.ax_spec.set_xlim(20, SAMPLERATE / 2)
        self.ax_spec.set_ylim(-100, -40)
        self.ax_spec.set_xlabel('Frequency (Hz)')
        self.ax_spec.set_ylabel('Output Level (dB)')
        self.ax_spec.set_title('Live Output Spectrum')
        self.ax_spec.grid(True, which='both', linestyle='-', alpha=0.2)
        self.ax_spec.legend(loc='upper right')

        self.plt.tight_layout()

        # FFT window and smoothing state
        self.window = np.hanning(BLOCKSIZE)
        self.smooth_mag = np.full(len(self.x_freqs), -100.0)
        self.avg_mag = np.full(len(self.x_freqs), -100.0)
        self.alpha_smooth = 0.6
        self.alpha_avg = 0.15

    def update_plot(self, frame):
        """Update the spectrum lines from the queue data."""
        data = None
        while not self.plot_queue.empty():
            data = self.plot_queue.get()
        
        if data is not None:
            fft_data = np.fft.rfft(data[:, 0] * self.window)
            mag = np.abs(fft_data) * 2.0 / np.sum(self.window)
            freq_compensation = np.sqrt(np.arange(len(mag)))
            mag *= freq_compensation
            mag_db = 20 * np.log10(mag + 1e-10)
            
            self.smooth_mag = (self.alpha_smooth * mag_db) + (1 - self.alpha_smooth) * self.smooth_mag
            self.line_spectrum.set_ydata(self.smooth_mag)
            
            self.avg_mag = (self.alpha_avg * mag_db) + (1 - self.alpha_avg) * self.avg_mag
            self.line_average.set_ydata(self.avg_mag)
        
        return self.line_spectrum, self.line_average

    def callback(self, indata, outdata, frames, time, status):
        """Audio stream callback."""
        if status:
            print(status, file=sys.stderr)
        
        processed = self.eq.process(indata)
        outdata[:] = processed
        
        if ENABLE_VISUALIZATION:
            try:
                self.plot_queue.put_nowait(processed.copy())
            except queue.Full:
                pass

    def start_stream(self):
        """Start the audio stream."""
        stream = sd.Stream(
            device=(INPUT_DEVICE, OUTPUT_DEVICE),
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=self.callback,
            latency=LATENCY
        )
        stream.start()
        return stream

    def stop_stream(self, stream):
        """Stop the audio stream."""
        if stream:
            stream.stop()
            stream.close()

    def run(self):
        """Start the audio stream and visualization."""
        
        stream = self.start_stream()
        
        if ENABLE_VISUALIZATION:
            print("\nStarting real-time spectrum analyzer...")
            ani = self.FuncAnimation(self.fig, self.update_plot, interval=5, blit=True, cache_frame_data=False)
            self.plt.show()
        else:
            print(f"Input latency: {stream.latency[0]*1000:.2f}ms\tOutput latency: {stream.latency[1]*1000:.2f}ms")
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                print("\nStopped.")
        
        self.stop_stream(stream)


class MenuBarApp(rumps.App):
    """macOS Menu Bar application wrapper."""
    
    def __init__(self, processor):
        super(MenuBarApp, self).__init__("EQ", template=True)
        self.processor = processor
        self.stream = None
        
        self.toggle_button = rumps.MenuItem("Start EQ", callback=self.toggle_eq)
        self.menu = [self.toggle_button]
        
        # Start automatically
        self.toggle_eq(None)

    def toggle_eq(self, sender):
        if self.stream is None:
            try:
                self.stream = self.processor.start_stream()
                self.toggle_button.title = "Stop EQ"
                self.title = "EQ ON"
            except Exception as e:
                rumps.alert("Error", f"Could not start audio stream: {e}")
        else:
            self.processor.stop_stream(self.stream)
            self.stream = None
            self.toggle_button.title = "Start EQ"
            self.title = "EQ"


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    app_logic = AudioProcessorApp()
    app = MenuBarApp(app_logic)
    app.run()