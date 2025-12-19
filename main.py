#!/usr/bin/env python3.13

import sounddevice as sd
import numpy as np
from scipy.signal import sosfilt, sosfreqz, butter
from dataclasses import dataclass
import queue
import sys
import rumps

# --- Configuration ---
INPUT_DEVICE_NAME = "BlackHole 2ch"
OUTPUT_DEVICE_NAME = "Scarlett"

SAMPLERATE = 192000
BLOCKSIZE = 64
LATENCY = 0
ENABLE_VISUALIZATION = False

CROSSOVER_FREQ = 100

@dataclass
class AudioPathConfig:
    """Configuration for an audio processing path."""
    name: str
    in_channels: list[int]
    out_channels: list[int]
    eq_bands: list
    mono_mix: bool = False
    gain_db: float = 0.0

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

# Define the processing paths
AUDIO_PATHS = [
    AudioPathConfig(
        name="Stereo Speakers",
        in_channels=[0, 1],
        out_channels=[0, 1],
        gain_db=0.0,
        eq_bands=[
            HighPassFilter(frequency=CROSSOVER_FREQ, q=0.707, steepness=24),
        ]
    ),
    AudioPathConfig(
        name="Subwoofer",
        in_channels=[0, 1],
        out_channels=[2],
        mono_mix=True,
        gain_db=-24.0,
        eq_bands=[
            LowPassFilter(frequency=CROSSOVER_FREQ, q=0.707, steepness=24),
        ]
    )
]

# Calculate total channels needed
INPUT_CHANNELS = max([max(p.in_channels) for p in AUDIO_PATHS]) + 1
OUTPUT_CHANNELS = max([max(p.out_channels) for p in AUDIO_PATHS]) + 1

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
        return f"{type_name:<16},\tfreq={self.frequency:>7.1f} Hz, gain={self.gain_db:>+6.1f} dB, Q={self.q:.2f}{steep_str}"


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
        # Initialize filter state: shape (n_sections, 2, n_channels)
        # For axis=0 in sosfilt, zi must be (n_sections, 2, n_channels)
        self._zi = np.zeros((self._sos.shape[0], 2, self.n_channels), dtype=np.float64)
    
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


@dataclass
class AudioPath:
    """Runtime instance of an audio processing path."""
    config: AudioPathConfig
    eq: ParametricEQ
    linear_gain: float = 1.0

class AudioProcessorApp:
    """Main application class for audio processing and visualization."""

    stream: sd.Stream
    paths: list[AudioPath]
    
    def __init__(self):
        # Find devices
        self.input_device_id = self._find_device(INPUT_DEVICE_NAME, is_input=True)
        self.output_device_id = self._find_device(OUTPUT_DEVICE_NAME, is_input=False)
        
        if self.input_device_id is None or self.output_device_id is None:
            raise RuntimeError("Could not find specified input or output devices.")

        # Initialize processing paths
        self.paths = []
        for path_cfg in AUDIO_PATHS:
            n_out = len(path_cfg.out_channels)
            eq = ParametricEQ(SAMPLERATE, n_channels=n_out, bands=path_cfg.eq_bands)
            linear_gain = 10 ** (path_cfg.gain_db / 20)
            self.paths.append(AudioPath(config=path_cfg, eq=eq, linear_gain=linear_gain))
        
        self.plot_queue = queue.Queue()
        
        self.stream = sd.Stream(
            device=(self.input_device_id, self.output_device_id),
            samplerate=SAMPLERATE,
            blocksize=BLOCKSIZE,
            channels=(INPUT_CHANNELS, OUTPUT_CHANNELS),
            dtype="float32",
            callback=self.callback,
            latency=LATENCY
        )

        # print setup
        self._print_io()
        print(f"Sample rate: {SAMPLERATE} Hz")
        print(f"Input: {INPUT_CHANNELS} channels, Output: {OUTPUT_CHANNELS} channels")
        print(f"Block size: {BLOCKSIZE}")
        
        for p in self.paths:
            print(f"\nPath: {p.config.name}")
            print(p.eq)

        if ENABLE_VISUALIZATION:
            self._setup_visualization()

    def _find_device(self, name_substring, is_input=True):
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if name_substring.lower() in d['name'].lower():
                if is_input and d['max_input_channels'] > 0:
                    return i
                if not is_input and d['max_output_channels'] > 0:
                    return i
        return None

    def _print_io(self):
        print("Available audio devices:")
        for i, device in enumerate(sd.query_devices()):
            in_label = f"Input channels: {device['max_input_channels']}"
            out_label = f"Output channels: {device['max_output_channels']}"
            
            # Bold the labels for the devices currently in use
            if i == self.input_device_id:
                in_label = f"\033[1m{in_label}\033[0m"
            if i == self.output_device_id:
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

        # --- Top Plot: EQ Curves ---
        colors = ['b', 'r', 'g', 'm', 'c', 'y']
        for i, p in enumerate(self.paths):
            eq_freqs, eq_mag = p.eq.get_frequency_response(n_points=2048)
            # Add path gain to the magnitude response
            eq_mag += p.config.gain_db
            self.ax_eq.semilogx(eq_freqs, eq_mag, color=colors[i % len(colors)], 
                               linewidth=2, label=p.config.name)

        self.ax_eq.set_ylabel('EQ Gain (dB)')
        self.ax_eq.set_title('System EQ Response')
        self.ax_eq.set_ylim(-24, 24)
        self.ax_eq.grid(True, which='both', linestyle='-', alpha=0.2)
        self.ax_eq.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax_eq.legend(loc='lower left')

        # --- Bottom Plot: Real-time Spectrum ---
        self.line_spectrum, = self.ax_spec.semilogx(self.x_freqs, np.full(len(self.x_freqs), -100), 'g-', alpha=0.4, label='Live')
        self.line_average, = self.ax_spec.semilogx(self.x_freqs, np.full(len(self.x_freqs), -100), 'y-', linewidth=2, label='Long-term Avg')
        self.ax_spec.set_xlim(20, 20000)
        self.ax_spec.set_ylim(-60, 0)
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
            # Use the first channel of the first path for visualization
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
        
        # Clear output buffer
        outdata.fill(0)
        
        for p in self.paths:
            # Extract input channels
            path_input = indata[:, p.config.in_channels]
            
            # Apply mono mix if requested
            if p.config.mono_mix:
                path_input = np.mean(path_input, axis=1, keepdims=True)
            
            # Process through EQ
            processed = p.eq.process(path_input)
            
            # Apply path gain
            if p.linear_gain != 1.0:
                processed *= p.linear_gain
            
            # Map to output channels
            outdata[:, p.config.out_channels] = processed
        
        if ENABLE_VISUALIZATION:
            try:
                self.plot_queue.put_nowait(outdata.copy())
            except queue.Full:
                pass

    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()

    def print_latency(self):
        """Print the current stream latency."""
        if self.stream:
            print(f"Input latency: {self.stream.latency[0]*1000:.2f}ms\tOutput latency: {self.stream.latency[1]*1000:.2f}ms")

    def start_stream(self):
        """Start the audio stream, print latency, and handle visualization."""
        if self.stream and not self.stream.active:
            self.stream.start()
            self.print_latency()
            if ENABLE_VISUALIZATION:
                print("\nStarting real-time spectrum analyzer...")
                self.ani = self.FuncAnimation(self.fig, self.update_plot, interval=5, blit=True, cache_frame_data=False)
                self.plt.show()


class MenuBarApp(rumps.App):
    """macOS Menu Bar application wrapper."""
    
    def __init__(self, processor: AudioProcessorApp):
        super(MenuBarApp, self).__init__("EQ", template=True)
        self.processor = processor
        
        self.toggle_button = rumps.MenuItem("Start EQ", callback=self.toggle_eq)
        self.menu = [self.toggle_button]
        
        # Start automatically
        self.toggle_eq(None)

    def toggle_eq(self, sender):
        if not self.processor.stream.active:
            try:
                self.processor.start_stream()
                self.toggle_button.title = "Stop EQ"
                self.title = "EQ ON"
            except Exception as e:
                rumps.alert("Error", f"Could not start audio stream: {e}")
        else:
            self.processor.stop_stream()
            self.toggle_button.title = "Start EQ"
            self.title = "EQ"


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    app_logic = AudioProcessorApp()
    app = MenuBarApp(app_logic)
    app.run()