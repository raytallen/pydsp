#!/usr/bin/env python3.13

import sounddevice as sd
import numpy as np
from scipy.signal import sosfilt, butter
from dataclasses import dataclass
import sys
import rumps
import yaml
import os

@dataclass
class AudioPathConfig:
    name: str
    in_channels: list[int]
    out_channels: list[int]
    eq_bands: list
    mono_mix: bool = False
    gain_db: float = 0.0
    invert: bool = False

@dataclass
class LowPassFilter:
    frequency: float
    q: float = 0.707
    steepness: int = 12

@dataclass
class HighPassFilter:
    frequency: float
    q: float = 0.707
    steepness: int = 12

@dataclass
class LowShelf:
    frequency: float
    gain_db: float
    q: float = 0.707

@dataclass
class HighShelf:
    frequency: float
    gain_db: float
    q: float = 0.707

@dataclass
class ParametricBand:
    frequency: float
    gain_db: float
    q: float = 1.0

def load_config(filename="config.yaml"):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Map filter types to classes
    filter_map = {
        'LowPassFilter': LowPassFilter,
        'HighPassFilter': HighPassFilter,
        'LowShelf': LowShelf,
        'HighShelf': HighShelf,
        'ParametricBand': ParametricBand
    }
    
    audio_paths = []
    for path_data in config_data['audio_paths']:
        eq_bands = []
        for band_data in path_data['eq_bands']:
            filter_type = band_data.pop('type')
            filter_class = filter_map[filter_type]
            eq_bands.append(filter_class(**band_data))
        
        path_data['eq_bands'] = eq_bands
        audio_paths.append(AudioPathConfig(**path_data))
    
    return config_data, audio_paths

# Load configuration
config_data, AUDIO_PATHS = load_config()

INPUT_DEVICE_NAME = config_data['input_device_name']
OUTPUT_DEVICE_NAME = config_data['output_device_name']
SAMPLERATE = config_data.get('samplerate', 44100)
BLOCKSIZE = config_data.get('blocksize', 512)
LATENCY = config_data.get('latency', 0)

# Calculate total channels needed
INPUT_CHANNELS = max([max(p.in_channels) for p in AUDIO_PATHS]) + 1
OUTPUT_CHANNELS = max([max(p.out_channels) for p in AUDIO_PATHS]) + 1

class ParametricEQBand:
    """A single parametric EQ band using biquad filters."""
    
    def __init__(self, frequency: float, gain_db: float, q: float, samplerate: float, 
                 config_type: type, steepness: int):

        self.samplerate = samplerate
        self.frequency = frequency
        self.gain_db = gain_db
        self.q = q
        self.config_type = config_type
        self.steepness = steepness
        
        self._calculate_coefficients()
    
    def _calculate_coefficients(self):
        """Calculate biquad filter coefficients in SOS format."""
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
        
        # Add a tiny amount of noise to prevent denormals
        data = data + 1e-18
        
        output, self._zi = sosfilt(self._sos, data, axis=0, zi=self._zi)
        return output
    
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
    in_idx: slice | list[int]
    out_idx: slice | list[int]

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
            
            # Initialize EQ (gain is now handled separately)
            eq = ParametricEQ(
                SAMPLERATE, 
                n_channels=n_out, 
                bands=path_cfg.eq_bands
            )
            
            # OPTIMIZATION: Use slices for contiguous channels to avoid fancy indexing copies
            in_idx = self._get_best_index(path_cfg.in_channels)
            out_idx = self._get_best_index(path_cfg.out_channels)
            
            self.paths.append(AudioPath(
                config=path_cfg, 
                eq=eq, 
                in_idx=in_idx, 
                out_idx=out_idx
            ))
        
        self.max_input_peaks = np.zeros(INPUT_CHANNELS)
        self.max_output_peaks = np.zeros(OUTPUT_CHANNELS)
        
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
            invert_str = " (INVERTED)" if p.config.invert else ""
            print(f"\nPath: {p.config.name}{invert_str}")
            print(p.eq)

    def _get_best_index(self, channels):
        """Return a slice if channels are contiguous, else the list."""
        if not channels:
            return slice(0, 0)
        if len(channels) == 1:
            return slice(channels[0], channels[0] + 1)
        
        # Check if contiguous
        sorted_ch = sorted(channels)
        if sorted_ch == list(range(min(channels), max(channels) + 1)):
            return slice(min(channels), max(channels) + 1)
        return channels

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

    def callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time, status):
        """Audio stream callback."""
        
        # Clear output buffer (fast in NumPy)
        outdata.fill(0)
        
        for p in self.paths:
            # Extract input channels (view if slice, copy if list)
            path_input = indata[:, p.in_idx]
            
            # Apply mono mix if requested
            if p.config.mono_mix:
                if path_input.shape[1] == 2:
                    path_input = (path_input[:, 0:1] + path_input[:, 1:2]) * 0.5
                else:
                    path_input = path_input.mean(axis=1, keepdims=True)
            
            # Process through EQ
            processed = p.eq.process(path_input)
            
            # Apply gain separately (calculated here to allow for future dynamic changes)
            if p.config.gain_db != 0:
                processed *= 10 ** (p.config.gain_db / 20)
            
            # Apply phase inversion if requested
            if p.config.invert:
                processed *= -1.0
                
            # Map to output channels
            outdata[:, p.out_idx] = processed
        
        # Track peak levels for headroom monitoring
        self.max_input_peaks = np.maximum(self.max_input_peaks, np.max(np.abs(indata), axis=0))
        self.max_output_peaks = np.maximum(self.max_output_peaks, np.max(np.abs(outdata), axis=0))

    def stop_stream(self):
        """Stop the audio stream."""
        if self.stream:
            self.stream.stop()

    def print_latency(self):
        """Print the current stream latency."""
        if self.stream:
            print(f"Input latency: {self.stream.latency[0]*1000:.2f}ms\tOutput latency: {self.stream.latency[1]*1000:.2f}ms")

    def reset_peaks(self):
        """Reset the maximum peak trackers."""
        self.max_input_peaks.fill(0)
        self.max_output_peaks.fill(0)

    def get_headroom(self):
        """Return the current headroom in dB for input and output."""
        in_peaks = np.maximum(self.max_input_peaks, 1e-10)
        out_peaks = np.maximum(self.max_output_peaks, 1e-10)
        
        in_headroom = -20 * np.log10(np.max(in_peaks))
        out_headroom = -20 * np.log10(np.max(out_peaks))
        
        return in_headroom, out_headroom

    def start_stream(self):
        """Start the audio stream and print latency."""
        if self.stream and not self.stream.active:
            self.stream.start()
            self.print_latency()


class MenuBarApp(rumps.App):
    """macOS Menu Bar application wrapper."""
    
    def __init__(self, processor: AudioProcessorApp):
        super(MenuBarApp, self).__init__("EQ", template=True)
        self.processor = processor
        
        self.toggle_button = rumps.MenuItem("Start EQ", callback=self.toggle_eq)
        self.in_headroom_item = rumps.MenuItem("Input Headroom: -- dB")
        self.out_headroom_item = rumps.MenuItem("Output Headroom: -- dB")
        self.reset_peaks_button = rumps.MenuItem("Reset Peaks", callback=self.reset_peaks)
        
        self.menu = [
            self.toggle_button,
            None, # Separator
            self.in_headroom_item,
            self.out_headroom_item,
            self.reset_peaks_button
        ]
        
        # Timer to update headroom display
        self.timer = rumps.Timer(self.update_headroom, 1.0)
        self.timer.start()
        
        # Start automatically
        self.toggle_eq(None)

    def update_headroom(self, sender):
        if self.processor.stream.active:
            in_hr, out_hr = self.processor.get_headroom()
            
            # Update menu items
            self.in_headroom_item.title = f"Input Headroom: {in_hr:.1f} dB"
            self.out_headroom_item.title = f"Output Headroom: {out_hr:.1f} dB"
            
            # Visual warnings
            if in_hr <= 0.1:
                self.in_headroom_item.title = f"INPUT CLIP! ({in_hr:.1f} dB)"
                self.title = "EQ IN-CLIP"
            elif out_hr <= 0.1:
                self.out_headroom_item.title = f"OUTPUT CLIP! ({out_hr:.1f} dB)"
                self.title = "EQ OUT-CLIP"
            else:
                self.title = "EQ ON"
        else:
            self.in_headroom_item.title = "Input Headroom: -- dB"
            self.out_headroom_item.title = "Output Headroom: -- dB"
            self.title = "EQ"

    def reset_peaks(self, sender):
        self.processor.reset_peaks()
        print("Peaks reset.")

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