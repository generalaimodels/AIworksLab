#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Visualization Pipeline (single-file, from-scratch, high-signal-to-noise).

This file implements a professional-grade, pedagogical audio visualization toolkit with:
- Robust audio container (Audio) with resampling and decoding fallbacks
- Core DSP utilities: Slaney mel scale (Hz<->mel), triangular mel filter bank, mel filter bank matrix,
  and log-mel spectrogram computation
- Seven visualization methods (saved to a 'plots/' directory) that present the signal and features
  with professional plotting: axes/labels/legends/titles/consistent styling
- A didactic pipeline that explains linear vs log vs mel scale with side-by-side subplots

Each method is thoroughly documented with technical notes and best practices to help coders master
audio analysis and visualization. All explanations live within this file (docstrings and comments).

Dependencies (auto-detected; graceful fallbacks where possible):
- numpy (required)
- matplotlib (required)
- soundfile (optional decode/encode; recommended)
- pydub (optional decode/encode; requires ffmpeg)
- soxr (optional, best resampler)
- scipy (optional, STFT and resample_poly fallback)

Output:
- All plots are written to the 'plots/' folder with informative filenames.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import os
import textwrap
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Optional imports (graceful fallback)
try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    import soxr as _soxr
except Exception:
    _soxr = None

try:
    from scipy import signal as sp_signal
except Exception:
    sp_signal = None

# Global numeric epsilon for stability
_EPS = np.finfo(np.float32).eps


# -----------------------------------------------------------------------------
# Plot styling: professional, legible, and consistent across all figures
# -----------------------------------------------------------------------------
def _setup_matplotlib_style() -> None:
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "figure.figsize": (12, 6),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlepad": 10.0,
        "axes.labelpad": 6.0,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.borderpad": 0.4,
        "legend.fontsize": 9,
        "font.size": 10,
        "image.cmap": "magma",
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
    })


_setup_matplotlib_style()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    x = float(n)
    while x >= 1024.0 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    return f"{x:.2f} {units[idx]}"


def _timecode(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    h, m = divmod(m, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _safe_clip(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip(x, vmin, vmax, out=np.empty_like(x))


# -----------------------------------------------------------------------------
# Configuration dataclasses (common, waveform, spectrogram)
# -----------------------------------------------------------------------------
@dataclass
class CommonSettings:
    """
    Common settings controlling time-range cropping and output directory.

    time_range: limit visualization to (t_start, t_end) seconds. None => full file.
    output_dir: directory to save plots. Created if missing.
    """
    time_range: Optional[Tuple[float, float]] = None
    output_dir: str = "plots"


@dataclass
class WaveformSettings:
    """
    Waveform visualization settings.

    visible: if False, waveform panels are omitted where applicable.
    amplitude_range: y-axis range for amplitude (e.g., (-1, 1)). None => auto-scale.
    """
    visible: bool = True
    amplitude_range: Optional[Tuple[float, float]] = (-1.0, 1.0)


@dataclass
class SpectrogramSettings:
    """
    Spectrogram computation and visualization settings.

    visible: whether to draw spectrogram panels in multi-view methods.
    window_size: STFT window length (samples)
    hop_length: STFT hop length (samples) [if None, defaults to window_size//4].
    window: window type string (e.g., 'hann')
    frequency_scale: 'linear', 'log', or 'mel' (plotting axis scaling/hz<->mel mapping)
    frequency_range: frequency axis clipping (Hz), e.g., (30, 8000). None => (0, sr/2)
    top_db: dynamic range clipping for dB-scaled spectrograms
    cmap: colormap for images
    """
    visible: bool = True
    window_size: int = 1024
    hop_length: Optional[int] = None
    window: str = "hann"
    frequency_scale: str = "mel"  # 'linear' | 'log' | 'mel'
    frequency_range: Optional[Tuple[float, float]] = None
    top_db: float = 80.0
    cmap: str = "magma"


# -----------------------------------------------------------------------------
# Raw audio protocol for method-7 pipeline
# -----------------------------------------------------------------------------
@dataclass
class RawAudio:
    """
    Raw PCM audio descriptor used by method-7 (from raw audio -> log-mel pipeline).

    data: interleaved PCM samples
    sample_rate: Hz
    channels: number of channels
    sample_width: bytes per sample per channel (1|2|3|4)
    endianness: 'little' or 'big' for widths > 1
    signed: True for signed PCM (typical for >=16-bit)
    """
    data: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2
    endianness: str = "little"
    signed: bool = True


# -----------------------------------------------------------------------------
# DSP: Slaney Mel scale (Hz <-> Mel), Filters, and Log-Mel Spectrogram
# -----------------------------------------------------------------------------
def hertz_to_mel(f_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Hertz to Mel using the Slaney formulation (linear below 1 kHz, log above).

    This mapping mirrors the widely used implementation in librosa (htk=False).
    """
    f = np.asanyarray(f_hz, dtype=float)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15 mels
    logstep = np.log(6.4) / 27.0

    m = np.empty_like(f)
    lin = f < min_log_hz
    m[lin] = f[lin] / f_sp
    m[~lin] = min_log_mel + np.log(f[~lin] / min_log_hz) / logstep
    if np.isscalar(f_hz):
        return float(m)
    return m


def mel_to_hertz(m_mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Inverse of hertz_to_mel: Slaney Mel to Hertz.
    """
    m = np.asanyarray(m_mel, dtype=float)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    f = np.empty_like(m)
    lin = m < min_log_mel
    f[lin] = m[lin] * f_sp
    f[~lin] = min_log_hz * np.exp(logstep * (m[~lin] - min_log_mel))
    if np.isscalar(m_mel):
        return float(f)
    return f


def _create_triangular_filter_bank(
    fft_freqs_hz: np.ndarray,
    mel_center_hz: np.ndarray,
    norm: Optional[str] = "slaney",
) -> np.ndarray:
    """
    Build triangular mel filters given FFT bin frequencies (Hz) and mel-spaced centers (Hz).

    mel_center_hz must include edges: length n_mels+2. Filters are triangles spanning
    [center[i-1], center[i], center[i+1]].

    norm: 'slaney' applies equal-area scaling (recommended).
    """
    if mel_center_hz.ndim != 1 or fft_freqs_hz.ndim != 1:
        raise ValueError("fft_freqs_hz and mel_center_hz must be 1-D arrays.")
    if not np.all(np.diff(fft_freqs_hz) >= 0):
        raise ValueError("fft_freqs_hz must be non-decreasing.")
    if mel_center_hz.size < 3:
        raise ValueError("mel_center_hz must have at least 3 values (edges included).")

    n_mels = mel_center_hz.size - 2
    n_bins = fft_freqs_hz.size
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)

    for i in range(n_mels):
        left = mel_center_hz[i]
        center = mel_center_hz[i + 1]
        right = mel_center_hz[i + 2]

        left_slope = (fft_freqs_hz - left) / max(center - left, _EPS)
        right_slope = (right - fft_freqs_hz) / max(right - center, _EPS)
        fb[i, :] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    if norm and norm.lower() == "slaney":
        scale = 2.0 / (mel_center_hz[2:] - mel_center_hz[:-2] + _EPS)
        fb *= scale[:, None]

    return fb


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = "slaney",
) -> np.ndarray:
    """
    Create a Slaney mel filter bank matrix of shape (n_mels, 1 + n_fft//2).
    """
    if fmax is None:
        fmax = sr / 2.0
    if not (0.0 <= fmin < fmax <= sr / 2.0 + 1e-6):
        raise ValueError("Require 0 <= fmin < fmax <= Nyquist (sr/2).")

    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1, dtype=float)
    m_min = hertz_to_mel(fmin)
    m_max = hertz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, num=n_mels + 2)
    hz_pts = mel_to_hertz(m_pts)
    fb = _create_triangular_filter_bank(fft_freqs, hz_pts, norm=norm)
    return fb.astype(np.float32)


def _stft(
    x: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: Optional[int],
    win_length: Optional[int],
    window: str,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT with scipy fallback; returns (frequencies, times, complex STFT).
    """
    hop = hop_length if hop_length is not None else n_fft // 4
    win = win_length if win_length is not None else n_fft
    if sp_signal is not None:
        w = sp_signal.get_window(window, win, fftbins=True)
        f, t, Zxx = sp_signal.stft(
            x, fs=sr, nperseg=win, noverlap=win - hop, nfft=n_fft, window=w,
            boundary="even" if center else None, padded=True, return_onesided=True
        )
        return f.astype(float), t.astype(float), Zxx.astype(np.complex64)
    # Minimal numpy fallback
    if center:
        x = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    w = np.hanning(win).astype(np.float32) if window == "hann" else np.ones(win, dtype=np.float32)
    n_frames = 1 + max(0, (len(x) - win) // hop)
    f = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1, dtype=float)
    t = (np.arange(n_frames) * hop) / float(sr)
    Z = np.empty((len(f), n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + win]
        if frame.size < win:
            frame = np.pad(frame, (0, win - frame.size))
        Z[:, i] = np.fft.rfft(frame * w, n=n_fft)
    return f, t, Z


def _power_spectrogram(x: np.ndarray, sr: int, n_fft: int, hop: Optional[int], win: Optional[int], window: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Zxx = _stft(x, sr, n_fft, hop, win, window, center=True)
    S = np.abs(Zxx) ** 2
    return f, t, S.astype(np.float32)


def _amplitude_to_db(S: np.ndarray, top_db: Optional[float] = 80.0) -> np.ndarray:
    S = np.maximum(S, _EPS)
    S_db = 10.0 * np.log10(S)
    if top_db is not None:
        S_db = np.maximum(S_db, np.max(S_db) - float(top_db))
    return S_db.astype(np.float32)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    top_db: float = 80.0,
    mel_norm: Optional[str] = "slaney",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute log-mel spectrogram and return:
      S_mel_db: (n_mels, n_frames), mel_fb: (n_mels, 1+n_fft//2), mel_centers_hz: (n_mels,), times: (n_frames,)
    """
    x = waveform.mean(axis=1) if waveform.ndim == 2 else waveform
    x = x.astype(np.float32)
    hop = hop_length if hop_length is not None else n_fft // 4

    f, t, S = _power_spectrogram(x, sample_rate, n_fft, hop_length, win_length, window)
    fb = mel_filter_bank(sample_rate, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=mel_norm)
    M = fb @ S
    M = np.maximum(M, _EPS)
    S_mel_db = _amplitude_to_db(M, top_db=top_db)

    m_edges = np.linspace(hertz_to_mel(fmin), hertz_to_mel(fmax if fmax is not None else sample_rate / 2.0), num=n_mels + 2)
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    mel_centers_hz = mel_to_hertz(m_centers).astype(np.float32)
    return S_mel_db, fb, mel_centers_hz, t.astype(np.float32)


# -----------------------------------------------------------------------------
# Audio class with robust I/O and resampling
# -----------------------------------------------------------------------------
class Audio:
    """
    Immutable audio container with float32 waveform in [-1,1]; shape: (N, C).
    """

    def __init__(self, data: np.ndarray, rate: int, fmt: Optional[str] = None, source_path: Optional[str] = None):
        x = np.asanyarray(data)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2:
            raise ValueError("data must be shape (N,) or (N, C).")
        x = x.astype(np.float32)
        x = np.clip(x, -1.0, 1.0, out=x)
        if rate <= 0:
            raise ValueError("rate must be positive.")
        self._data = x
        self._rate = int(rate)
        self._format = fmt.upper() if isinstance(fmt, str) else fmt
        self._path = source_path

    def __repr__(self) -> str:
        n, c = self._data.shape
        return f"Audio(rate={self._rate}, duration={n/self._rate:.3f}s, shape=({n},{c}), format={self._format or 'Unknown'})"

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def rate(self) -> int:
        return self._rate

    @property
    def channels(self) -> int:
        return self._data.shape[1]

    @property
    def duration(self) -> float:
        return self._data.shape[0] / float(self._rate)

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def source_path(self) -> Optional[str]:
        return self._path

    @staticmethod
    def _pcm_to_float(x: np.ndarray, sample_width: int) -> np.ndarray:
        if x.dtype.kind == "f":
            return x.astype(np.float32, copy=False)
        if sample_width == 1:
            # uint8 -> float32 in [-1,1]
            if x.dtype == np.uint8:
                return ((x.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
            return (x.astype(np.float32) / 128.0).astype(np.float32)
        if sample_width == 2:
            return (x.astype(np.float32) / 32768.0).astype(np.float32)
        if sample_width == 3:
            return (x.astype(np.float32) / (2 ** 23)).astype(np.float32)
        if sample_width == 4:
            return (x.astype(np.float32) / (2 ** 31)).astype(np.float32)
        raise ValueError("Unsupported sample width")

    @classmethod
    def from_file(cls, path: str) -> "Audio":
        fmt = os.path.splitext(path)[1].lstrip(".").upper() or None
        data = None
        rate = None
        if sf is not None:
            try:
                y, sr = sf.read(path, dtype="float32", always_2d=True)
                data, rate = y, int(sr)
                info = sf.info(path)
                fmt = info.format or fmt
            except Exception:
                data = None
        if data is None and AudioSegment is not None:
            try:
                seg = AudioSegment.from_file(path)
                rate = seg.frame_rate
                ch = seg.channels
                sw = seg.sample_width
                arr = np.array(seg.get_array_of_samples())
                if ch > 1:
                    arr = arr.reshape(-1, ch)
                else:
                    arr = arr.reshape(-1, 1)
                data = cls._pcm_to_float(arr, sw)
            except Exception:
                data = None
        if data is None:
            # Minimal WAV fallback
            import wave
            with wave.open(path, "rb") as w:
                rate = w.getframerate()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                frames = w.getnframes()
                pcm = w.readframes(frames)
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, None)
            if dtype is None:
                raise RuntimeError("Unsupported WAV sample width without soundfile/pydub.")
            arr = np.frombuffer(pcm, dtype=dtype)
            if ch > 1:
                arr = arr.reshape(-1, ch)
            else:
                arr = arr.reshape(-1, 1)
            data = cls._pcm_to_float(arr, sw)
            fmt = "WAV"
        return cls(data, rate, fmt=fmt, source_path=path)

    @classmethod
    def from_raw(cls, raw: RawAudio) -> "Audio":
        # Build dtype
        if raw.sample_width == 1:
            dtype = np.uint8 if not raw.signed else np.int8
            x = np.frombuffer(raw.data, dtype=dtype)
            if x.size % raw.channels != 0:
                raise ValueError("Raw data size not divisible by channels.")
            x = x.reshape(-1, raw.channels)
            x = cls._pcm_to_float(x, raw.sample_width)
        elif raw.sample_width in (2, 4):
            endian = "<" if raw.endianness == "little" else ">"
            dt_char = "i" if raw.signed else "u"
            dt = np.dtype(f"{endian}{dt_char}{raw.sample_width}")
            x = np.frombuffer(raw.data, dtype=dt)
            if x.size % raw.channels != 0:
                raise ValueError("Raw data size not divisible by channels.")
            x = x.reshape(-1, raw.channels)
            x = cls._pcm_to_float(x, raw.sample_width)
        elif raw.sample_width == 3:
            # 24-bit unpack
            a = np.frombuffer(raw.data, dtype=np.uint8)
            if a.size % (3 * raw.channels) != 0:
                raise ValueError("24-bit raw data size not divisible by 3*channels.")
            a = a.reshape(-1, raw.channels, 3)
            if raw.endianness == "little":
                val = (a[:, :, 0].astype(np.uint32)
                       | (a[:, :, 1].astype(np.uint32) << 8)
                       | (a[:, :, 2].astype(np.uint32) << 16)).astype(np.int32)
            else:
                val = (a[:, :, 2].astype(np.uint32)
                       | (a[:, :, 1].astype(np.uint32) << 8)
                       | (a[:, :, 0].astype(np.uint32) << 16)).astype(np.int32)
            if raw.signed:
                sign_mask = 1 << 23
                val = (val ^ sign_mask) - sign_mask
            x = (val.astype(np.float32) / (2 ** 23)).astype(np.float32)
        else:
            raise ValueError("Unsupported sample_width for raw audio.")
        return cls(x, raw.sample_rate, fmt="RAW", source_path=None)

    def resample(self, new_rate: int, quality: str = "HQ") -> "Audio":
        if new_rate <= 0:
            raise ValueError("new_rate must be positive.")
        if new_rate == self._rate:
            return Audio(self._data.copy(), self._rate, fmt=self._format, source_path=self._path)
        y = self._data
        if _soxr is not None:
            if y.ndim == 2:
                outs = [_soxr.resample(y[:, c], self._rate, new_rate, quality=quality).astype(np.float32) for c in range(y.shape[1])]
                y_out = np.stack(outs, axis=1)
            else:
                y_out = _soxr.resample(y, self._rate, new_rate, quality=quality).astype(np.float32)
        elif sp_signal is not None:
            from math import gcd
            g = gcd(self._rate, new_rate)
            up = new_rate // g
            down = self._rate // g
            y_out = sp_signal.resample_poly(y, up, down, axis=0).astype(np.float32)
        else:
            # Nearest-neighbor fallback (not ideal; instructive).
            ratio = new_rate / float(self._rate)
            idx = (np.arange(int(y.shape[0] * ratio)) / ratio).astype(int)
            idx = np.minimum(idx, y.shape[0] - 1)
            y_out = y[idx]
        return Audio(y_out, new_rate, fmt=self._format, source_path=self._path)


# -----------------------------------------------------------------------------
# Visualization primitives
# -----------------------------------------------------------------------------
def _crop_time(audio: Audio, time_range: Optional[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    x = audio.data
    sr = audio.rate
    N = x.shape[0]
    t_full = np.arange(N) / float(sr)
    if time_range is None:
        return t_full, x
    t0, t1 = time_range
    t0 = max(0.0, t0)
    t1 = min(audio.duration, t1)
    if t1 <= t0:
        return t_full, x
    i0 = int(round(t0 * sr))
    i1 = int(round(t1 * sr))
    return t_full[i0:i1], x[i0:i1, :]


def _draw_metadata(fig: matplotlib.figure.Figure, meta: Dict[str, str]) -> None:
    # Place metadata as a neat textbox at the bottom
    text = "\n".join(f"{k}: {v}" for k, v in meta.items())
    fig.text(0.01, 0.01, text, fontsize=9, family="monospace",
             bbox=dict(facecolor="#f9f9f9", edgecolor="#cccccc", boxstyle="round,pad=0.5", alpha=0.95))


def _savefig(fig: matplotlib.figure.Figure, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Method 1: original audio plot with channel/resampling detail
# -----------------------------------------------------------------------------
def method1_plot_original_with_resampling(
    audio_or_path: Union[str, Audio],
    target_rates: Iterable[int] = (16000, 8000),
    common: CommonSettings = CommonSettings(),
    wave: WaveformSettings = WaveformSettings(),
    title: Optional[str] = None,
    filename_suffix: str = "method1_original_and_resampled.png",
) -> str:
    """
    Plot the original audio (per channel) and resampled versions, with metadata overlay.

    - Top row: original waveform per channel (in time_range if provided)
    - Bottom row: resampled overlays (all target_rates) per channel
    - Metadata: encoding, format, channels, sample rate, file size, duration
    """
    _ensure_dir(common.output_dir)
    audio = Audio.from_file(audio_or_path) if isinstance(audio_or_path, str) else audio_or_path
    t, x = _crop_time(audio, common.time_range)
    n_channels = x.shape[1]
    channels_to_plot = n_channels

    # Prepare resampled audios
    resampled = []
    for r in target_rates:
        try:
            resampled.append((r, audio.resample(r)))
        except Exception as e:
            warnings.warn(f"Resample to {r} Hz failed: {e}", RuntimeWarning)

    # Figure with 2 rows x C columns
    fig, axes = plt.subplots(2, channels_to_plot, figsize=(14, max(6, 3 * channels_to_plot)), sharex=False, sharey=True)
    axes = np.atleast_2d(axes)

    # Top: original per channel
    for c in range(channels_to_plot):
        ax = axes[0, c]
        ax.plot(t, x[:, c], lw=0.9, label=f"Channel {c+1}")
        ax.set_title(f"Original Waveform - Ch {c+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        if wave.amplitude_range:
            ax.set_ylim(wave.amplitude_range)
        ax.legend(loc="upper right")

    # Bottom: resampled overlays per channel
    for c in range(channels_to_plot):
        ax = axes[1, c]
        for (r, a_r) in resampled:
            t_r, x_r = _crop_time(a_r, common.time_range)
            ax.plot(t_r, x_r[:, c if c < a_r.channels else 0], lw=0.9, label=f"{r/1000:.1f} kHz")
        ax.set_title(f"Resampled Overlay - Ch {c+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        if wave.amplitude_range:
            ax.set_ylim(wave.amplitude_range)
        ax.legend(loc="upper right", title="Rates")

    # Metadata
    file_size = None
    if isinstance(audio_or_path, str) and os.path.exists(audio_or_path):
        file_size = os.path.getsize(audio_or_path)
    meta = {
        "encoding": audio.format or "Unknown",
        "format": (os.path.splitext(audio.source_path)[1].lstrip(".").upper() if audio.source_path else (audio.format or "Unknown")),
        "number_of_channel": str(audio.channels),
        "sample_rate": f"{audio.rate} Hz",
        "file_size": _human_bytes(file_size),
        "duration": f"{audio.duration:.3f} s ({_timecode(audio.duration)})",
    }
    _draw_metadata(fig, meta)
    fig.suptitle(title or "Original and Resampled Waveforms", fontsize=13)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 2: visualize Hertz -> Mel mapping
# -----------------------------------------------------------------------------
def method2_plot_hertz_to_mel(
    fmax_hz: float = 22050.0,
    n_points: int = 2000,
    common: CommonSettings = CommonSettings(),
    filename_suffix: str = "method2_hz_to_mel.png",
) -> str:
    """
    Plot Slaney Hz->Mel mapping, highlighting the linear (<=1kHz) and logarithmic (>1kHz) regions.
    """
    _ensure_dir(common.output_dir)
    f = np.linspace(0.0, max(1.0, fmax_hz), n_points)
    m = hertz_to_mel(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f, m, color="#1f77b4", lw=2.0, label="Slaney Mel(f)")
    ax.axvline(1000.0, color="gray", ls="--", lw=1.0, alpha=0.7, label="1 kHz boundary")
    ax.set_title("Hertz to Mel (Slaney)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mel")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Secondary y-axis: show approximate Hz at some Mel ticks for intuition
    mel_ticks = np.linspace(0, m.max(), 6)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(mel_ticks)
    ax2.set_yticklabels([f"{mel_to_hertz(mt):.0f} Hz" for mt in mel_ticks])
    ax2.set_ylabel("Equivalent Hz (for reference)")

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 3: visualize Mel -> Hertz mapping (inverse)
# -----------------------------------------------------------------------------
def method3_plot_mel_to_hertz(
    mel_max: float = 2595.0,  # roughly mel(22050) in HTK; Slaney differs; we compute directly anyway
    n_points: int = 2000,
    common: CommonSettings = CommonSettings(),
    filename_suffix: str = "method3_mel_to_hz.png",
) -> str:
    """
    Plot Slaney Mel->Hz mapping and annotate the nonlinearity.
    """
    _ensure_dir(common.output_dir)
    m = np.linspace(0.0, mel_max, n_points)
    f = mel_to_hertz(m)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(m, f, color="#d62728", lw=2.0, label="Hz(Mel)")
    # Mark Mel at 1kHz
    mel_1k = hertz_to_mel(1000.0)
    ax.axvline(mel_1k, color="gray", ls="--", lw=1.0, alpha=0.7, label=f"Mel(1000 Hz) ≈ {mel_1k:.1f}")
    ax.set_title("Mel to Hertz (Slaney)")
    ax.set_xlabel("Mel")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 4: visualize triangular mel filters on Hz axis
# -----------------------------------------------------------------------------
def method4_plot_create_triangular_filter_bank(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 20,
    fmin: float = 30.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = "slaney",
    common: CommonSettings = CommonSettings(),
    filename_suffix: str = "method4_triangular_filters.png",
) -> str:
    """
    Construct and plot several triangular mel filters (Slaney) across the Hz axis.
    """
    _ensure_dir(common.output_dir)
    if fmax is None:
        fmax = sr / 2.0
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    m_min, m_max = hertz_to_mel(fmin), hertz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    hz_pts = mel_to_hertz(m_pts)
    fb = _create_triangular_filter_bank(fft_freqs, hz_pts, norm=norm)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(n_mels):
        ax.plot(fft_freqs, fb[i], lw=1.2, alpha=0.9)
    ax.set_xlim(fmin, fmax)
    ax.set_title(f"Triangular Mel Filters (n_mels={n_mels}, n_fft={n_fft}, sr={sr} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 5: visualize the Mel filter bank matrix as a heatmap
# -----------------------------------------------------------------------------
def method5_plot_mel_filter_bank(
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: Optional[str] = "slaney",
    common: CommonSettings = CommonSettings(),
    filename_suffix: str = "method5_mel_filter_bank.png",
) -> str:
    """
    Show the mel filter bank matrix as a heatmap (mel bands x FFT bins).
    """
    _ensure_dir(common.output_dir)
    fmax = fmax if fmax is not None else sr / 2.0
    fb = mel_filter_bank(sr, n_fft, n_mels, fmin, fmax, norm=norm)
    fft_freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(fb, aspect="auto", origin="lower", interpolation="nearest",
                   extent=[fft_freqs[0], fft_freqs[-1], 0, fb.shape[0]])
    ax.set_title(f"Mel Filter Bank (shape {fb.shape})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mel Band Index")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Weight")
    ax.grid(False)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 6: log-mel spectrogram of an audio file
# -----------------------------------------------------------------------------
def method6_plot_log_mel_spectrogram(
    audio_or_path: Union[str, Audio],
    spec: SpectrogramSettings = SpectrogramSettings(window_size=1024, frequency_scale="mel", top_db=80.0),
    common: CommonSettings = CommonSettings(),
    n_mels: int = 80,
    fmin: float = 30.0,
    fmax: Optional[float] = None,
    title: Optional[str] = None,
    filename_suffix: str = "method6_log_mel_spectrogram.png",
) -> str:
    """
    Compute and plot the log-mel spectrogram. Also includes a linear-frequency power spectrogram for context.
    """
    _ensure_dir(common.output_dir)
    audio = Audio.from_file(audio_or_path) if isinstance(audio_or_path, str) else audio_or_path
    t, x = _crop_time(audio, common.time_range)
    x_mono = x.mean(axis=1).astype(np.float32)

    n_fft = spec.window_size
    hop = spec.hop_length if spec.hop_length is not None else n_fft // 4
    fmax_eff = fmax if fmax is not None else audio.rate / 2.0

    # Compute linear spectrogram
    f_lin, t_lin, S_pow = _power_spectrogram(x_mono, audio.rate, n_fft, hop, None, spec.window)
    S_db = _amplitude_to_db(S_pow, top_db=spec.top_db)

    # Log-mel
    S_mel_db, fb, mel_centers_hz, t_mel = log_mel_spectrogram(
        x_mono, audio.rate, n_fft=n_fft, hop_length=hop, win_length=None, window=spec.window,
        n_mels=n_mels, fmin=fmin, fmax=fmax_eff, top_db=spec.top_db, mel_norm="slaney"
    )

    # Plot both views
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    # Linear spectrogram
    im0 = axes[0].imshow(S_db, aspect="auto", origin="lower",
                         extent=[t_lin[0] if len(t_lin) else 0, t_lin[-1] if len(t_lin) else t[-1] if len(t) else 0, f_lin[0], f_lin[-1]],
                         vmin=S_db.max() - spec.top_db, vmax=S_db.max(), cmap=spec.cmap)
    axes[0].set_title("Linear-Frequency Power Spectrogram (dB)")
    axes[0].set_ylabel("Frequency (Hz)")
    if spec.frequency_range:
        axes[0].set_ylim(spec.frequency_range)
    cb0 = fig.colorbar(im0, ax=axes[0], pad=0.01)
    cb0.set_label("dB")

    # Log-mel spectrogram
    im1 = axes[1].imshow(S_mel_db, aspect="auto", origin="lower",
                         extent=[t_mel[0] if len(t_mel) else 0, t_mel[-1] if len(t_mel) else t[-1] if len(t) else 0, 0, n_mels],
                         vmin=S_mel_db.max() - spec.top_db, vmax=S_mel_db.max(), cmap=spec.cmap)
    axes[1].set_title(f"Log-Mel Spectrogram (n_mels={n_mels})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel Band Index")
    cb1 = fig.colorbar(im1, ax=axes[1], pad=0.01)
    cb1.set_label("dB")

    # Metadata
    file_size = os.path.getsize(audio.source_path) if audio.source_path and os.path.exists(audio.source_path) else None
    meta = {
        "encoding": audio.format or "Unknown",
        "format": (os.path.splitext(audio.source_path)[1].lstrip(".").upper() if audio.source_path else (audio.format or "Unknown")),
        "number_of_channel": str(audio.channels),
        "sample_rate": f"{audio.rate} Hz",
        "file_size": _human_bytes(file_size),
        "duration": f"{audio.duration:.3f} s ({_timecode(audio.duration)})",
        "window size": f"{n_fft} samples",
        "hop length": f"{hop} samples",
        "frequency range": f"{fmin:.1f} Hz - {fmax_eff:.1f} Hz",
        "spectrogram amplitude range": f"{spec.top_db:.1f} dB dynamic",
    }
    _draw_metadata(fig, meta)
    fig.suptitle(title or "Log-Mel Spectrogram", fontsize=13)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Method 7: from raw audio -> log-mel spectrogram with all intermediate steps
# -----------------------------------------------------------------------------
def method7_plot_raw_to_logmel_pipeline(
    raw: RawAudio,
    common: CommonSettings = CommonSettings(),
    wave: WaveformSettings = WaveformSettings(),
    spec: SpectrogramSettings = SpectrogramSettings(window_size=1024, frequency_scale="mel", top_db=80.0),
    n_mels: int = 80,
    fmin: float = 30.0,
    fmax: Optional[float] = None,
    title: Optional[str] = None,
    filename_suffix: str = "method7_raw_to_logmel_pipeline.png",
) -> str:
    """
    Full pedagogical pipeline from raw PCM to log-mel spectrogram with step-by-step visuals.

    Subplot layout (3x3):
      [1] Waveform (time)             [2] Window function              [3] Frame spectrum (linear amplitude)
      [4] Power spectrum (dB)         [5] Mel filters overlay          [6] Mel energies (one frame)
      [7] Linear spectrogram (dB)     [8] Log-frequency spectrogram    [9] Log-Mel spectrogram (dB)

    This layout juxtaposes linear, log, and mel scales to highlight why mel features are robust.
    """
    _ensure_dir(common.output_dir)

    # Decode raw to Audio
    audio = Audio.from_raw(raw)
    # Crop
    t, x = _crop_time(audio, common.time_range)
    x_mono = x.mean(axis=1).astype(np.float32)

    # STFT params
    n_fft = spec.window_size
    hop = spec.hop_length if spec.hop_length is not None else n_fft // 4
    fmax_eff = fmax if fmax is not None else audio.rate / 2.0

    # Prepare window and one analysis frame (centered)
    if sp_signal is not None:
        w = sp_signal.get_window(spec.window, n_fft, fftbins=True)
    else:
        w = np.hanning(n_fft).astype(np.float32) if spec.window == "hann" else np.ones(n_fft, dtype=np.float32)
    w = w.astype(np.float32)

    # Choose a frame index near 1/3 of the signal
    if x_mono.size < n_fft:
        x_pad = np.pad(x_mono, (0, n_fft - x_mono.size))
    else:
        x_pad = x_mono
    frame_idx = max(0, min((x_pad.size - n_fft) // hop // 3, max(0, (x_pad.size - n_fft) // hop)))
    i0 = frame_idx * hop
    frame = x_pad[i0:i0 + n_fft]
    if frame.size < n_fft:
        frame = np.pad(frame, (0, n_fft - frame.size))
    frame_win = frame * w

    # Frame FFT
    F = np.fft.rfft(frame_win, n=n_fft)
    mag = np.abs(F).astype(np.float32)
    pow_spec = (mag ** 2).astype(np.float32)
    freq_axis = np.linspace(0.0, audio.rate / 2.0, n_fft // 2 + 1)

    # Mel bank and mel energies for this frame
    fb = mel_filter_bank(audio.rate, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax_eff, norm="slaney")
    mel_edges = np.linspace(hertz_to_mel(fmin), hertz_to_mel(fmax_eff), n_mels + 2)
    mel_centers_hz = mel_to_hertz(0.5 * (mel_edges[:-1] + mel_edges[1:]))
    mel_energies = fb @ pow_spec
    mel_energies = np.maximum(mel_energies, _EPS)

    # Full spectrograms
    f_lin, t_lin, S_pow = _power_spectrogram(x_mono, audio.rate, n_fft, hop, None, spec.window)
    S_db = _amplitude_to_db(S_pow, top_db=spec.top_db)
    S_mel_db, fb_all, mel_centers_hz_all, t_mel = log_mel_spectrogram(
        x_mono, audio.rate, n_fft=n_fft, hop_length=hop, win_length=None, window=spec.window,
        n_mels=n_mels, fmin=fmin, fmax=fmax_eff, top_db=spec.top_db, mel_norm="slaney"
    )

    # Build figure 3x3
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    ax = axes[0, 0]
    # [1] Waveform
    time_axis = np.arange(x_mono.size) / float(audio.rate)
    ax.plot(time_axis, x_mono, color="#1f77b4", lw=0.9)
    ax.set_title("Waveform (Time Domain)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if wave.amplitude_range:
        ax.set_ylim(wave.amplitude_range)

    # [2] Window function
    ax = axes[0, 1]
    ax.plot(np.arange(n_fft) / float(audio.rate), w, color="#2ca02c", lw=1.2)
    ax.set_title(f"Window: {spec.window} (N={n_fft})")
    ax.set_xlabel("Time (s) within Frame")
    ax.set_ylabel("Amplitude")

    # [3] Frame spectrum (linear amplitude)
    ax = axes[0, 2]
    ax.plot(freq_axis, mag, color="#ff7f0e", lw=1.0)
    ax.set_title("Single-Frame Spectrum (|X(f)|, Linear)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(fmin, fmax_eff)

    # [4] Power spectrum (dB)
    ax = axes[1, 0]
    frame_db = _amplitude_to_db(pow_spec, top_db=spec.top_db)
    ax.plot(freq_axis, frame_db, color="#d62728", lw=1.0)
    ax.set_title("Single-Frame Power Spectrum (dB)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_xlim(fmin, fmax_eff)

    # [5] Mel filters overlay
    ax = axes[1, 1]
    for i in range(n_mels):
        ax.plot(freq_axis, fb[i], lw=0.8, alpha=0.85)
    ax.set_title(f"Mel Triangular Filters (n_mels={n_mels})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight")
    ax.set_xlim(fmin, fmax_eff)

    # [6] Mel energies (one frame)
    ax = axes[1, 2]
    ax.bar(np.arange(n_mels), 10.0 * np.log10(np.maximum(mel_energies, _EPS)), width=0.8, color="#9467bd")
    ax.set_title("Mel Filterbank Energies (dB) - Single Frame")
    ax.set_xlabel("Mel Band Index")
    ax.set_ylabel("dB")

    # [7] Linear spectrogram (dB)
    ax = axes[2, 0]
    im0 = ax.imshow(S_db, aspect="auto", origin="lower",
                    extent=[t_lin[0] if len(t_lin) else 0, t_lin[-1] if len(t_lin) else 0, f_lin[0], f_lin[-1]],
                    vmin=S_db.max() - spec.top_db, vmax=S_db.max(), cmap=spec.cmap)
    ax.set_title("Linear-Frequency Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(fmin, fmax_eff)
    fig.colorbar(im0, ax=ax, pad=0.01).set_label("dB")

    # [8] Log-frequency spectrogram (dB) - same data, log-scaled y-axis
    ax = axes[2, 1]
    im1 = ax.imshow(S_db, aspect="auto", origin="lower",
                    extent=[t_lin[0] if len(t_lin) else 0, t_lin[-1] if len(t_lin) else 0, f_lin[0], f_lin[-1]],
                    vmin=S_db.max() - spec.top_db, vmax=S_db.max(), cmap=spec.cmap)
    ax.set_yscale("log")
    ax.set_title("Log-Frequency Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz, log scale)")
    ax.set_ylim(max(fmin, 1.0), fmax_eff)
    fig.colorbar(im1, ax=ax, pad=0.01).set_label("dB")

    # [9] Log-mel spectrogram (dB)
    ax = axes[2, 2]
    im2 = ax.imshow(S_mel_db, aspect="auto", origin="lower",
                    extent=[t_mel[0] if len(t_mel) else 0, t_mel[-1] if len(t_mel) else 0, 0, n_mels],
                    vmin=S_mel_db.max() - spec.top_db, vmax=S_mel_db.max(), cmap=spec.cmap)
    ax.set_title("Log-Mel Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Band Index")
    fig.colorbar(im2, ax=ax, pad=0.01).set_label("dB")

    # Metadata
    meta = {
        "encoding": "RAW",
        "format": "RAW PCM",
        "number_of_channel": str(audio.channels),
        "sample_rate": f"{audio.rate} Hz",
        "file_size": _human_bytes(len(raw.data)),
        "duration": f"{audio.duration:.3f} s ({_timecode(audio.duration)})",
        "window size": f"{n_fft} samples",
        "hop length": f"{hop} samples",
        "frequency scale": "linear | log | mel (see panels)",
        "frequency range": f"{fmin:.1f} Hz - {fmax_eff:.1f} Hz",
        "spectrogram amplitude range": f"{spec.top_db:.1f} dB dynamic",
    }
    _draw_metadata(fig, meta)
    fig.suptitle(title or "Raw PCM → Log-Mel Spectrogram: Step-by-Step", fontsize=14)

    out_path = os.path.join(common.output_dir, filename_suffix)
    _savefig(fig, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Demonstrations / Examples
# -----------------------------------------------------------------------------
def _example_sine_stereo(sr: int = 22050, seconds: float = 2.0) -> Audio:
    """
    Create a simple stereo tone for demonstrations: left 440 Hz, right 880 Hz.
    """
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    left = 0.6 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.6 * np.sin(2 * np.pi * 880.0 * t)
    x = np.stack([left, right], axis=1).astype(np.float32)
    return Audio(x, sr, fmt="RAW")


def _example_raw_from_audio(audio: Audio, sample_width: int = 2) -> RawAudio:
    """
    Convert an Audio object to RawAudio PCM for pipeline demonstration.
    """
    x = np.clip(audio.data, -1.0, 1.0)
    if sample_width == 2:
        pcm = (x * 32767.0).astype(np.int16)
        data = pcm.tobytes()
    elif sample_width == 1:
        pcm = ((x * 127.0) + 128.0).astype(np.uint8)
        data = pcm.tobytes()
    elif sample_width == 4:
        pcm = (x * (2 ** 31 - 1)).astype(np.int32)
        data = pcm.tobytes()
    else:
        # 24-bit pack
        s = np.clip((x * (2 ** 23 - 1)).astype(np.int32), -(2 ** 23), 2 ** 23 - 1)
        b0 = (s & 0xFF).astype(np.uint8)
        b1 = ((s >> 8) & 0xFF).astype(np.uint8)
        b2 = ((s >> 16) & 0xFF).astype(np.uint8)
        interleaved = np.stack([b0, b1, b2], axis=-1).reshape(-1)
        data = interleaved.tobytes()
        sample_width = 3
    return RawAudio(data=data, sample_rate=audio.rate, channels=audio.channels, sample_width=sample_width, endianness="little", signed=(sample_width != 1))


def _run_examples():
    common = CommonSettings(time_range=None, output_dir="plots")
    wave = WaveformSettings(visible=True, amplitude_range=(-1.0, 1.0))
    spec = SpectrogramSettings(visible=True, window_size=1024, hop_length=None, window="hann",
                               frequency_scale="mel", frequency_range=(30.0, 8000.0), top_db=80.0, cmap="magma")

    # Create synthetic audio for deterministic examples
    audio = _example_sine_stereo(sr=22050, seconds=2.0)

    # Method 1
    p1 = method1_plot_original_with_resampling(audio, target_rates=(16000, 8000, 11025), common=common, wave=wave,
                                               title="Original & Resampled (Stereo Tone)")
    print(f"[OK] Saved: {p1}")

    # Method 2
    p2 = method2_plot_hertz_to_mel(fmax_hz=audio.rate / 2.0, common=common)
    print(f"[OK] Saved: {p2}")

    # Method 3
    mel_max = hertz_to_mel(audio.rate / 2.0)
    p3 = method3_plot_mel_to_hertz(mel_max=mel_max, common=common)
    print(f"[OK] Saved: {p3}")

    # Method 4
    p4 = method4_plot_create_triangular_filter_bank(sr=audio.rate, n_fft=2048, n_mels=24, fmin=30.0, fmax=audio.rate/2.0, common=common)
    print(f"[OK] Saved: {p4}")

    # Method 5
    p5 = method5_plot_mel_filter_bank(sr=audio.rate, n_fft=2048, n_mels=64, fmin=30.0, fmax=audio.rate/2.0, common=common)
    print(f"[OK] Saved: {p5}")

    # Method 6
    p6 = method6_plot_log_mel_spectrogram(audio, spec=spec, common=common, n_mels=64, fmin=30.0, fmax=audio.rate/2.0,
                                          title="Log-Mel Spectrogram (Stereo Tone)")
    print(f"[OK] Saved: {p6}")

    # Method 7 (raw pipeline)
    raw = _example_raw_from_audio(audio, sample_width=2)
    p7 = method7_plot_raw_to_logmel_pipeline(raw, common=common, wave=wave, spec=spec, n_mels=64, fmin=30.0, fmax=audio.rate/2.0,
                                             title="Raw PCM → Log-Mel (Stereo Tone)")
    print(f"[OK] Saved: {p7}")


if __name__ == "__main__":
    _run_examples()