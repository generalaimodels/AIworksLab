#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIworksLab Audio Utils
Audio processing toolkit (from-scratch, single-file, technically rigorous).

This module provides:
- A robust Audio class that abstracts waveform, sample rate, format, with factory constructors for:
  from_file, from_bytes, from_base64, from_url, from_raw_audio, from_microphone
- Encoding support to base64 for multiple formats
- High-quality resampling (soxr if available, with scipy fallback)
- Standalone DSP utilities implementing the Slaney mel scale, mel filter-bank creation, and log-mel spectrograms
- Self-contained explanations in docstrings and comments; no external text needed to understand usage.

Design goals:
- Generalized: works across diverse formats, codecs, sources (URL/bytes/file/mic).
- Robust: multiple decoding backends (soundfile -> pydub/ffmpeg) with graceful fallbacks.
- Scalable: vectorized NumPy, safe defaults, and stable numerics for large workloads.
- Standards: clean code style, type hints, and deterministic behavior when possible.

Optional dependencies (auto-detected, graceful fallback):
- soundfile (libsndfile): decoding/encoding WAV/FLAC/OGG/etc.
- pydub + ffmpeg/avlib: decoding/encoding MP3/AAC/etc.
- soxr: high-quality resampling
- scipy: STFT and resample_poly fallback
- requests: HTTP downloads
- sounddevice (PortAudio): microphone capture

Everything is carefully documented inline to teach next-generation coders the reasoning and edge cases.


"""

from __future__ import annotations

import base64
import io
import math
import os
import struct
import subprocess
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Iterable

import numpy as np

# Optional imports (robust detection and lazy fallback)
try:
    import soundfile as sf
except Exception:
    sf = None  # Decoding may fallback to pydub/ffmpeg

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None  # Optional; requires ffmpeg/avlib present

try:
    import soxr as _soxr
except Exception:
    _soxr = None  # Fallback to scipy

try:
    from scipy import signal as sp_signal
except Exception:
    sp_signal = None  # Fallback will be numpy implementations where possible

try:
    import requests as _requests
except Exception:
    _requests = None  # Fallback to urllib

try:
    import sounddevice as sd
except Exception:
    sd = None  # Microphone capture becomes unavailable

# Numeric stability constant used across DSP
_EPS = np.finfo(np.float32).eps


# ===========================
# Utility: Slaney mel <-> Hz
# ===========================

def hertz_to_mel(f_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Hertz to Mel (Slaney formulation, used by e.g., Librosa with htk=False).

    Details:
    - Linear up to 1000 Hz; logarithmic thereafter.
    - Matches Slaney's Auditory Toolkit mapping: see librosa.core.mel.__doc__ for constants.

    Parameters
    ----------
    f_hz : float or np.ndarray
        Frequency in Hertz.

    Returns
    -------
    float or np.ndarray
        Frequency in Mel (Slaney scale).
    """
    f = np.asanyarray(f_hz, dtype=float)
    f_sp = 200.0 / 3.0          # Mel steps per Hz below 1000 Hz
    min_log_hz = 1000.0         # Linear-to-logarithmic cutoff in Hz
    min_log_mel = min_log_hz / f_sp  # 15.0 mels
    logstep = np.log(6.4) / 27.0

    m = np.empty_like(f)
    # Linear part
    lin = f < min_log_hz
    m[lin] = f[lin] / f_sp
    # Logarithmic part
    m[~lin] = min_log_mel + np.log(f[~lin] / min_log_hz) / logstep
    if np.isscalar(f_hz):
        return float(m)
    return m


def mel_to_hertz(m_mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert Mel to Hertz (Slaney formulation, inverse of hertz_to_mel).

    Parameters
    ----------
    m_mel : float or np.ndarray
        Frequency in Mel (Slaney).

    Returns
    -------
    float or np.ndarray
        Frequency in Hertz.
    """
    m = np.asanyarray(m_mel, dtype=float)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15 mels
    logstep = np.log(6.4) / 27.0

    f = np.empty_like(m)
    # Linear part
    lin = m < min_log_mel
    f[lin] = m[lin] * f_sp
    # Logarithmic part
    f[~lin] = min_log_hz * np.exp(logstep * (m[~lin] - min_log_mel))
    if np.isscalar(m_mel):
        return float(f)
    return f


# ==========================================
# Triangular Mel filter bank (Slaney-style)
# ==========================================

def _create_triangular_filter_bank(
    fft_freqs_hz: np.ndarray,
    mel_center_hz: np.ndarray,
    norm: Optional[str] = "slaney",
) -> np.ndarray:
    """
    Generate triangular filters from FFT bin frequencies and mel center frequencies (in Hz).

    The filters are constructed with edges at [center[i-1], center[i], center[i+1]] for each mel band i.

    Parameters
    ----------
    fft_freqs_hz : np.ndarray, shape (n_fft_bins,)
        Monotonic non-negative FFT bin center frequencies in Hertz for the positive spectrum (0..Nyquist).
    mel_center_hz : np.ndarray, shape (n_mels + 2,)
        Mel-spaced frequencies in Hertz including edges (2 extra: one below first filter center and one above last).
        Typically generated by linearly spacing mels between fmin and fmax, then converted back to Hertz.
    norm : Optional[str]
        If "slaney": equal-area normalization of filters (recommended).
        If None or "none": no normalization.

    Returns
    -------
    np.ndarray, shape (n_mels, n_fft_bins)
        Filter bank matrix. Multiplying by a power spectrum yields mel energies.
    """
    if mel_center_hz.ndim != 1 or fft_freqs_hz.ndim != 1:
        raise ValueError("fft_freqs_hz and mel_center_hz must be 1-D arrays.")
    if not np.all(np.diff(fft_freqs_hz) >= 0):
        raise ValueError("fft_freqs_hz must be non-decreasing.")
    if mel_center_hz.size < 3:
        raise ValueError("mel_center_hz must have at least 3 values (include edges).")

    # n_mels = len(centers) - 2 because we include left and right edges
    n_mels = mel_center_hz.size - 2
    n_fft_bins = fft_freqs_hz.size

    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)

    for m in range(n_mels):
        f_left = mel_center_hz[m]
        f_center = mel_center_hz[m + 1]
        f_right = mel_center_hz[m + 2]

        # Piecewise linear triangular filters
        left_slope = (fft_freqs_hz - f_left) / max(f_center - f_left, _EPS)
        right_slope = (f_right - fft_freqs_hz) / max(f_right - f_center, _EPS)
        w = np.maximum(0.0, np.minimum(left_slope, right_slope))
        fb[m, :] = w

    if norm and norm.lower() == "slaney":
        # Equal-area (Slaney) normalization across frequency axis (in Hz)
        # Each triangle scaled by 2 / (f_{m+2} - f_{m}) per Slaney-style
        enorm = 2.0 / (mel_center_hz[2:] - mel_center_hz[:-2] + _EPS)
        fb *= enorm[:, np.newaxis]

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
    Create a Mel filter bank matrix (Slaney mel scale), mapping power spectrum bins to mel bands.

    Parameters
    ----------
    sr : int
        Sampling rate.
    n_fft : int
        FFT size (number of samples per frame for FFT). The filter bank length will be n_fft//2 + 1.
    n_mels : int
        Number of mel bands.
    fmin : float
        Minimum frequency in Hertz for the mel filters (lower edge).
    fmax : Optional[float]
        Maximum frequency in Hertz (upper edge). Defaults to Nyquist (sr/2).
    norm : Optional[str]
        "slaney" (default) or None for no normalization.

    Returns
    -------
    np.ndarray, shape (n_mels, 1 + n_fft//2)
        Mel filter bank matrix.
    """
    if fmax is None:
        fmax = float(sr) / 2.0
    if not (0.0 <= fmin < fmax <= sr / 2 + 1e-6):
        raise ValueError("Require 0 <= fmin < fmax <= sr/2")

    # FFT bin center frequencies (0..Nyquist), linearly spaced in Hz
    fft_freqs = np.linspace(0.0, float(sr) / 2.0, n_fft // 2 + 1)

    # Mel grid in Slaney mels, including edges
    m_min = hertz_to_mel(fmin)
    m_max = hertz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, num=n_mels + 2, dtype=float)
    hz_pts = mel_to_hertz(m_pts)

    fb = _create_triangular_filter_bank(fft_freqs_hz=fft_freqs, mel_center_hz=hz_pts, norm=norm)
    return fb.astype(np.float32)


# ==================================
# Log-Mel Spectrogram (robust)
# ==================================

def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    top_db: Optional[float] = 80.0,
    mel_norm: Optional[str] = "slaney",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a log-scaled Mel spectrogram, optimized for robust feature extraction.

    Pipeline:
    1) Convert multi-channel to mono (mean)
    2) STFT -> magnitude or power spectrum
    3) Apply Mel filter bank (Slaney scale)
    4) Convert to decibels with numerical stability and optional dynamic range clipping

    Parameters
    ----------
    waveform : np.ndarray
        Audio samples; shapes accepted:
        - (num_samples,) mono
        - (num_samples, num_channels) multi-channel
    sample_rate : int
        Sampling rate of waveform.
    n_fft : int
        FFT size.
    hop_length : Optional[int]
        Hop size between successive frames. Defaults to n_fft//4 if None.
    win_length : Optional[int]
        Window length. Defaults to n_fft if None.
    window : str
        Window function name, passed to scipy.signal.get_window.
    center : bool
        If True, pads waveform so that frames are centered; else left-aligned frames.
    n_mels : int
        Number of mel bands.
    fmin : float
        Minimum frequency in Hz.
    fmax : Optional[float]
        Maximum frequency in Hz. Defaults to sr/2.
    power : float
        Exponent for magnitude spectrogram. 1.0 -> magnitude, 2.0 -> power (default).
    top_db : Optional[float]
        If not None, clip the dynamic range of the output in dB to this value below the peak.
    mel_norm : Optional[str]
        Mel filterbank normalization, "slaney" (default) or None.

    Returns
    -------
    S_db : np.ndarray, shape (n_mels, n_frames)
        Log-mel spectrogram in decibels.
    mel_fb : np.ndarray, shape (n_mels, 1 + n_fft//2)
        Mel filter bank matrix used.
    freqs_mel : np.ndarray, shape (n_mels,)
        Mel band center frequencies in Hertz (midpoint of each band).
    """
    if waveform.ndim == 2:
        # Convert to mono via average (robust to channel mismatch)
        x = waveform.astype(np.float32).mean(axis=1)
    elif waveform.ndim == 1:
        x = waveform.astype(np.float32)
    else:
        raise ValueError("waveform must be 1-D or 2-D (N) or (N, C)")

    hop = hop_length if hop_length is not None else int(n_fft // 4)
    win = win_length if win_length is not None else n_fft

    # Window and STFT using scipy if available, else manual numpy implementation
    if sp_signal is None:
        # Manual framing + FFT (centered padding if requested)
        pad = (n_fft // 2) if center else 0
        if pad > 0:
            x = np.pad(x, (pad, pad), mode="reflect")
        n_frames = 1 + (len(x) - win) // hop if len(x) >= win else 0
        if n_frames <= 0:
            return np.empty((n_mels, 0), np.float32), mel_filter_bank(sample_rate, n_fft, n_mels, fmin, fmax, mel_norm), np.linspace(fmin, (fmax or sample_rate/2), n_mels)

        # Window function
        if window.lower() == "hann":
            w = np.hanning(win).astype(np.float32)
        else:
            # Fallback window: rectangular
            w = np.ones(win, dtype=np.float32)

        # STFT magnitude
        spec = np.empty((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = x[start:start + win]
            if frame.shape[0] < win:
                frame = np.pad(frame, (0, win - frame.shape[0]))
            frame_win = frame * w
            fft = np.fft.rfft(frame_win, n=n_fft)
            mag = np.abs(fft).astype(np.float32)
            spec[:, i] = mag
    else:
        # scipy.signal.stft handles centering via "boundary"
        window_arr = sp_signal.get_window(window, win, fftbins=True)
        boundary = "even" if center else None
        padded = "zeros"  # internal detail for stft; keep default
        f, t, Zxx = sp_signal.stft(
            x,
            fs=sample_rate,
            nperseg=win,
            noverlap=win - hop,
            nfft=n_fft,
            window=window_arr,
            boundary=boundary,
            padded=True,
            return_onesided=True,
        )
        spec = np.abs(Zxx).astype(np.float32)

    if power is not None and abs(power - 2.0) < 1e-12:
        S = spec ** 2
    elif power is not None and abs(power - 1.0) < 1e-12:
        S = spec
    else:
        S = spec ** float(power)

    fb = mel_filter_bank(sample_rate, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=mel_norm)

    # Mel energies
    M = fb @ S  # (n_mels, n_frames)

    # Numerical stability
    M = np.maximum(M, _EPS).astype(np.float32)

    # Convert to decibel scale
    S_db = 10.0 * np.log10(M)
    if top_db is not None:
        max_val = np.max(S_db)
        S_db = np.maximum(S_db, max_val - float(top_db))

    # Provide mel band center frequencies in Hz (midpoints in mel-space -> convert to Hz)
    m_edges = np.linspace(hertz_to_mel(fmin), hertz_to_mel((fmax if fmax else sample_rate / 2.0)), num=n_mels + 2)
    m_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    mel_centers_hz = mel_to_hertz(m_centers)

    return S_db.astype(np.float32), fb, mel_centers_hz.astype(np.float32)


# ==================================
# Raw audio protocol (for from_raw)
# ==================================

@dataclass
class RawAudio:
    """
    Raw PCM audio payload definition.

    Fields
    ------
    data : bytes
        Raw PCM interleaved samples.
    sample_rate : int
        Sampling rate in Hertz.
    channels : int
        Number of channels (1=mono, 2=stereo, etc.).
    sample_width : int
        Bytes per sample per channel (1, 2, 3, 4 for 8/16/24/32-bit).
    endianness : str
        "little" or "big" (endianness for widths > 1).
    signed : bool
        Whether PCM is signed (True for 16/24/32-bit PCM, False typically for 8-bit PCM).
    """
    data: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2
    endianness: str = "little"
    signed: bool = True


# ==================================
# Backend helpers (I/O and codecs)
# ==================================

def _require(module, name: str) -> None:
    """Raise an informative error if module is None."""
    if module is None:
        raise RuntimeError(f"{name} is required for this operation but was not found. "
                           f"Install it with `pip install {name}`.")


def _np_float32_pcm(x: np.ndarray, sample_width: int, signed: bool) -> np.ndarray:
    """
    Normalize integer PCM to float32 in [-1, 1], handling sample width.
    """
    if x.dtype.kind == "f":
        return x.astype(np.float32, copy=False)

    if sample_width == 1:
        # 8-bit PCM: unsigned usually. Map [0,255] -> [-1,1]
        if signed:
            max_val = 2 ** 7 - 1
            x = x.astype(np.float32) / max(max_val, 1)
        else:
            x = x.astype(np.float32)
            x = (x - 128.0) / 128.0
    elif sample_width == 2:
        max_val = 2 ** 15
        x = x.astype(np.float32) / max_val
    elif sample_width == 3:
        # 24-bit packed; if given as int32 with 24-bit significant
        max_val = 2 ** 23
        x = x.astype(np.float32) / max_val
    elif sample_width == 4:
        max_val = 2 ** 31
        x = x.astype(np.float32) / max_val
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    return x.astype(np.float32)


def _bytes_to_ndarray(
    data: bytes,
    sample_width: int,
    channels: int,
    endianness: str,
    signed: bool,
) -> np.ndarray:
    """Interpret raw PCM bytes into a (num_samples, channels) float32 array in [-1, 1]."""
    if sample_width not in (1, 2, 3, 4):
        raise ValueError("sample_width must be 1,2,3,4 bytes.")
    if channels <= 0:
        raise ValueError("channels must be positive.")

    # Determine NumPy dtype
    if sample_width == 1:
        dtype = np.uint8 if not signed else np.int8
    elif sample_width == 2:
        dtype = np.dtype("<i2" if endianness == "little" else ">i2")
    elif sample_width == 3:
        # No native 24-bit dtype; read as bytes then convert
        byte_order = "<" if endianness == "little" else ">"
        # Convert 24-bit to 32-bit with sign extension
        a = np.frombuffer(data, dtype=np.uint8)
        if a.size % (3 * channels) != 0:
            raise ValueError("Raw 24-bit data size is not divisible by 3*channels.")
        a = a.reshape(-1, 3 * channels)
        # Interleave to per-channel samples (3 bytes per sample)
        a = a.reshape(-1, channels, 3)
        # Assemble 24-bit little/big endian into int32 with sign
        if endianness == "little":
            out = (a[:, :, 0].astype(np.uint32)
                   | (a[:, :, 1].astype(np.uint32) << 8)
                   | (a[:, :, 2].astype(np.uint32) << 16))
        else:
            out = ((a[:, :, 2].astype(np.uint32))
                   | (a[:, :, 1].astype(np.uint32) << 8)
                   | (a[:, :, 0].astype(np.uint32) << 16))
        # Sign extension for 24-bit signed
        if signed:
            sign_mask = 1 << 23
            out = out.astype(np.int32)
            out = (out ^ sign_mask) - sign_mask
        else:
            out = out.astype(np.int32)
        x = out.astype(np.float32) / float(2 ** 23)
        return x
    else:  # sample_width == 4
        dtype = np.dtype("<i4" if signed and endianness == "little" else
                         ">i4" if signed and endianness == "big" else
                         "<u4" if endianness == "little" else ">u4")

    x = np.frombuffer(data, dtype=dtype)
    if x.size % channels != 0:
        raise ValueError("Byte length is not divisible by the number of channels.")
    x = x.reshape(-1, channels)
    x = _np_float32_pcm(x, sample_width=sample_width, signed=signed)
    return x


def _decode_with_soundfile(data: bytes) -> Tuple[np.ndarray, int, Optional[str]]:
    """Try decoding bytes with soundfile (libsndfile)."""
    _require(sf, "soundfile")
    with sf.SoundFile(io.BytesIO(data)) as f:
        sr = int(f.samplerate)
        ch = int(f.channels)
        fmt = f.format  # Container format, e.g., 'WAV', 'FLAC', etc.
    # Read data in float32
    y, sr2 = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return y, int(sr2), fmt


def _decode_with_pydub(data: bytes, file_hint: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[str]]:
    """Try decoding using pydub + ffmpeg/avlib."""
    _require(AudioSegment, "pydub")
    # Pydub can auto-detect format if header is present; hint may help for ambiguous cases
    seg = AudioSegment.from_file(io.BytesIO(data), format=file_hint)
    sr = seg.frame_rate
    ch = seg.channels
    sw = seg.sample_width  # bytes per sample
    arr = np.array(seg.get_array_of_samples())
    if ch > 1:
        arr = arr.reshape(-1, ch)
    else:
        arr = arr.reshape(-1, 1)
    y = _np_float32_pcm(arr, sample_width=sw, signed=True if sw > 1 else False)
    return y, int(sr), file_hint


def _decode_with_ffmpeg(data: bytes, file_hint: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[str]]:
    """
    Decode via ffmpeg CLU: pipe bytes to stdin, get WAV PCM float32 on stdout, then parse.

    Note: Requires ffmpeg in PATH. We request float32 wav for robust pipeline.
    """
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-f", "mp3" if file_hint == "mp3" else "wav" if file_hint == "wav" else "matroska" if file_hint == "mka" else "-",
        "-i", "pipe:0",
        "-f", "wav",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise RuntimeError("ffmpeg decoding failed.") from e

    # Parse WAV from bytes using soundfile if available, else wave module
    if sf is not None:
        y, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=True)
        return y, int(sr), "WAV"
    else:
        import wave
        with wave.open(io.BytesIO(proc.stdout), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            frames = w.getnframes()
            pcm = w.readframes(frames)
        # Now pcm is float32 little-endian from ffmpeg
        y = np.frombuffer(pcm, dtype="<f4").reshape(-1, ch).astype(np.float32)
        return y, int(sr), "WAV"


def _guess_format_from_path(path: str) -> Optional[str]:
    """Infer format from file extension."""
    _, ext = os.path.splitext(path)
    return ext.lower().lstrip(".") or None


def _load_audio_bytes(data: bytes, file_hint: Optional[str] = None) -> Tuple[np.ndarray, int, Optional[str]]:
    """
    Robust decoding pipeline from bytes -> float32 numpy (N, C), sample rate, format string.

    Tries: soundfile -> pydub/ffmpeg -> ffmpeg subprocess.
    """
    # Try soundfile first
    if sf is not None:
        try:
            return _decode_with_soundfile(data)
        except Exception:
            pass

    # Try pydub
    if AudioSegment is not None:
        try:
            return _decode_with_pydub(data, file_hint=file_hint)
        except Exception:
            pass

    # Try ffmpeg CLI fallback
    try:
        return _decode_with_ffmpeg(data, file_hint=file_hint)
    except Exception as e:
        raise RuntimeError("Failed to decode audio bytes with available backends.") from e


def _encode_with_soundfile(
    y: np.ndarray,
    sr: int,
    fmt: str,
    subtype: Optional[str] = None,
) -> bytes:
    """
    Encode numpy audio to container using soundfile (libsndfile).
    """
    _require(sf, "soundfile")
    buf = io.BytesIO()
    # soundfile expects shape (frames, channels)
    if y.ndim == 1:
        y2 = y[:, None]
    else:
        y2 = y
    sf.write(buf, y2, sr, format=fmt.upper(), subtype=subtype or "PCM_16")
    return buf.getvalue()


def _encode_with_pydub(
    y: np.ndarray,
    sr: int,
    fmt: str,
    bitrate: Optional[str] = None,
) -> bytes:
    """
    Encode numpy audio using pydub (requires ffmpeg/avlib).
    """
    _require(AudioSegment, "pydub")
    # pydub expects 16-bit or raw; we convert to 16-bit PCM
    y16 = np.clip((y * 32767.0), -32768, 32767).astype(np.int16)
    ch = 1 if y16.ndim == 1 else y16.shape[1]
    if ch == 1:
        samples = y16.flatten()
    else:
        samples = y16.reshape(-1)
    seg = AudioSegment(
        samples.tobytes(),
        frame_rate=int(sr),
        sample_width=2,
        channels=int(ch),
    )
    buf = io.BytesIO()
    export_kwargs = {}
    if bitrate:
        export_kwargs["bitrate"] = bitrate
    seg.export(buf, format=fmt, **export_kwargs)
    return buf.getvalue()


# ==================================
# High-quality resampling
# ==================================

def _resample_soxr(y: np.ndarray, sr_in: int, sr_out: int, quality: str = "HQ") -> np.ndarray:
    """
    Resample using soxr if available. Accepts (N,C) or (N,).
    """
    _require(_soxr, "soxr")
    if y.ndim == 1:
        return _soxr.resample(y.astype(np.float32), sr_in, sr_out, quality=quality).astype(np.float32)
    else:
        # soxr can operate channel-wise; we map over columns
        outs = []
        for c in range(y.shape[1]):
            outs.append(_soxr.resample(y[:, c].astype(np.float32), sr_in, sr_out, quality=quality))
        y_out = np.stack(outs, axis=1).astype(np.float32)
        return y_out


def _resample_scipy(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """
    Resample using scipy.signal.resample_poly as a high-quality fallback.
    """
    _require(sp_signal, "scipy")
    # Rational approximation of rate ratio
    from math import gcd
    g = gcd(sr_out, sr_in)
    up = sr_out // g
    down = sr_in // g
    axis = 0
    y_out = sp_signal.resample_poly(y.astype(np.float32), up, down, axis=axis)
    return y_out.astype(np.float32)


# ==================================
# Audio class
# ==================================

class Audio:
    """
    Immutable audio container with robust I/O and processing utilities.

    Internal representation:
    - data: np.ndarray float32 in range [-1, 1], shape (num_samples, channels)
            (mono audio has channels = 1)
    - rate: int sample rate
    - format: Optional[str] original or chosen encoding container ("WAV", "FLAC", "mp3", etc.)

    Notes:
    - All classmethods return Audio instances with float32 multichannel arrays.
    - Use resample() to change sample rate with high-quality interpolation.
    - Use to_base64() to serialize; supports soundfile and pydub backends.

    Examples
    --------
    See the __main__ block at bottom for runnable demonstrations.
    """

    def __init__(self, data: np.ndarray, rate: int, fmt: Optional[str] = None):
        """
        Construct an Audio object from waveform, sample rate, and optional format spec.

        Parameters
        ----------
        data : np.ndarray
            Either shape (N,) mono or (N, C) multichannel. Any numeric dtype will be coerced to float32.
        rate : int
            Sampling rate in Hertz.
        fmt : Optional[str]
            Container/format hint (e.g., "WAV", "FLAC", "mp3"). Used by encoders; can be None.
        """
        if rate <= 0:
            raise ValueError("Sample rate must be positive.")
        x = np.asanyarray(data)
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim != 2:
            raise ValueError("data must be 1-D or 2-D (N) or (N, C).")
        # Cast to float32 and clamp to [-1, 1]
        x = x.astype(np.float32)
        # No hard clipping unless critical; here we soft-clip to preserve invariants
        x = np.clip(x, -1.0, 1.0, out=x)

        self._data = x
        self._rate = int(rate)
        self._format = fmt.upper() if isinstance(fmt, str) else fmt
        self._check_valid()

    def __repr__(self) -> str:
        n, c = self._data.shape
        dur = n / float(self._rate)
        fmt = self._format or "Unknown"
        return f"Audio(rate={self._rate}, duration={dur:.3f}s, shape=({n}, {c}), format={fmt})"

    # -----------------------
    # Validation and helpers
    # -----------------------

    def _check_valid(self) -> None:
        """
        Validate data type, shape, and format compatibility.
        """
        x = self._data
        if x.ndim != 2:
            raise ValueError("Internal data must be 2-D (N, C).")
        if x.dtype != np.float32:
            raise TypeError("Internal dtype must be float32.")
        if not np.isfinite(x).all():
            raise ValueError("Data contains NaNs or infinities.")
        if not (-1.0001 <= x.min() <= 1.0001 and -1.0001 <= x.max() <= 1.0001):
            warnings.warn("Data outside [-1,1] range; consider normalization.", RuntimeWarning)
        if self._rate <= 0:
            raise ValueError("Invalid sample rate.")
        # No strict checks on format; encoding paths re-validate as needed.

    # -----------------------
    # Properties
    # -----------------------

    @property
    def data(self) -> np.ndarray:
        """Return audio samples as (num_samples, channels) float32 in [-1, 1]."""
        return self._data

    @property
    def rate(self) -> int:
        """Return sample rate."""
        return self._rate

    @property
    def format(self) -> Optional[str]:
        """Return format/container hint."""
        return self._format

    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        return self._data.shape[0] / float(self._rate)

    @property
    def channels(self) -> int:
        """Number of channels."""
        return self._data.shape[1]

    # -----------------------
    # Factory constructors
    # -----------------------

    @classmethod
    def from_url(cls, url: str, timeout: float = 20.0) -> "Audio":
        """
        Download audio from a URL and construct an Audio object.

        Parameters
        ----------
        url : str
            HTTP(S) URL pointing to an audio file.
        timeout : float
            Network timeout in seconds.

        Returns
        -------
        Audio
        """
        if _requests is not None:
            resp = _requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
        else:
            # Fallback to urllib
            import urllib.request
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
        fmt_hint = _guess_format_from_path(url)
        y, sr, fmt = _load_audio_bytes(data, file_hint=fmt_hint)
        return cls(y, sr, fmt if fmt else (fmt_hint.upper() if fmt_hint else None))

    @classmethod
    def from_base64(cls, b64: str, format_hint: Optional[str] = None) -> "Audio":
        """
        Decode base64 audio string into an Audio object.

        Parameters
        ----------
        b64 : str
            Base64 encoded audio. Data-URI prefixes are supported (e.g., 'data:audio/wav;base64,...').
        format_hint : Optional[str]
            Optional container hint for decoding ambiguous payloads.

        Returns
        -------
        Audio
        """
        if "," in b64 and b64.strip().startswith("data:"):
            # Strip data URI prefix
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        y, sr, fmt = _load_audio_bytes(data, file_hint=(format_hint.lower() if format_hint else None))
        return cls(y, sr, fmt if fmt else (format_hint.upper() if format_hint else None))

    @classmethod
    def from_file(cls, path: str) -> "Audio":
        """
        Read audio from a local file path.

        Parameters
        ----------
        path : str
            Local filesystem path.

        Returns
        -------
        Audio
        """
        with open(path, "rb") as f:
            data = f.read()
        fmt_hint = _guess_format_from_path(path)
        y, sr, fmt = _load_audio_bytes(data, file_hint=fmt_hint)
        return cls(y, sr, fmt if fmt else (fmt_hint.upper() if fmt_hint else None))

    @classmethod
    def from_bytes(cls, data: bytes, format_hint: Optional[str] = None) -> "Audio":
        """
        Construct Audio from raw container bytes (not PCM) using decoding backends.

        Parameters
        ----------
        data : bytes
            Encoded audio container.
        format_hint : Optional[str]
            Optional container hint (e.g., 'mp3', 'wav').

        Returns
        -------
        Audio
        """
        y, sr, fmt = _load_audio_bytes(data, file_hint=(format_hint.lower() if format_hint else None))
        return cls(y, sr, fmt if fmt else (format_hint.upper() if format_hint else None))

    @classmethod
    def from_raw_audio(
        cls,
        data: Union[bytes, RawAudio],
        sample_rate: Optional[int] = None,
        channels: int = 1,
        sample_width: int = 2,
        endianness: str = "little",
        signed: bool = True,
    ) -> "Audio":
        """
        Construct from raw PCM protocol.

        Two usages:
        - Provide a RawAudio(data=..., sample_rate=..., channels=..., sample_width=..., endianness=..., signed=...)
        - Provide raw args directly: data=bytes, sample_rate=..., channels=..., sample_width=...

        Parameters
        ----------
        data : bytes or RawAudio
            PCM payload (interleaved) or RawAudio descriptor.
        sample_rate : Optional[int]
            Required if 'data' is bytes. Ignored if 'data' is RawAudio.
        channels : int
            Number of channels.
        sample_width : int
            Bytes per sample (1/2/3/4).
        endianness : str
            'little' or 'big' for widths > 1.
        signed : bool
            Signedness for PCM (True recommended for >=16-bit).

        Returns
        -------
        Audio
        """
        if isinstance(data, RawAudio):
            ra = data
            y = _bytes_to_ndarray(ra.data, ra.sample_width, ra.channels, ra.endianness, ra.signed)
            return cls(y, ra.sample_rate, fmt="RAW")
        else:
            if sample_rate is None:
                raise ValueError("sample_rate must be provided when 'data' is raw bytes.")
            y = _bytes_to_ndarray(data, sample_width, channels, endianness, signed)
            return cls(y, int(sample_rate), fmt="RAW")

    @classmethod
    def from_microphone(
        cls,
        duration_sec: float = 3.0,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[Union[int, str]] = None,
    ) -> "Audio":
        """
        Capture live audio from system microphone (blocking) and return an Audio instance.

        Requires 'sounddevice' (PortAudio). If not available, an informative error is raised.

        Parameters
        ----------
        duration_sec : float
            Duration in seconds to record.
        sample_rate : int
            Target sample rate.
        channels : int
            Number of channels to record.
        device : Optional[Union[int, str]]
            Device index or name (see sounddevice.query_devices()).

        Returns
        -------
        Audio
        """
        _require(sd, "sounddevice")
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive.")
        frames = int(round(duration_sec * sample_rate))
        rec = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32", device=device)
        sd.wait()  # blocking
        # sounddevice returns shape (frames, channels)
        return cls(rec.astype(np.float32), sample_rate, fmt="RAW")

    # -----------------------
    # Serialization
    # -----------------------

    def to_base64(
        self,
        fmt: str = "wav",
        subtype: Optional[str] = None,
        bitrate: Optional[str] = None,
        prepend_data_uri: bool = False,
    ) -> str:
        """
        Encode audio to base64 string.

        Parameters
        ----------
        fmt : str
            Target container format, e.g., 'wav', 'flac', 'mp3', 'ogg'.
        subtype : Optional[str]
            For soundfile-supported encoders (e.g., 'PCM_16', 'FLOAT').
        bitrate : Optional[str]
            For pydub/ffmpeg encoders (e.g., '192k' for mp3).
        prepend_data_uri : bool
            If True, prepend a 'data:audio/<fmt>;base64,' URI header.

        Returns
        -------
        str
            Base64-encoded audio payload (optionally a full Data-URI).
        """
        y = self._data
        sr = self._rate
        fmt_l = fmt.lower()

        enc_data: Optional[bytes] = None
        # Try soundfile if it supports format
        if sf is not None:
            try:
                enc_data = _encode_with_soundfile(y, sr, fmt=fmt_l, subtype=subtype)
            except Exception:
                enc_data = None

        # Fallback to pydub/ffmpeg if needed
        if enc_data is None and AudioSegment is not None:
            try:
                enc_data = _encode_with_pydub(y, sr, fmt=fmt_l, bitrate=bitrate)
            except Exception:
                enc_data = None

        if enc_data is None:
            raise RuntimeError(f"Unable to encode to format '{fmt}': no suitable backend available.")

        b64 = base64.b64encode(enc_data).decode("ascii")
        if prepend_data_uri:
            return f"data:audio/{fmt_l};base64,{b64}"
        return b64

    # -----------------------
    # Processing
    # -----------------------

    def resample(self, new_rate: int, quality: str = "HQ") -> "Audio":
        """
        Resample the audio to a new sample rate using soxr (if available) or scipy fallback.

        Parameters
        ----------
        new_rate : int
            Target sampling rate.
        quality : str
            soxr quality preset (e.g., 'Q', 'LQ', 'MQ', 'HQ', 'VHQ'). Ignored by scipy fallback.

        Returns
        -------
        Audio
            New Audio with resampled data.
        """
        if new_rate <= 0:
            raise ValueError("new_rate must be positive.")
        if new_rate == self._rate:
            return Audio(self._data.copy(), self._rate, self._format)

        y = self._data
        if _soxr is not None:
            y_out = _resample_soxr(y, self._rate, new_rate, quality=quality)
        elif sp_signal is not None:
            y_out = _resample_scipy(y, self._rate, new_rate)
        else:
            raise RuntimeError("No resampler available. Install 'soxr' or 'scipy'.")

        return Audio(y_out, new_rate, self._format)


# ==================================
# Demonstrations and tests
# ==================================

if __name__ == "__main__":
    # Demonstrations are self-contained and safe to run in any environment.

    # 1) Synthetic signal (sine + stereo)
    sr = 22050
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    x_left = 0.5 * np.sin(2 * np.pi * 440.0 * t)            # A4
    x_right = 0.5 * np.sin(2 * np.pi * (440.0 * 2) * t)     # A5
    stereo = np.stack([x_left, x_right], axis=1).astype(np.float32)

    audio = Audio(stereo, sr, fmt="RAW")
    print("Constructed:", audio)

    # 2) Duration and channels
    print("Duration (s):", audio.duration)
    print("Channels:", audio.channels)

    # 3) Resample to 16k (soxr if present)
    try:
        audio_16k = audio.resample(16000)
        print("Resampled:", audio_16k)
    except Exception as e:
        print("Resample skipped (missing deps):", e)
        audio_16k = audio  # fallback for subsequent examples

    # 4) Serialize to base64 WAV (always available with soundfile or pydub; else error)
    try:
        b64_wav = audio_16k.to_base64(fmt="wav", subtype="PCM_16", prepend_data_uri=False)
        print("Base64 WAV length:", len(b64_wav))
    except Exception as e:
        print("WAV encoding skipped:", e)

    # 5) Serialize to base64 MP3 (requires pydub + ffmpeg)
    try:
        b64_mp3 = audio_16k.to_base64(fmt="mp3", bitrate="192k", prepend_data_uri=False)
        print("Base64 MP3 length:", len(b64_mp3))
    except Exception as e:
        print("MP3 encoding skipped:", e)

    # 6) Log-mel spectrogram (Slaney) with robust defaults
    S_db, mel_fb, mel_centers_hz = log_mel_spectrogram(
        waveform=audio_16k.data,
        sample_rate=audio_16k.rate,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        fmin=30.0,
        fmax=audio_16k.rate / 2.0,
        power=2.0,
        top_db=80.0,
    )
    print("Log-mel spectrogram shape:", S_db.shape)
    print("Mel filter bank shape:", mel_fb.shape)
    print("First 5 mel center freqs (Hz):", mel_centers_hz[:5])

    # 7) Bytes -> Audio via raw protocol (simulate 16-bit PCM)
    # Pack mono 16-bit PCM from sine wave
    mono = x_left.astype(np.float32)
    pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    raw = RawAudio(data=pcm16, sample_rate=sr, channels=1, sample_width=2, endianness="little", signed=True)
    audio_raw = Audio.from_raw_audio(raw)
    print("Raw->Audio:", audio_raw)

    # 8) Base64 round-trip using WAV container
    try:
        b64 = audio_raw.to_base64(fmt="wav", subtype="PCM_16", prepend_data_uri=True)
        audio_rt = Audio.from_base64(b64)
        print("Round-trip:", audio_rt)
    except Exception as e:
        print("Round-trip skipped:", e)

    # 9) Decode from bytes (WAV) using soundfile, then compute log-mel
    try:
        wav_bytes = _encode_with_soundfile(audio.data, audio.rate, fmt="wav", subtype="PCM_16")
        audio_b = Audio.from_bytes(wav_bytes, format_hint="wav")
        S_db_b, _, _ = log_mel_spectrogram(audio_b.data, audio_b.rate, n_fft=1024, hop_length=256, n_mels=64)
        print("Decoded-from-bytes log-mel shape:", S_db_b.shape)
    except Exception as e:
        print("Decode-from-bytes skipped:", e)

    # 10) URL download example (commented; requires external connectivity)
    # url = "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"
    # audio_url = Audio.from_url(url)
    # print("From URL:", audio_url)

    print("All demos executed. This module is ready for robust audio pipelines")