#!/usr/bin/env python3
# hinglish_deepfake_detector.py
# Real-time style Hinglish audio deepfake detector using signal-level cues.
# No external ML deps (numpy only). Processes a WAV file offline, simulating streaming.
#
# Usage:
#   python hinglish_deepfake_detector.py input.wav
# Optional flags:
#   --sr 16000           # expected sample rate
#   --frame_ms 25        # frame length in ms
#   --hop_ms 10          # hop length in ms
#   --mel_bands 40       # number of mel bands
#   --roll_sec 2.0       # rolling window seconds for scoring
#   --print_windows 1    # print per-window scores
#
# Output: prints per-window scores and an overall decision.
#
# Notes:
# - This is a lightweight baseline that uses spectral entropy, spectral flux,
#   simple pitch/voicing stability, and VAD-gated statistics.
# - Calibrate thresholds/weights for your deployment audio path.
#
# Author: ChatGPT (Python reference implementation)

import argparse
import numpy as np
import wave
import struct
import math
from collections import deque

EPS = 1e-10

def read_wav_mono(path, expected_sr=16000):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Only 16-bit PCM WAV is supported.")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    if framerate != expected_sr:
        raise ValueError(f"Sample rate {framerate} != expected {expected_sr}. Resample externally or pass --sr.")
    return data, framerate

def framing(x, sr, frame_ms=25.0, hop_ms=10.0):
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len   = int(round(sr * hop_ms / 1000.0))
    if len(x) < frame_len:
        return np.zeros((0, frame_len), dtype=np.float32), frame_len, hop_len
    n_frames = 1 + (len(x) - frame_len) // hop_len
    frames = np.stack([x[i*hop_len : i*hop_len + frame_len] for i in range(n_frames)], axis=0)
    return frames, frame_len, hop_len

def hamming_window(n):
    return 0.54 - 0.46 * np.cos(2*np.pi*np.arange(n)/(n-1))

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f/700.0)

def mel_to_hz(m):
    return 700.0 * (10**(m/2595.0) - 1.0)

def mel_filterbank(sr, nfft, n_mels=40, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr/2.0
    mmin = hz_to_mel(fmin)
    mmax = hz_to_mel(fmax)
    mpts = np.linspace(mmin, mmax, n_mels+2)
    fpts = mel_to_hz(mpts)
    bins = np.floor((nfft+1) * fpts / sr).astype(int)
    fb = np.zeros((n_mels, nfft//2+1), dtype=np.float32)
    for m in range(1, n_mels+1):
        f_m_left = bins[m-1]
        f_m = bins[m]
        f_m_right = bins[m+1]
        if f_m == f_m_left: f_m += 1
        if f_m_right == f_m: f_m_right += 1
        for k in range(f_m_left, f_m):
            if 0 <= k < fb.shape[1]:
                fb[m-1, k] = (k - f_m_left) / float(max(1, f_m - f_m_left))
        for k in range(f_m, f_m_right):
            if 0 <= k < fb.shape[1]:
                fb[m-1, k] = (f_m_right - k) / float(max(1, f_m_right - f_m))
    # Normalize filters (slaney-style energy preservation)
    enorm = 2.0 / (fpts[2:n_mels+2] - fpts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb

def stft_mag(frames, nfft):
    win = hamming_window(frames.shape[1]).astype(np.float32)
    W = frames * win[None, :]
    spec = np.fft.rfft(W, n=nfft, axis=1)
    mag = np.abs(spec).astype(np.float32)
    return mag  # shape: [T, nfft//2+1]

def log_mel_spectrogram(frames, sr, nfft=512, n_mels=40):
    mag = stft_mag(frames, nfft)
    fb = mel_filterbank(sr, nfft, n_mels=n_mels)
    mel = np.maximum(EPS, mag @ fb.T)  # [T, n_mels]
    logmel = np.log(mel).astype(np.float32)
    return logmel, mag

def spectral_entropy(mag):
    # mag: [freq_bins] magnitude spectrum
    p = mag + EPS
    p = p / np.sum(p)
    H = -np.sum(p * np.log(p + EPS))
    Hn = H / np.log(len(mag) + EPS)  # normalized 0..1
    return float(Hn)

def spectral_flux(prev_mag, curr_mag):
    if prev_mag is None:
        return 0.0
    diff = curr_mag - prev_mag
    diff[diff < 0] = 0.0
    return float(np.sum(diff) / (np.sum(curr_mag) + EPS))

def zero_crossing_rate(frame):
    x = frame
    return float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)

def frame_energy_dbfs(frame):
    rms = np.sqrt(np.mean(frame**2) + EPS)
    # reference is 1.0 full scale
    return 20.0 * np.log10(rms + EPS)

def simple_vad(frame, energy_thresh_dbfs=-45.0, zcr_max=0.18):
    e = frame_energy_dbfs(frame)
    z = zero_crossing_rate(frame)
    return (e > energy_thresh_dbfs) and (z < zcr_max)

def autocorr_pitch(frame, sr, fmin=70.0, fmax=350.0):
    x = frame - np.mean(frame)
    if np.max(np.abs(x)) < 1e-4:
        return 0.0, 0.0  # unvoiced
    # autocorr using FFT conv trick would be faster; direct is fine for 25 ms
    r = np.correlate(x, x, mode='full')[len(x)-1:]
    r /= (np.max(r) + EPS)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    if max_lag >= len(r): max_lag = len(r) - 1
    if min_lag < 1: min_lag = 1
    seg = r[min_lag:max_lag]
    if len(seg) == 0:
        return 0.0, 0.0
    idx = np.argmax(seg)
    peak = seg[idx]
    lag = min_lag + idx
    if peak < 0.3:  # voicing threshold
        return 0.0, peak
    f0 = float(sr / lag)
    return f0, peak

def rolling_stats(window_vals):
    arr = np.array(window_vals, dtype=np.float32)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return mean, std, float(np.median(arr))

def detect_from_wav(path, args):
    x, sr = read_wav_mono(path, expected_sr=args.sr)
    frames, frame_len, hop_len = framing(x, sr, args.frame_ms, args.hop_ms)
    if frames.shape[0] == 0:
        print("Audio too short.")
        return

    nfft = 1
    while nfft < frame_len: nfft <<= 1
    if nfft < 512: nfft = 512

    logmel, mag_spec = log_mel_spectrogram(frames, sr, nfft=nfft, n_mels=args.mel_bands)

    T = frames.shape[0]
    ent = np.zeros(T, dtype=np.float32)
    flux = np.zeros(T, dtype=np.float32)
    f0 = np.zeros(T, dtype=np.float32)
    voiced = np.zeros(T, dtype=np.float32)
    prev_mag = None

    for t in range(T):
        e = spectral_entropy(mag_spec[t])
        ent[t] = e
        fl = spectral_flux(prev_mag, mag_spec[t])
        flux[t] = fl
        prev_mag = mag_spec[t].copy()
        v = simple_vad(frames[t], energy_thresh_dbfs=args.energy_vad_dbfs, zcr_max=args.zcr_vad_max)
        voiced[t] = 1.0 if v else 0.0
        if v:
            pitch, pk = autocorr_pitch(frames[t], sr, args.min_f0, args.max_f0)
            f0[t] = pitch
        else:
            f0[t] = 0.0

    # Rolling window scoring
    roll_len = max(1, int(round(args.roll_sec * 1000.0 / args.hop_ms)))
    state = "bonafide"
    suspicious_windows = 0
    total_windows = 0
    scores = []

    for start in range(0, T, roll_len):
        end = min(T, start + roll_len)
        if end - start < max(10, roll_len//4):
            break
        e_win = ent[start:end]
        f_win = flux[start:end]
        v_win = voiced[start:end]
        f0_win = f0[start:end]

        # Entropy spikes: z-score > 2.5
        e_mean = np.mean(e_win); e_std = np.std(e_win) + EPS
        e_z = (e_win - e_mean) / e_std
        ent_spikes = float(np.sum(e_z > 2.5)) / len(e_win)

        # Entropy std
        ent_std = float(np.std(e_win))

        # Flux high fraction (robust)
        fl_med = np.median(f_win)
        fl_mad = np.median(np.abs(f_win - fl_med)) + EPS
        fl_thr = fl_med + 3.0 * 1.4826 * fl_mad
        flux_hi_frac = float(np.mean(f_win > fl_thr))

        # Pitch delta variance on voiced frames
        f0_v = f0_win[f0_win > 0.0]
        if len(f0_v) >= 3:
            df0 = np.diff(f0_v)
            df0_var = float(np.var(df0))
        else:
            df0_var = 0.0

        voiced_frac = float(np.mean(v_win > 0.5))

        score = (
            args.w_ent_std * ent_std +
            args.w_ent_spikes * ent_spikes +
            args.w_flux_frac * flux_hi_frac +
            args.w_df0_var * df0_var -
            args.w_voiced_frac * voiced_frac
        )
        scores.append(score)
        total_windows += 1

        label = "suspicious" if score > args.score_thresh else "bonafide"
        if label == "suspicious":
            suspicious_windows += 1

        if args.print_windows:
            t0 = start * args.hop_ms / 1000.0
            t1 = end   * args.hop_ms / 1000.0
            print(f"[{t0:7.2f}s–{t1:7.2f}s] score={score:6.3f}  ent_std={ent_std:.3f} spikes={ent_spikes:.3f} fluxHi={flux_hi_frac:.3f} dF0var={df0_var:.3f} voiced={voiced_frac:.3f}  -> {label}")

    if total_windows == 0:
        print("Audio too short for windowing.")
        return

    suspicious_ratio = suspicious_windows / total_windows
    overall = "deepfake suspected" if (suspicious_ratio > 0.35 or np.mean(scores) > args.score_thresh) else "bonafide"
    print("\n=== Summary ===")
    print(f"windows={total_windows}, suspicious={suspicious_windows} ({100.0*suspicious_ratio:.1f}%)")
    print(f"mean_score={np.mean(scores):.3f}, max_score={np.max(scores):.3f}")
    print(f"Decision: {overall}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Hinglish audio deepfake detector (signal-level baseline)")
    ap.add_argument("wav_path", help="Path to 16-bit PCM mono WAV at 16 kHz")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--frame_ms", type=float, default=25.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--mel_bands", type=int, default=40)
    ap.add_argument("--roll_sec", type=float, default=2.0)
    ap.add_argument("--min_f0", type=float, default=70.0)
    ap.add_argument("--max_f0", type=float, default=350.0)
    ap.add_argument("--energy_vad_dbfs", type=float, default=-45.0)
    ap.add_argument("--zcr_vad_max", type=float, default=0.18)
    # weights
    ap.add_argument("--w_ent_std", type=float, default=1.4)
    ap.add_argument("--w_ent_spikes", type=float, default=0.9)
    ap.add_argument("--w_flux_frac", type=float, default=1.1)
    ap.add_argument("--w_df0_var", type=float, default=0.8)
    ap.add_argument("--w_voiced_frac", type=float, default=1.0)
    ap.add_argument("--score_thresh", type=float, default=1.8)
    ap.add_argument("--print_windows", type=int, default=1)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    detect_from_wav(args.wav_path, args)
