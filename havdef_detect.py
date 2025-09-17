
import argparse
import os
import sys
import numpy as np
import soundfile as sf
import librosa
import sounddevice as sd

SAMPLE_RATE = 20000
# Demo detector (works out-of-the-box with HF "audio-classification" pipeline).
# NOTE: Good for a demo; not production-hardened. Swap with a stronger model later (see notes).
HF_MODEL_ID = "Kaustubh911/havdef-audio-detector"  # fallback models listed below

def info(msg): print(f"[HAVDEF] {msg}")

def record_microphone(seconds: int, out_path: str, samplerate=SAMPLE_RATE):
    try:
        import sounddevice as sd
    except Exception as e:
        raise RuntimeError("sounddevice not available. Install it or use --file.") from e
    info(f"Recording {seconds}s from mic at {samplerate} Hz …")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(out_path, audio, samplerate)
    info(f"Saved raw recording -> {out_path}")
    return out_path

def preprocess_wav(in_wav: str, out_wav: str, target_sr=SAMPLE_RATE):
    y, sr = librosa.load(in_wav, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # normalize to [-1, 1]
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    sf.write(out_wav, y, target_sr)
    return out_wav

def trim_with_vad(in_wav: str, out_wav: str, sr=SAMPLE_RATE):
    """
    Uses Silero VAD to keep only speech. Falls back to untrimmed audio on any error.
    """
    try:
        import torch
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')  # tiny, CPU-friendly
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        wav = read_audio(in_wav, sampling_rate=sr)
        ts = get_speech_timestamps(wav, model, sampling_rate=sr)
        if not ts:
            info("VAD found no speech; using the original audio.")
            return preprocess_wav(in_wav, out_wav, sr)
        speech = collect_chunks(ts, wav)
        sf.write(out_wav, speech, sr)
        return out_wav
    except Exception as e:
        info(f"VAD unavailable ({e}); using untrimmed audio.")
        return preprocess_wav(in_wav, out_wav, sr)

def load_detector(model_id: str = HF_MODEL_ID):
    from transformers import pipeline
    info(f"Loading detector: {model_id}")
    clf = pipeline("audio-classification", model=model_id, device=-1, top_k=None)
    return clf

def decide_from_labels(results, fake_threshold=0.5, real_threshold=0.5):
    """
    Rule:
      - If ANY label indicating fake/spoof/synth has score >= fake_threshold -> DEEPFAKE
      - Else if ANY label indicating real/bona has score >= real_threshold and no fake above threshold -> REAL
      - Else fall back to comparing best fake vs best real; if still ambiguous, use the top label.
    """
    score_fake, score_real = 0.0, 0.0
    top_fake, top_real = 0.0, 0.0

    for r in results:
        lab = r["label"].lower()
        sc = float(r["score"])
        if any(k in lab for k in ["fake", "spoof", "synth", "ai", "generated"]):
            top_fake = max(top_fake, sc)
        if any(k in lab for k in ["real", "bona", "bonafide", "human"]):
            top_real = max(top_real, sc)

    # Hard guarantees
    if top_fake >= fake_threshold:
        return "DEEPFAKE", top_fake
    if top_real >= real_threshold and top_fake < fake_threshold:
        return "REAL", top_real

    # Soft comparison if thresholds not met
    if top_fake > 0.0 or top_real > 0.0:
        if top_fake >= top_real:
            return "DEEPFAKE", top_fake
        else:
            return "REAL", top_real

    # Fallback: take top label
    top = max(results, key=lambda x: x["score"])
    if any(k in top["label"].lower() for k in ["fake", "spoof", "synth", "ai", "generated"]):
        return "DEEPFAKE", float(top["score"])
    return "REAL", float(top["score"])


def run_detection(path: str, model_id: str, fake_threshold: float = 0.5):
    pre = preprocess_wav(path, "_preprocessed.wav", SAMPLE_RATE)
    trimmed = trim_with_vad(pre, "_speech.wav", SAMPLE_RATE)
    clf = load_detector(model_id)
    results = clf(trimmed)
    verdict, conf = decide_from_labels(results, fake_threshold=fake_threshold)
    info(f"Verdict: {verdict}  |  confidence ~ {conf:.3f}  |  fake_threshold={fake_threshold}")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
        print(f"  - {r['label']}: {r['score']:.3f}")
    return verdict, conf


def main():
    ap = argparse.ArgumentParser(description="HAVDEF demo: mic record + deepfake detection")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", type=int, help="Record N seconds from mic, then analyze.")
    g.add_argument("--file", type=str, help="Analyze an existing audio file (.wav/.mp3/.flac).")
    ap.add_argument("--model", type=str, default=HF_MODEL_ID, help="Hugging Face model id to use.")
    args = ap.parse_args()

    if args.record is not None:
        raw = "_mic.wav"
        record_microphone(args.record, raw, SAMPLE_RATE)
        run_detection(raw, args.model)
    else:
        if not os.path.exists(args.file):
            print("Input file not found. In Colab, upload a file and pass its path.")
            sys.exit(1)
        run_detection(args.file, args.model)

if __name__ == "__main__":
    main()