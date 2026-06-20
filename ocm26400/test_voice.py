"""Tests TDD — voix conversationnelle (VAD + STT/TTS/STS, OCM-26400).

Valide le VAD réel (détection parole/silence, segmentation, fin de tour) sur de vrais
signaux audio (tons/silence), et le pipeline STS (stub).
"""
import torch

from ocm26400.voice import (
    VoiceActivityDetector, StubSTT, StubTTS, SpeechToSpeech, ConversationalLoop,
)


def _tone(freq=220, dur=0.5, sr=16000):
    t = torch.arange(int(sr * dur)).float() / sr
    return torch.sin(2 * torch.pi * freq * t)


def _silence(dur=0.5, sr=16000):
    return torch.zeros(int(sr * dur))


def test_vad_detects_speech_in_tone():
    """Un ton (parole) -> au moins un segment de parole détecté."""
    vad = VoiceActivityDetector(threshold=0.05, sr=16000)
    segs = vad.segments(_tone(220, 0.5))
    assert len(segs) >= 1
    assert segs[0][1] > segs[0][0]          # durée > 0


def test_vad_silence_detected_as_no_speech():
    """Silence (zéros) -> aucun segment de parole."""
    vad = VoiceActivityDetector(threshold=0.05)
    assert vad.segments(_silence(0.5)) == []
    assert vad.end_of_speech(_silence(0.5)) is None


def test_vad_splits_on_silence_gap():
    """Ton + silence + ton -> 2 segments distincts (détection de pause = fin de tour)."""
    vad = VoiceActivityDetector(threshold=0.05, min_silence_frames=4)
    wav = torch.cat([_tone(220, 0.3), _silence(0.3), _tone(330, 0.3)])
    segs = vad.segments(wav)
    assert len(segs) == 2


def test_end_of_speech_returns_last_segment_end():
    """Fin de parole = fin du dernier segment (silence soutenu après)."""
    vad = VoiceActivityDetector(threshold=0.05, min_silence_frames=4)
    wav = torch.cat([_tone(220, 0.3), _silence(0.3)])
    eos = vad.end_of_speech(wav)
    assert eos is not None and 0.2 < eos < 0.4


def test_tts_synthesizes_waveform():
    tts = StubTTS(sr=16000)
    w = tts.synthesize("bonjour")
    assert w.dim() == 1 and len(w) > 0
    assert float(w.abs().max()) > 0          # signal non silencieux


def test_sts_roundtrip():
    """STS : waveform -> STT -> réponse -> TTS -> waveform (pipeline conversationnel)."""
    sts = SpeechToSpeech(StubSTT({"default": "bonjour"}), StubTTS(),
                         respond=lambda t: f"ok {t}")
    out = sts(_tone(220, 0.5))
    assert out.dim() == 1 and len(out) > 0


def test_conversational_loop_turn():
    """Boucle conversationnelle : parole détectée -> fin de tour -> réponse audio."""
    vad = VoiceActivityDetector(threshold=0.05, min_silence_frames=4)
    sts = SpeechToSpeech(StubSTT({"default": "hello"}), StubTTS())
    loop = ConversationalLoop(vad, sts)
    wav = torch.cat([_tone(220, 0.3), _silence(0.3)])
    eos, reply = loop.turn(wav)
    assert eos is not None
    assert reply is not None and len(reply) > 0
