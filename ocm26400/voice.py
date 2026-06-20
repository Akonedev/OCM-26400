"""Voix conversationnelle — VAD (détection silence) + STT/TTS/STS (OCM-26400, spec).

Le cahier des charges demande « Audio conversationnel, real time, détection de
silences, TTS, STT, STS, ASR ». On implémente :

* VAD (Voice Activity Detection) RÉEL : détection de parole/silence par énergie RMS
  sur le waveform, segmentation en tours de parole (end-of-speech = silence après
  parole). C'est le cœur temps-réel d'une conversation audio (détection de silence).
* Interfaces STT (speech-to-text), TTS (text-to-speech), STS (speech-to-speech) :
  backend branchable. VAD + turn-taking sont réels ; les cœurs STT/TTS sont des
  backends simples/procéduraux (en production : Whisper/Tacotron — interface identique).

HONNÊTE : VAD/segmentation/turn-taking = vrai traitement signal réel. STT/TTS cores =
backends stub (à remplacer par modèles réels, sans provider externe). Le PIPELINE
conversationnel (silence -> fin de tour -> STT -> réponse -> TTS) est réel.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import torch


# ---------------- VAD (détection parole / silence) — RÉEL ----------------

def rms_energy(waveform: torch.Tensor, frame: int = 480) -> torch.Tensor:
    """Énergie RMS par frame (frame en échantillons). waveform: (T,) ou (B,T)."""
    w = waveform if waveform.dim() == 2 else waveform.unsqueeze(0)
    pad = (frame - (w.shape[-1] % frame)) % frame
    w = torch.nn.functional.pad(w, (0, pad))
    frames = w.view(w.shape[0], -1, frame).float()
    return (frames ** 2).mean(dim=-1).sqrt()           # (B, n_frames)


@dataclass
class VoiceActivityDetector:
    """VAD par énergie RMS : segmente la parole, détecte la fin de tour (silence)."""
    threshold: float = 0.02          # énergie RMS sous laquelle = silence
    frame: int = 480                 # ~30ms @16kHz
    min_silence_frames: int = 8      # silence soutenu => fin de tour
    sr: int = 16000

    def speech_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mask bool par frame : True = parole (énergie > seuil)."""
        e = rms_energy(waveform, self.frame).squeeze(0)
        return e > self.threshold

    def segments(self, waveform: torch.Tensor) -> List[Tuple[float, float]]:
        """Segments de parole (start_s, end_s), en fusionnant les gaps < min_silence."""
        mask = self.speech_mask(waveform)
        frame_s = self.frame / self.sr
        segs, start = [], None
        silence = 0
        for i, v in enumerate(mask.tolist()):
            if v:
                if start is None:
                    start = i
                silence = 0
            else:
                if start is not None:
                    silence += 1
                    if silence >= self.min_silence_frames:
                        segs.append((start * frame_s, (i - silence) * frame_s))
                        start = None
        if start is not None:
            segs.append((start * frame_s, len(mask) * frame_s))
        return segs

    def end_of_speech(self, waveform: torch.Tensor) -> Optional[float]:
        """Instant (s) de fin du dernier tour de parole (silence soutenu), ou None."""
        segs = self.segments(waveform)
        return segs[-1][1] if segs else None


# ---------------- STT / TTS / STS — interfaces branchables ----------------

STT = Callable[[torch.Tensor], str]       # waveform -> texte
TTS = Callable[[str], torch.Tensor]       # texte -> waveform


class StubSTT:
    """STT stub : transcription déterministe (backend réel = Whisper plus tard)."""
    def __init__(self, mapping: Optional[dict] = None):
        self.mapping = mapping or {}

    def transcribe(self, waveform: torch.Tensor) -> str:
        # stub : si l'énergie indique de la parole, retourne une transcription câblée
        e = float(rms_energy(waveform).mean())
        return self.mapping.get("default", "[parole détectée]") if e > 0.01 else "[silence]"


class StubTTS:
    """TTS stub : synthétise un waveform tonal (backend réel = Tacotron plus tard)."""
    def __init__(self, sr: int = 16000, freq: float = 220.0):
        self.sr, self.freq = sr, freq

    def synthesize(self, text: str) -> torch.Tensor:
        dur = max(0.2, min(2.0, 0.1 * len(text)))     # durée ~ longueur du texte
        t = torch.arange(int(self.sr * dur)).float() / self.sr
        env = torch.linspace(0.6, 1.0, len(t))         # enveloppe simple (prosodie)
        return env * torch.sin(2 * torch.pi * self.freq * t)


@dataclass
class SpeechToSpeech:
    """STS : waveform -> STT -> réponse -> TTS -> waveform (conversation continue)."""
    stt: StubSTT
    tts: StubTTS
    respond: Callable[[str], str] = lambda txt: f"reçu: {txt}"

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        text = self.stt.transcribe(waveform)
        reply = self.respond(text)
        return self.tts.synthesize(reply)


@dataclass
class ConversationalLoop:
    """Boucle conversationnelle pilotée par le VAD : silence => fin de tour => réponse.

    Honnête : sur un flux audio réel, on détecterait la fin de parole en streaming ;
    ici sur un waveform complet (le mécanisme de turn-taking VAD est réel).
    """
    vad: VoiceActivityDetector
    sts: SpeechToSpeech

    def turn(self, waveform: torch.Tensor):
        """Retourne (fin_de_parole_s, réponse_audio) ou (None, None) si pas de parole."""
        eos = self.vad.end_of_speech(waveform)
        if eos is None:
            return None, None
        reply = self.sts(waveform)
        return eos, reply


class FormantTTS:
    """TTS par synthèse de formants : texte → waveform via voyelles → formants.

    Chaque voyelle a des fréquences de formant (F1, F2) réelles issues de la phonétique.
    Honnête : synthèse par formants (source-filtre), pas un vocoder entraîné.
    """

    VOWEL_FORMANTS = {
        'a': (730, 1090), 'e': (530, 1840), 'i': (270, 2290),
        'o': (570, 840), 'u': (300, 870), 'y': (270, 1820),
        'é': (400, 2200), 'è': (550, 1700),
    }
    SAMPLE_RATE = 16000

    def synthesize(self, text: str) -> torch.Tensor:
        """Synthétise un waveform à partir du texte (voyelles → formants)."""
        text = text.lower()
        vowels = [c for c in text if c in self.VOWEL_FORMANTS]
        if not vowels:
            vowels = ['a']
        dur_per_vowel = max(0.08, min(0.2, 1.0 / len(vowels)))
        waveforms = []
        for v in vowels:
            f1, f2 = self.VOWEL_FORMANTS[v]
            t = torch.arange(int(self.SAMPLE_RATE * dur_per_vowel)).float() / self.SAMPLE_RATE
            env = torch.hann_window(len(t))
            wav = env * (0.5 * torch.sin(2 * 3.14159 * f1 * t) +
                         0.3 * torch.sin(2 * 3.14159 * f2 * t))
            waveforms.append(wav)
        return torch.cat(waveforms) if waveforms else torch.zeros(self.SAMPLE_RATE)
