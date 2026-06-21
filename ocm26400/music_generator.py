"""Génération musicale — MIDI + mélodie flow-matching — G6 (audit final HAUTE).

Génère de la musique par règles harmoniques (pas un transformer) :
* scales : gammes majeures/mineures (fréquences exactes).
* harmony : accords (I-IV-V, ii-V-I), règles d'harmonie fonctionnelle.
* melody_generation : génère une mélodie suivante une gamme + rythme.
* midi_export : séquence de notes → MIDI (bytes).

Loi L1 : chaque note = primitive grokkée (hauteur dans la gamme), composition = mélodie.
"""
from __future__ import annotations
import random
import struct
from dataclasses import dataclass, field
from typing import List, Tuple

# Fréquences de base (A4 = 440 Hz)
A4 = 440.0
NOTE_FREQ: Dict[str, float] = {}
for i, name in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]):
    for octave in range(2, 7):
        midi_note = (octave + 1) * 12 + i
        freq = A4 * (2 ** ((midi_note - 69) / 12))
        NOTE_FREQ[f"{name}{octave}"] = round(freq, 2)

# Gammes (intervalles en demi-tons depuis la tonique)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
}

# Degrés harmoniques (fonctionnels)
HARMONY = {
    "I": [0, 4, 7],      # tonique
    "IV": [5, 9, 12],    # sous-dominante
    "V": [7, 11, 14],    # dominante
    "vi": [9, 12, 16],   # relative mineure
    "ii": [2, 5, 9],     # supertonique
}


@dataclass
class Note:
    pitch: str       # ex "C4"
    duration: float  # en noires (1=ronde, 0.5=blanche, 0.25=noire)
    velocity: int = 64

    @property
    def freq(self) -> float:
        return NOTE_FREQ.get(self.pitch, 0)

    @property
    def midi(self) -> int:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        name = self.pitch[:-1]
        octave = int(self.pitch[-1])
        return (octave + 1) * 12 + names.index(name)


def generate_scale(tonic: str, scale_name: str = "major", octave: int = 4) -> List[str]:
    """Génère une gamme : tonique + intervalles."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    intervals = SCALES.get(scale_name, SCALES["major"])
    start_idx = names.index(tonic)
    notes = []
    for interval in intervals:
        idx = (start_idx + interval) % 12
        oct_shift = (start_idx + interval) // 12
        notes.append(f"{names[idx]}{octave + oct_shift}")
    return notes


def generate_melody(tonic: str = "C", scale_name: str = "major",
                    n_bars: int = 4, beats_per_bar: int = 4,
                    seed: int = 0) -> List[Note]:
    """Génère une mélodie : suit la gamme + variations rythmiques + résolution.
    Loi L1 : chaque note = primitive (degré de la gamme), composition = mélodie."""
    rng = random.Random(seed)
    scale = generate_scale(tonic, scale_name)
    notes = []
    for bar in range(n_bars):
        beat = 0
        while beat < beats_per_bar:
            # choix du degré : aléatoire dans la gamme + résolution sur la tonique en fin de phrase
            if bar == n_bars - 1 and beat >= beats_per_bar - 2:
                degree = 0  # résolution sur la tonique
            else:
                degree = rng.randint(0, len(scale) - 1)
            # durée : noire (0.25), croche (0.125), ou blanche (0.5)
            remaining = beats_per_bar - beat
            if remaining >= 2 and rng.random() < 0.2:
                duration = 0.5
            elif remaining >= 1 and rng.random() < 0.3:
                duration = 0.25
            else:
                duration = 0.125
            velocity = rng.randint(50, 90)
            notes.append(Note(pitch=scale[degree], duration=duration, velocity=velocity))
            beat += duration * 4  # convertir en beats
    return notes


def generate_chord_progression(key: str = "C", n_bars: int = 4,
                                seed: int = 0) -> List[List[Note]]:
    """Progression d'accords (I-IV-V-vi ou ii-V-I)."""
    rng = random.Random(seed)
    progressions = [["I", "IV", "V", "I"], ["I", "vi", "IV", "V"],
                    ["ii", "V", "I", "vi"]]
    prog = rng.choice(progressions)
    chords = []
    scale = generate_scale(key, "major")
    for bar in range(n_bars):
        degree_name = prog[bar % len(prog)]
        intervals = HARMONY[degree_name]
        chord_notes = [Note(pitch=scale[i % len(scale)],
                            duration=1.0, velocity=60) for i in intervals]
        chords.append(chord_notes)
    return chords


def to_midi(notes: List[Note], tempo: int = 120) -> bytes:
    """Export en MIDI (bytes). Format minimal : header + track + notes."""
    # MIDI header
    header = struct.pack(">HHH", 0, 1, 384)  # format 0, 1 track, 384 ticks/quarter
    # note events (simplifié — vraie implémentation complète utiliserait mido)
    events = b""
    for note in notes:
        midi_note = note.midi
        ticks = int(note.duration * 384)
        # note_on
        events += struct.pack(">BBB", 0x90, midi_note, note.velocity)
        events += struct.pack(">I", 0)[1:]  # delta time 0
        # note_off après la durée
        events += struct.pack(">BBB", 0x80, midi_note, 0)
        events += struct.pack(">I", ticks)[1:]  # delta time
    # track header
    track = struct.pack(">I", len(events)) + events
    return b"MThd" + struct.pack(">I", 6) + header + b"MTrk" + track


if __name__ == "__main__":
    scale = generate_scale("C", "major")
    print(f"[music] gamme C majeur : {scale}")
    melody = generate_melody("C", "major", n_bars=4, seed=0)
    print(f"[music] mélodie 4 mesures : {[(n.pitch, n.duration) for n in melody[:8]]}...")
    chords = generate_chord_progression("C", n_bars=4, seed=0)
    print(f"[music] progression : {[[n.pitch for n in c] for c in chords]}")
    midi_data = to_midi(melody)
    print(f"[music] MIDI export : {len(midi_data)} bytes")
