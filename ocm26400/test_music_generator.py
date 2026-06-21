"""Tests génération musicale (OCM-26400)."""
from ocm26400.music_generator import (
    generate_scale, generate_melody, generate_chord_progression, to_midi, Note,
    SCALES, HARMONY,
)


def test_generate_scale_major():
    scale = generate_scale("C", "major")
    assert scale[0] == "C4" and len(scale) == 7


def test_generate_scale_pentatonic():
    scale = generate_scale("A", "pentatonic")
    assert len(scale) == 5


def test_generate_melody_returns_notes():
    melody = generate_melody("C", "major", n_bars=4, seed=0)
    assert len(melody) > 0
    assert all(isinstance(n, Note) for n in melody)


def test_melody_resolves_to_tonic():
    """La dernière note de la mélodie tend vers la tonique (loi L1 : résolution)."""
    melody = generate_melody("G", "major", n_bars=4, seed=0)
    last = melody[-1]
    assert "G" in last.pitch  # résolution sur la tonique


def test_chord_progression():
    chords = generate_chord_progression("C", n_bars=4, seed=0)
    assert len(chords) == 4
    assert all(len(c) >= 3 for c in chords)  # accords = au moins 3 notes


def test_to_midi_returns_bytes():
    melody = generate_melody("C", "major", n_bars=2, seed=0)
    midi = to_midi(melody)
    assert isinstance(midi, bytes) and len(midi) > 0


def test_note_midi_number():
    n = Note(pitch="A4", duration=0.25)
    assert n.midi == 69  # A4 = MIDI note 69
    assert abs(n.freq - 440.0) < 0.5
