"""Shared prompt wording for dataset generation and user input."""


def build_prompt(instrument, description, midi_pitch, duration=6.0):
    """Describe a single synth note; pitch stays outside the predicted preset."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # MIDI advances one semitone per integer. Modulo 12 selects the note name;
    # integer division finds the octave (MIDI 60 is C4).
    note = f'{note_names[midi_pitch % 12]}{midi_pitch // 12 - 1}'
    # MIDI 69 is A4 at 440 Hz. Each 12-semitone step doubles the frequency.
    frequency = 440 * 2 ** ((midi_pitch - 69) / 12)
    # Keep the user's description as written, trimming only surrounding spaces.
    # An empty description leaves out this sentence entirely.
    description = description.strip()
    detail = f' Desired timbre: {description}.' if description else ''
    # The same wrapper surrounds sampled training descriptions and user input.
    # Acoustic instrument names describe timbre inspiration for a synth note.
    return (
        f'A single isolated {instrument.lower()}-inspired subtractive synthesizer note '
        f'at {note} (MIDI {midi_pitch}, {frequency:.2f} Hz).'
        f'{detail} Dry and monophonic, triggered once, '
        f'within a {duration:g}-second clip.'
    )
