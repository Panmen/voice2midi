import numpy as np

import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import get_window
from scipy.fft import fft, ifft

from mido import Message, MidiFile, MidiTrack, second2tick


def rcep(signal):
    """
    Compute the real cepstrum of the given signal.
    """

    dft = fft(signal)

    magnitude = np.abs(dft)

    return np.abs(ifft(np.log(magnitude)))


def pulse(low, high, length, double=False):
    """
    Creates a numpy array of the given length
    that has a unit pulse from low to high.
    """

    # TODO: remove double if it is not needed

    s = np.zeros(length)
    indexes = np.arange(high - low + 1) + low
    s[indexes] = np.ones(high - low + 1)

    if double:
        indexes = np.arange(high - low + 1) + low
        indexes = length - indexes
        s[indexes] = np.ones(high - low + 1)

    return s


def normalize(signal):
    """
    Scales the signal so that every sample is in the range [-1, 1].
    It does not add a DC bias.
    """

    M = max(signal)
    m = min(signal)

    M, m = abs(M), abs(m)
    scale = 1 / max(M, m)

    return signal * scale


def get_frames(signal, frame_size, frame_stride, window="hamming"):
    """
    Splits the given signal into frames.
    It returns a numpy array of shape (signal.shape[0], num_of_windows).
    """

    # Step 1: add padding if needed

    signal_len = signal.shape[0]
    num_frames = signal_len // frame_stride

    # the final size of the padded signal
    target_len = num_frames * frame_stride + frame_size
    pad_len = target_len - signal_len

    # add zeros only to the end
    signal = np.pad(signal, (0, pad_len))

    # Step 2: split into frames
    indexes = (
        np.arange(frame_size)[None, :] + frame_stride * np.arange(num_frames)[:, None]
    )
    frames = signal[indexes]

    # Step 3: apply window
    frames *= get_window(window, frame_size, fftbins=False)

    return frames


def load_wav(filename):
    """Load data from wav and normalize it."""

    samplerate, data = wavfile.read(filename)

    # if it is stereo, keep only one channel
    if data.ndim == 2:
        data = data[:, 0]

    # normalize the data,
    # Why?? => if its PCM 16, data is in range [-2^15, 2^15 - 1]
    data = normalize(data)

    return samplerate, data


def notes_to_midi(filename, notes, mask, frame_stride_sec):
    """
    Saves a sequence of notes (midi note numbers)
    to a midi file of the specified filename.
    The mask specifies if the note is valid or not.
    """

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # ticks between two consecutive frames
    t = int(second2tick(frame_stride_sec, mid.ticks_per_beat, 500_000))

    delta = 0
    i = 0

    while i < len(notes):

        if mask[i] == 0:
            # if there is no note
            delta += t
            i += 1
        else:
            # if there is a note

            # group all consecutive notes
            start_i = i
            while mask[i] != 0 and i < len(notes):
                i += 1
            end_i = i

            duration_frames = end_i - start_i
            duration = duration_frames * t

            # Filter out the start and the end
            # of the sequence. This is done because
            # of transient behavior of the human voice.
            start_i += int(duration_frames * 0.1)
            end_i -= int(duration_frames * 0.1)

            # get the average note
            avg_note = int(np.mean(notes[start_i:end_i]))

            # add the note to the track
            track.append(Message("note_on", note=avg_note, velocity=64, time=delta))
            track.append(
                Message("note_off", note=avg_note, velocity=127, time=duration)
            )

            delta = 0

    mid.save(filename)


def debug_cep_pitch(cepstrum, pitch, mask, samplerate, split=1):

    for i in range(split):
        segment = cepstrum.shape[0] // split
        start, fin = segment * i, segment * (i + 1)

        plt.subplot(split, 1, i + 1)
        plt.imshow(
            cepstrum[start:fin, : int(cepstrum.shape[1] // 2)].T ** 2,
            cmap="hot",
            interpolation="nearest",
            aspect="auto",
        )
        plt.colorbar()

        plt.plot(mask * samplerate / pitch[start:fin], alpha=0.7)

    plt.show()
