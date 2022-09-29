#!/usr/bin/env python3

import argparse
from util import *

###########
# Constants
###########

FRAME_SIZE_SEC = 40e-3
FRAME_STRIDE_SEC = FRAME_SIZE_SEC / 8
POWER_THRES = 0.01

###############################
# Step 0: Get the cli arguments
###############################

parser = argparse.ArgumentParser(description="Convert wav files of human voice to midi.")

parser.add_argument("input_file", type=str, help="The path to the wav file")
parser.add_argument("-o", type=str, help="The path to the output file (default: out.mid)", default="out.mid")
parser.add_argument("-d", "--debug", action="store_true", help="Debug flag. If it is set then the cepstrum is plotted.")

args = parser.parse_args()

DEBUG = args.debug

INPUT_FILE = args.input_file
OUTPUT_FILE = args.o


#######################
# Step 1: load the file
#######################

samplerate, voice = load_wav(INPUT_FILE)


#################################
# Step 2: split audio into frames
#################################

frame_size_samples = int(samplerate * FRAME_SIZE_SEC)
frame_stride_samples = int(samplerate * FRAME_STRIDE_SEC)

frames = get_frames(voice, frame_size_samples, frame_stride_samples, window="hamming")


##############################################
# Step 3: calculate the cepstrum of each frame
##############################################

cepstrum = np.apply_along_axis(rcep, 1, frames)

# Smoothen cepstrum of each frame with a low pass filter
h = np.ones(3) / 3
cepstrum = np.apply_along_axis(np.convolve, 1, cepstrum, h, "same")


###########################
# Step 4: extract the pitch
###########################

# Lower and upper bound on human voice pitch
low_bound = 85
high_bound = 255

low_bound_quefrency = 1 / high_bound
high_bound_quefrency = 1 / low_bound

low_samples = int(low_bound_quefrency * samplerate)
high_samples = int(high_bound_quefrency * samplerate)

lifter = pulse(low_samples, high_samples, frame_size_samples, double=True)

cepstrum *= lifter

# get the peak/pitch in Hz
pitch = samplerate / np.argmax(cepstrum[:, : int(frame_size_samples // 2)], axis=1)

# apply low pass filter on the pitch
n = 8
h = np.ones(n) / n
pitch = np.convolve(pitch, h, "same")


##################################################
# Step 5: Find the closest midi note to each frame
##################################################

# notes_hz[i] = frequency in Hz of the ith midi note
notes_hz = 440 * np.power(2, (np.arange(0, 128) - 69) / 12)

# get the closest note for each frame
# based on the pitch
tmp = np.abs(pitch[:, None] - notes_hz[None, :])
note_seq = np.argmin(tmp, axis=1)


########################################################
# Step 6: Create a mask based on the power of each frame
########################################################

power = np.apply_along_axis(lambda x: np.sum(x**2) / x.shape[0], 1, frames)
mask = power > POWER_THRES


##############################
# Step 7: Create the midi file
##############################

notes_to_midi(OUTPUT_FILE, note_seq, mask, FRAME_STRIDE_SEC)


#######
# DEBUG
#######

# debug plot cepstrum and pitch
if DEBUG:
    debug_cep_pitch(cepstrum, pitch, mask, samplerate, split=1)
