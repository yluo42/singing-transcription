# -*- coding: utf-8 -*-

__author__ = 'Roy'

import numpy as np
from midiutil.MidiFile import MIDIFile


# 将频率转换为音分

def freq2cent(freq):
    # 以A4为基准，即440Hz为基准
    # 这似乎是个公认的标准
    cent = []
    for i in range(len(freq)):
        cent.append(1200*np.log2(freq[i]/(440. * np.power(2., 3./12.-5))))
    return cent


# 将音分转换回频率

def cent2freq(cent):
    # 同样以A4为基准
    freq = []
    for i in range(len(cent)):
        freq.append((440. * np.power(2., 3./12.-5)) * np.power(2., cent[i]/1200.))
    return freq

# 把音高匹配到它最近的MIDI音高上

def freq_to_nearest_MIDI(freq):

    near_MIDI = {'midi_string': [], 'accurate_midi_note': [], 'midi_note': []}

    for i in range(len(freq)):
        # 先记录这时候的midi note
        note_value = 69 + 12 * np.log2(freq[i] / 440.)
        near_MIDI['accurate_midi_note'].append(round(note_value, 2))
        # 匹配到更近的
        current_midi = int(np.rint(note_value))
        near_MIDI['midi_note'].append(current_midi)
        octave = str(current_midi / 12 - 1)
        pitch = current_midi % 12
        if pitch == 0:
            pitch = 'C'
        elif pitch == 1:
            pitch = 'C#'
        elif pitch == 2:
            pitch = 'D'
        elif pitch == 3:
            pitch = 'D#'
        elif pitch == 4:
            pitch = 'E'
        elif pitch == 5:
            pitch = 'F'
        elif pitch == 6:
            pitch = 'F#'
        elif pitch == 7:
            pitch = 'G'
        elif pitch == 8:
            pitch = 'G#'
        elif pitch == 9:
            pitch = 'A'
        elif pitch == 10:
            pitch = 'A#'
        elif pitch == 11:
            pitch = 'B'
        near_MIDI['midi_string'].append(pitch + octave)

    return near_MIDI


# 用每个note的起止时间和MIDI pitch创建一个MIDI文件

def generate_MIDI(note_pitch, final_onset, final_offset, file_name):
    # 创建一个MIDI track
    MIDI_file = MIDIFile(1)
    # 由于这个库只能用tempo来建立MIDI音符，所以要先设设置BPM
    # 这里设成一个beat的时间跟pYIN的hop size一样即可，即为5.8ms
    # 但事实证明这样的tempo太大了无法读取
    # 所以只能用最大的分辨率（tempo为990）再取整了
    MIDI_file.addTempo(0, 0, tempo=990)
    MIDI_file.addTrackName(0, 0, file_name.split('.')[0])

    for i in range(len(note_pitch['midi_note'])):
        current_midi = note_pitch['midi_note'][i]
        current_onset = final_onset[i] / (60./990)
        current_offset = final_offset[i] / (60./990)
        # 将这个音符加入MIDI文件
        MIDI_file.addNote(track=0, channel=0, pitch=current_midi, time=current_onset, duration=current_offset-current_onset, volume=100)

    # 输出MIDI文件
    midi_write_file = open(file_name.split('.')[0]+".mid", 'wb')
    MIDI_file.writeFile(midi_write_file)
    midi_write_file.close()

    return MIDI_file