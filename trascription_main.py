# -*- coding: utf-8 -*-

__author__ = 'Roy'

from singing_transcription import *
import os
import string
import matplotlib.pyplot as plt



def singing_transcription(file_name, plot=False):
    """
    必须安装Vamp Plugin和pyin，否则无法运行!
    请保证vamp-simple-host文件在以下文件夹中，否则无法运行
    Mac用户默认Vamp host地址为$HOME/Library/Audio/Plug-Ins/Vamp
    Windows(64-bit)用户默认Vamp host地址为C:\Program Files (x86)\Vamp Plugins
    Windows(32-bit)用户默认Vamp host地址为C:\Program Files\Vamp Plugins
    Linux用户默认Vamp host地址为$HOME/vamp

    之后考虑写一个自动下载的脚本，如果检测不到的话就自动下载vamp和pyin


    总函数，输入为音频文件的地址
    参数为：
    file_name：音频文件的地址
    output_path：输出的位置，默认为空
    plot：是否画图，默认为False
    """

    os.system("$HOME/Library/Audio/Plug-Ins/Vamp/vamp-simple-host pyin:pyin:smoothedpitchtrack '"+file_name+"' -o 'evaluation/pitch/"+file_name.split('.')[0].split('/')[-1]+".txt'")
    test = open('evaluation/pitch/'+file_name.split('.')[0].split('/')[-1]+'.txt')



    time = []
    val = []

    for line in test:
        time.append(string.atof(line.split(':')[0]))
        val.append(string.atof(line.split(':')[1]))

    val = freq2cent(val)

    test.close()

    # 给无声区的pitch全部赋0
    full_pitch = []
    full_time = []
    step_size = 1.0/44100.0*256
    max_frame = int(np.rint(max(time) / step_size))
    cnt = 0
    for i in range(max_frame):
        full_time.append(step_size * i)
        if time[cnt] - step_size * i < 0.00001:
            cnt += 1
            full_pitch.append(val[cnt])
        else:
            full_pitch.append(0.1)

    original_pitch = full_pitch

    # 5点平滑
    smooth_pitch = full_pitch[0:2]
    for i in range(2, len(full_pitch)-2):
        smooth_pitch.append(np.mean(full_pitch[i-2:i+3]))
    smooth_pitch.append(full_pitch[-2])
    smooth_pitch.append(full_pitch[-1])

    full_pitch = smooth_pitch

    # 初步分割
    note = note_segment(full_time, full_pitch)

    print "Note segmentation 1 done."

    # 后处理合并分割的太细的
    note = note_postprocessing(note, full_pitch)

    print "Note postprocessing done."

    # 检测颤音
    vibrato, note = vibrato_detection(note, full_pitch)


    print "Vibrato detection done."

    # 检测分的太粗的note
    note = small_note_segment(note, full_pitch, vibrato)

    print "Note segmentation 2 done."

    # 调整onset和offset的位置
    note = onset_offset_adjust(note, full_pitch)

    print "Onset/offset adjustment done."

    # 检测修饰音
    grace_note = grace_note_detection(note, full_pitch)

    print 'Grace note detection done.'

    full_pitch = np.array(cent2freq(original_pitch))
    full_pitch = 69 + 12 * np.log2(full_pitch / 440.)
    full_time = np.array(full_time)

    final_onset = []
    final_offset = []
    onset_pitch = []
    offset_pitch = []
    note_pitch = []
    for i in range(len(note)):
        final_onset.append(full_time[note[i][0]])
        final_offset.append(full_time[note[i][1]])
        onset_pitch.append(full_pitch[note[i][0]])
        offset_pitch.append(full_pitch[note[i][1]])
        note_pitch.append(note[i][2])

    print "Note pitch in Hz:", cent2freq(note_pitch)
    note_pitch = freq_to_nearest_MIDI(cent2freq(note_pitch))


    grace_onset = []
    grace_offset = []
    grace_pitch = []
    for i in range(len(grace_note)):
        grace_onset.append(full_time[grace_note[i][0]])
        grace_offset.append(full_time[grace_note[i][1]])
        grace_pitch.append(grace_note[i][2])

    """
    vibrato_onset = []
    vibrato_offset = []
    for i in range(len(vibrato)):
        vibrato_onset.append(full_time[vibrato[i][0]])
        vibrato_offset.append(full_time[vibrato[i][1]])
    """

    grace_onset = sorted(grace_onset)
    grace_offset = sorted(grace_offset)

    # 我们需要处理一下装饰音
    # 在转录装饰音的时候，不能把它转录为单独的音符
    # 需要跟后一个音符合并，音高用被装饰音的
    note_number = 0
    note_length = len(note_pitch['midi_string'])
    while note_number < note_length-1:
        if final_offset[note_number] in grace_offset:
            # 当前note删除，合并到下一个note里面
            del note_pitch['midi_string'][note_number]
            del note_pitch['accurate_midi_note'][note_number]
            del note_pitch['midi_note'][note_number]
            final_onset[note_number+1] = final_onset[note_number]
            del final_onset[note_number]
            del final_offset[note_number]
            del onset_pitch[note_number]
            del offset_pitch[note_number]
            note_number += 1
            note_length -= 1
        else:
            note_number += 1


    print "Note name in MIDI:", note_pitch['midi_string']
    print "Note number in MIDI:", note_pitch['accurate_midi_note']
    print "Note onset:", final_onset
    print "Note offset:", final_offset
    #print "Vibrato onset:", vibrato_onset
    #print "Vibrato offset:", vibrato_offset
    print "Grace note onset:", grace_onset
    print "Grace note offset:", grace_offset

    # 输出到MIDI
    # MIDI_file = generate_MIDI(note_pitch, final_onset, final_offset, file_name)

    # 输出到文件
    file_out = open('evaluation/transcription/'+file_name.split('.')[0].split('/')[-1]+".txt", 'w')

    for i in range(len(final_onset)):
        print >> file_out, "%.3f %.3f %.3f" % (final_onset[i], final_offset[i], note_pitch['accurate_midi_note'][i])
    file_out.close()

    if plot:
        font = {'weight': 'bold',
        'size': 14}
        plt.rc('font', **font)
        plt.grid(True)
        # 把所有的音高都换成MIDI
        # 调整y坐标轴范围
        min_freq = min([freq_value for freq_value in full_pitch if freq_value > 23])
        plt.ylim((min_freq * 0.95, max(full_pitch) * 1.05))
        plt.plot(full_time, full_pitch, '-')
        plt.plot(final_onset, onset_pitch, 'ro')
        plt.plot(final_offset, offset_pitch, 'yo')
        plt.xlabel('Time (second)')
        plt.ylabel('MIDI note')
        plt.show()
