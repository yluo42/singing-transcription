# -*- coding: utf-8 -*-
__author__ = 'Roy'

from trascription_main import *
import os
import commands
import sys

def evaluate_transcription(eval_file_path):

    """
    现在默认在osx下的操作
    之后补上判断操作系统的代码
    """

    os.system("cd " + eval_file_path)
    os.system("ls "+eval_file_path+" > wav_files.txt")

    eval_file = open("wav_files.txt")

    file_names = []
    for line in eval_file:
        file_names.append(eval_file_path.strip("'")+"/"+line[:-1])

    output_path = 'evaluation'
    os.system('rm -r '+output_path)
    os.system('mkdir '+output_path)
    os.system('mkdir '+output_path+'/pitch')
    os.system('mkdir '+output_path+'/transcription')

    for file in file_names:
        singing_transcription(file, plot=False)

    os.system("rm "+"wav_files.txt")



evaluate_transcription(eval_file_path="'/Users/Roy/Dropbox/singing transcription project/Evaluation/Eval_framework-ISMIR14'")