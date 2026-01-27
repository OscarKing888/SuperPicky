#!/usr/bin/env python3
"""合并音频并生成字幕文件"""

import wave
import os
import struct

AUDIO_DIR = "/Users/jameszhenyu/Documents/JamesAPPS/SuperPicky2026/promo_video/audio"
OUTPUT_DIR = "/Users/jameszhenyu/Documents/JamesAPPS/SuperPicky2026/promo_video"

# 音频文件和对应文案（按顺序）
segments = [
    ("01_hook.wav", "拍片一时爽，选片火葬场"),
    ("02_problem.wav", "800张照片，哪张最锐？\n让AI帮你3分钟搞定"),
    ("03_feature1.wav", "自动检测鸟眼位置"),
    ("04_feature2.wav", "计算头部锐度"),
    ("05_feature3.wav", "识别飞行姿态"),
    ("06_feature4.wav", "一键评分分类"),
    ("07_result.wav", "精选照片，张张能打"),
    ("08_cta.wav", "免费下载\nSuperPicky 慧眼选鸟"),
]

# 每段之间的间隔（秒）
GAP_DURATION = 0.5  # 段落间隔

def format_srt_time(seconds):
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def get_wav_duration(filepath):
    """获取 WAV 文件时长"""
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def merge_wav_files(input_files, output_file, gap_seconds=0.5):
    """合并多个 WAV 文件，支持添加间隔"""
    # 读取第一个文件获取参数
    with wave.open(input_files[0], 'rb') as first:
        params = first.getparams()
        sample_rate = first.getframerate()
        sample_width = first.getsampwidth()
        channels = first.getnchannels()

    # 计算间隔的帧数
    gap_frames = int(gap_seconds * sample_rate)
    silence = b'\x00' * (gap_frames * sample_width * channels)

    # 合并所有音频
    with wave.open(output_file, 'wb') as output:
        output.setparams(params)

        for i, filepath in enumerate(input_files):
            with wave.open(filepath, 'rb') as wf:
                output.writeframes(wf.readframes(wf.getnframes()))

            # 在段落之间添加间隔（最后一个不加）
            if i < len(input_files) - 1:
                output.writeframes(silence)

    return output_file

def generate_srt(segments_info, output_file):
    """生成 SRT 字幕文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (start, end, text) in enumerate(segments_info, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(f"{text}\n\n")

def main():
    print("=" * 50)
    print("合并音频并生成字幕")
    print("=" * 50)

    # 获取每个音频的时长并构建时间轴
    input_files = []
    segments_info = []
    current_time = 0.0

    for filename, text in segments:
        filepath = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(filepath):
            print(f"警告: 文件不存在 {filepath}")
            continue

        duration = get_wav_duration(filepath)
        input_files.append(filepath)

        # 记录字幕时间
        start_time = current_time
        end_time = current_time + duration
        segments_info.append((start_time, end_time, text))

        print(f"{filename}: {duration:.2f}秒 [{format_srt_time(start_time)} - {format_srt_time(end_time)}]")
        print(f"  文案: {text.replace(chr(10), ' ')}")

        # 更新时间（加上间隔）
        current_time = end_time + GAP_DURATION

    print("-" * 50)

    # 合并音频
    output_wav = os.path.join(OUTPUT_DIR, "voiceover.wav")
    merge_wav_files(input_files, output_wav, GAP_DURATION)
    print(f"合并音频: {output_wav}")

    # 获取合并后的总时长
    total_duration = get_wav_duration(output_wav)
    print(f"总时长: {total_duration:.2f}秒")

    # 生成 SRT 字幕
    output_srt = os.path.join(OUTPUT_DIR, "subtitle.srt")
    generate_srt(segments_info, output_srt)
    print(f"字幕文件: {output_srt}")

    print("=" * 50)
    print("完成!")
    print("=" * 50)

if __name__ == "__main__":
    main()
