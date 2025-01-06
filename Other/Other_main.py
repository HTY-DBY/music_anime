import os

import numpy as np
from pydub import AudioSegment


def init_ffmpeg_IN_Other_main(ffmpeg_ins_path):
	os.environ["PATH"] += os.pathsep + ffmpeg_ins_path
	os.environ["PATH"] += os.pathsep + os.path.join(ffmpeg_ins_path, "bin")
	AudioSegment.converter = os.path.join(ffmpeg_ins_path, "bin", "ffmpeg.exe")
	AudioSegment.ffprobe = os.path.join(ffmpeg_ins_path, "bin", "ffprobe.exe")
