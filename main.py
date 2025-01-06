import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from matplotlib import rcParams
from moviepy import VideoClip, AudioFileClip
from pydub import AudioSegment
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from Other.Goble import GobleD
from Other.Other_main import init_ffmpeg_IN_Other_main

# 初始化FFmpeg路径
init_ffmpeg_IN_Other_main(GobleD().ffmpeg_ins_path)

# 配置参数
frame_set = 30  # 每秒的帧数
n_seconds = 5  # 设置窗口的持续时间，比如最近5秒的数据
x_offset = 0.5  # 调整x轴偏移量，单位为秒
roit = [0.09, 0.14]
y_offset = -700  # 调整y轴像素偏移量
audio_path = os.path.join(GobleD().music_down_path, 'kanong.flac')  # 输入音频文件路径
output_video_path = os.path.join(GobleD().music_down_path, 'output.mp4')  # 输出视频路径
person_image_path = GobleD().person_img_path  # 图片或GIF的路径

# 配置字体以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 加载音频文件
audio = AudioSegment.from_file(audio_path)
duration_ms = len(audio)  # 音频总时长（毫秒）
sampling_rate = audio.frame_rate  # 采样率

# 计算帧数
frame_count = int(duration_ms // (1000 / frame_set))  # 计算总帧数

begin_cal_rms = 0  # 用于控制 RMS 计算的全局变量


# 预计算所有的响度数据
def calculate_all_rms():
	"""预计算所有的音频数据的RMS值"""
	global begin_cal_rms  # 使用全局变量
	x_data_all, y_data_all = [], []

	for start_ms in range(0, duration_ms, 1000 // frame_set):
		end_ms = min(start_ms + (1000 // frame_set), duration_ms)
		audio_frame = audio[start_ms:end_ms]
		samples = np.array(audio_frame.get_array_of_samples())

		# 只要检测到样本和大于0，便开始计算RMS
		if np.sum(samples) > 0:
			begin_cal_rms = 1

		# 计算RMS，若begin_cal_rms为0，则跳过计算
		if begin_cal_rms == 1:
			rms_value = calculate_rms(samples)
		else:
			rms_value = 0.1

		# 如果rms_value无效，跳过
		if np.isnan(rms_value) or np.isinf(rms_value):
			continue

		x_data_all.append(start_ms / 1000)  # 转换为秒
		y_data_all.append(rms_value)

	return np.array(x_data_all), np.array(y_data_all)


# 计算RMS并避免NaN和Inf
def calculate_rms(samples):
	try:
		# 检查是否有无效值
		if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
			return 0.1  # 使用默认值，避免计算出无效值
		# 计算RMS，返回均方根值
		rms_value = np.sqrt(np.mean(samples ** 2))
		return rms_value if rms_value > 0 else 0.1  # 确保返回有效的RMS值
	except Exception as e:
		print(f"Error calculating RMS: {e}")
		return 0.1  # 默认值


# 获取响度数据
x_data_all, y_data_all = calculate_all_rms()

# 计算最大RMS值，并确保它是有效的
max_rms = np.max(y_data_all) if len(y_data_all) > 0 else 0.1
if np.isnan(max_rms) or np.isinf(max_rms):
	max_rms = 0.1
else:
	max_rms *= 1.1  # 增加最大RMS值的范围

# 创建图形
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)

# 初始化line并连接到ax
line, = ax.plot(x_data_all, y_data_all, lw=2)

ax.set_ylim(y_offset, max_rms)
ax.set_xlabel("时间 (秒)")  # x轴标签
ax.set_ylabel("响度 (RMS)")  # y轴标签

# 加载小人图像（支持JPG, PNG, GIF）
person_image = Image.open(person_image_path)
if person_image.format == 'GIF':
	# 如果是GIF，加载所有帧并保留它们
	gif_frames = [frame.copy() for frame in ImageSequence.Iterator(person_image)]
	# 将GIF转换为RGBA以支持透明背景
	gif_frames = [frame.convert("RGBA") for frame in gif_frames]
	gif_frame_count = len(gif_frames)
else:
	# 如果是静态图像（JPG/PNG），则转换为RGBA格式
	person_image = person_image.convert("RGBA")
	gif_frames = [person_image]  # 如果是单帧图像，我们直接用这个图像
	gif_frame_count = 1

# 初始化 person_marker，设置初始位置
person_marker = ax.imshow(gif_frames[0], aspect='auto', extent=(0, 0, 0, 0))  # 初始设置图片位置为(0, 0)
person_position = [0, 0]  # 小人的初始位置 [x, y]


# 更新函数：更新图像的位置和大小
def update(frame):
	current_time = frame / frame_set + x_offset  # 调整时间对齐
	ax.set_xlim(current_time - n_seconds, current_time + n_seconds)

	# 计算图像的中心位置
	x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
	person_position[0] = x_center

	# 找到最接近当前时间窗口中心的 RMS 点
	closest_idx = np.argmin(np.abs(x_data_all - x_center))
	person_position[1] = y_data_all[closest_idx]

	# 获取当前的坐标轴范围
	x_min, x_max = ax.get_xlim()
	y_min, y_max = ax.get_ylim()

	# 计算画布的缩放比例
	x_range = x_max - x_min
	y_range = y_max - y_min

	# 计算图像的宽度和高度，并保持原始宽高比
	width_set = x_range * roit[0]
	height_set = y_range * roit[1]

	# 选择当前帧的GIF图像
	gif_frame_idx = frame % gif_frame_count
	person_marker.set_data(gif_frames[gif_frame_idx])

	# 更新图像的位置和大小
	person_marker.set_extent([person_position[0] - width_set / 2, person_position[0] + width_set / 2,
							  person_position[1] - height_set / 2, person_position[1] + height_set / 2])

	return line, person_marker


# 渲染每一帧图像
def render_ani_to_image(frame):
	update(frame)
	canvas = FigureCanvas(fig)
	canvas.draw()
	image = np.array(canvas.buffer_rgba())[..., :3]  # 将 RGBA 转为 RGB
	return image


# 获取音频时长
audio_clip = AudioFileClip(audio_path)
duration = audio_clip.duration


# 创建视频帧
def make_frame(t):
	frame = int(t * frame_count / duration)
	return render_ani_to_image(frame)


# 创建视频剪辑对象
video_clip = VideoClip(make_frame, duration=duration)

# 添加音频到视频中
video_clip = video_clip.with_audio(audio_clip)

# 输出为 MP4 文件
video_clip.write_videofile(output_video_path, codec="libx264", fps=frame_set, threads=os.cpu_count(), bitrate="5000k")
