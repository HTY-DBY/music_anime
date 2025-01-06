import os


class GobleD:
	def __init__(self, ):
		self.main_path = r"D:\hty\creat\code\python\music_anime"
		self.ffmpeg_ins_path = os.path.join(self.main_path, r"ffmpeg")
		self.music_down_path = os.path.join(self.main_path, r"Music")
		self.temp_fig_path = os.path.join(self.main_path, r"temp_fig_path")
		self.person_img_path = "D:\hty\medi\常用\初音.gif"


if __name__ == '__main__':
	# %%
	# 初始化对象
	GobleD = GobleD()
	pass
