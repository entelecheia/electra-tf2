from timer import elapsed_timer
from omegaconf import OmegaConf


class TFRecords:
	def __init__(self, **kwargs):
		self.args = OmegaConf.create(kwargs)

	def run(self):
		print(self.args)
		# from create_tfrecords import create_tfrecords
		# with elapsed_timer(format_time=True) as elapsed:
		# 	create_tfrecords(self.args)
		# 	print(f"\n >>> Elapsed time: {elapsed()} <<< ")


class PreTrain:
	def __init__(self, **kwargs):
		self.args = OmegaConf.create(kwargs)

	def run(self):
		print(self.args)