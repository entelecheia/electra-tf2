from .util.timer import elapsed_timer
from omegaconf import OmegaConf


class TFRecords:
	def __init__(self, **kwargs):
		self.args = OmegaConf.create(kwargs)

	def run(self):
		print(self.args)
		from electra.dataset.build_pretraining_dataset import hydra_run
		with elapsed_timer(format_time=True) as elapsed:
			hydra_run(self.args)
			print(f"\n >>> Elapsed time: {elapsed()} <<< ")


class PreTrain:
	def __init__(self, **kwargs):
		self.args = OmegaConf.create(kwargs)

	def run(self):
		print(self.args)