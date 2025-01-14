import pandas as pd
import numpy as np

import os
import time
import json
import warnings
from tqdm import tqdm
from typing import Optional

warnings.filterwarnings('ignore')


class AmazonReviewsLoader:
	def __init__(self, in_dir: str, out_dir: Optional[str]=None, sample_size: int=100, seed: int=1):
		
		if isinstanse(in_dir, str) and os.path.exists(in_dir):
			self.in_dir = in_dir
		else:
			raise ValueError(f'Входная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"!')

		# if isinstanse(out_dir, str) and os.path.exists(out_dir):
		# 	self.out_dir = out_dir
		# else:
		# 	raise ValueError(f'Выходная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"!')
		
		with open(in_dir, 'r') as f:
			self._data = f.readlines()
			self._len = len(self._data)

		if isinstanse(sample_size, int) and 0 < size <= self._len:
			self.sample_size = sample_size
		else:
			raise ValueError(f'Размер выборки должен быть целым числом, удовлетворяющим неравенству: "0 < sample_size <= {self._len}"!')

		if isinstanse(seed, int) and 0 <= seed <= 2 ** 32 - 1:
			self.seed = seed
		else:
			raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')

		self.orig_indexes = np.random.randint(0, self._len, size=self.size)

	def get_sample(self):
		pass

	def save_sample(self):
		pass

	def reset_index(self):
		pass

	def reset_seed(self):
		pass
