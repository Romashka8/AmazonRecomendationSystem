import pandas as pd
import numpy as np

import os
import json
import warnings
from tqdm import tqdm
from typing import Optional

warnings.filterwarnings('ignore')


class AmazonReviewsLoader:
	def __init__(self, in_dir: str, sample_size: int=100, seed: int=1):
		
		if isinstance(in_dir, str) and os.path.exists(in_dir):
			self.in_dir = in_dir
		else:
			raise ValueError(f'Входная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"!')
		
		with open(in_dir, 'r') as f:
			self._data = f.readlines()
			self._len = len(self._data)

		if isinstance(sample_size, int) and 0 < sample_size <= self._len:
			self.sample_size = sample_size
		else:
			raise ValueError(f'Размер выборки должен быть целым числом, удовлетворяющим неравенству: "0 < sample_size <= {self._len}"!')

		if isinstance(seed, int) and 0 <= seed <= 2 ** 32 - 1:
			# (seed: int, used_in_sample_creation)
			self.seed = [seed, False]
			np.random.seed(self.seed[0])
		else:
			raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')

		self.out_dir = None
		self.orig_indexes = np.random.randint(0, self._len, size=self.sample_size)
		self.sample = None

	def get_sample(self, create_new: bool=False, reset_index: bool=False) -> pd.DataFrame:
		if self.sample is None or create_new:
			self.seed[1] = True
			first_row = json.loads(self._data[int(self.orig_indexes[0])])
			first_index = int(self.orig_indexes[0])
			# удаляем значение по ключу 'images' - оно нам не пригодится
			del first_row['images']
			self.sample = pd.DataFrame(first_row, index=[first_index])
			for index in tqdm(self.orig_indexes[1:]):
				load_row = json.loads(self._data[index])
				del load_row['images']
				raw = pd.DataFrame(load_row, index=[index])
				self.sample = pd.concat((self.sample, raw))
		if reset_index:
			self.reset_index()
		return self.sample

	def get_all(self) -> pd.DataFrame:
		first_row = json.loads(self._data[int(self.orig_indexes[0])])
		first_index = int(self.orig_indexes[0])
		# удаляем значение по ключу 'images' - оно нам не пригодится
		del first_row['images']
		self.sample = pd.DataFrame(first_row, index=[first_index])
		for index in tqdm(range(1, self._len)):
			load_row = json.loads(self._data[index])
			del load_row['images']
			raw = pd.DataFrame(load_row, index=[index])
			self.sample = pd.concat((self.sample, raw))
		self.sample_size = self._len
		return self.sample

	def save_sample(self, out_dir: str, f_format: str) -> None:

		if isinstance(out_dir, str) and os.path.exists(out_dir):
			self.out_dir = out_dir
		else:
			raise ValueError(f'Выходная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"!')

		if not(isinstance(f_format, str) and f_format in ('csv', 'xlsx')):
			raise ValueError(f'Выходная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"!')

		filename = f'sample_size_{self.sample_size}_seed_{self.seed[0]}_used_{self.seed[1]}.{f_format}'
		out = os.path.join(self.out_dir, filename)

		if not os.path.exists(out):
			if f_format == 'csv':
				self.sample.to_csv(out)
			else:
				self.sample.to_excel(out)
			print('Данные успешно сохранены')
		else:
			print('Данные уже существуют')
		return


	def reset_index(self) -> None:
		self.sample.reset_index()
		return

	def reset_seed(self, new_seed: int) -> None:
		if isinstance(new_seed, int) and 0 <= new_seed <= 2 ** 32 - 1:
			# (seed: int, used_in_sample_creation)
			self.seed = [new_seed, False]
			np.random.seed(self.seed[0])
			self.orig_indexes = np.random.randint(0, self._len, size=self.sample_size)
		else:
			raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')
		return
