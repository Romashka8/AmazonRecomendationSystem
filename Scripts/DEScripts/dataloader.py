import pandas as pd
import numpy as np

import os
import json
import pickle
import warnings
from tqdm import tqdm
from typing import Optional

warnings.filterwarnings('ignore')


class AmazonReviewsLoader:
	def __init__(self, in_dir: str, sample_size: int=100, seed: int=1):
		
		if isinstance(in_dir, str) and os.path.isfile(in_dir):
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
			data = []
			for index in tqdm(self.orig_indexes, desc="Загрузка данных"):
				row = json.loads(self._data[index])
				del row['images']
				data.append(row)

			self.sample = pd.json_normalize(data)

		if reset_index:
			self.sample.reset_index()

		return self.sample

	def get_all(self) -> pd.DataFrame:
		
		self.seed[1] = True
		data = []
		for index in tqdm(range(self._len), desc="Загрузка данных"):
			row = json.loads(self._data[index])
			del row['images']
			data.append(row)

		self.sample = pd.json_normalize(data)
		self.sample_size = self._len

		return self.sample

	def save_sample(self, out_dir: str, f_format: str) -> None:

		if isinstance(out_dir, str) and isinstance(f_format, str)\
			and f_format in ('csv', 'xlsx') and os.path.exists(out_dir):
			self.out_dir = out_dir
		else:
			raise ValueError(f'Выходная директория должна быть строкой следуещего вида: "{os.path.getcwd()}"! Формат файла должен быть либо "csv", либо "xlsx"!')

		filename = f'sample_size_{self.sample_size}_seed_{self.seed[0]}_used_{self.seed[1]}.{f_format}'
		bin_filename = f'indexes_size_{self.sample_size}_seed_{self.seed[0]}_used_{self.seed[1]}'
		out = os.path.join(self.out_dir, filename)
		bin_out = os.path.join(self.out_dir, bin_filename)

		if not os.path.exists(out):
			if f_format == 'csv':
				self.sample.to_csv(out)
			else:
				self.sample.to_excel(out)
			with open(f'{bin_out}.pickle', 'wb') as f:
				pickle.dump(self.orig_indexes, f)
			print('Данные успешно сохранены')
		else:
			print('Данные уже существуют')
		return

	def reset_seed(self, new_seed: int) -> None:

		if isinstance(new_seed, int) and 0 <= new_seed <= 2 ** 32 - 1:
			# (seed: int, used_in_sample_creation)
			self.seed = [new_seed, False]
			np.random.seed(self.seed[0])
			self.orig_indexes = np.random.randint(0, self._len, size=self.sample_size)
			print('Семя успешно изменено! Для создания новой выборки на его основе нужно заново вызвать метод с следующими параметрами: get_sample(create_new=True)')
		else:
			raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')
		return
