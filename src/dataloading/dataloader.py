import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import os
import json
import pickle
from tqdm import tqdm
from typing import Optional


class AmazonReviewsLoader:
    def __init__(self, in_dir: str, sample_size: int = 100, seed: int = 1):
        """
        Конструктор класса AmazonReviewsLoader.

        :param in_dir: Входная директория с данными.
        :param sample_size: Размер выборки данных (по умолчанию 100).
        :param seed: Семя для генерации случайных чисел (по умолчанию 1).
        """

        # Проверка, что входная директория является строкой и существует
        if isinstance(in_dir, str) and os.path.isfile(in_dir):
            self.in_dir = in_dir
        else:
            raise ValueError(
                f'Входная директория должна быть строкой следуещего вида: "{os.getcwd()}"!'
            )

        with open(in_dir, 'r') as f:
            # Чтение всех строк из файла
            self._data = f.readlines()
            # Определение длины данных
            self._len = len(self._data)

        # Проверка, что размер выборки является допустимым целым числом
        if isinstance(sample_size, int) and 0 < sample_size <= self._len:
            self.sample_size = sample_size
        else:
            raise ValueError(
                f'Размер выборки должен быть целым числом, удовлетворяющим неравенству: "0 < sample_size <= {self._len}"!'
            )

        # Проверка, что семя является допустимым целым числом
        if isinstance(seed, int) and 0 <= seed <= 2 ** 32 - 1:
            # (seed: int, used_in_sample_creation)
            self.seed = [seed, False]
            np.random.seed(self.seed[0])
        else:
            raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')            

        self.out_dir = None
        # Генерация случайных индексов для выборки
        self.orig_indexes = np.random.randint(0, self._len, size=self.sample_size)
        self.sample = None

    def __repr__(self):
        output = {
                'dtype' : str(type(self)),
                'in_dir' : self.in_dir,
                'out_dir' : self.out_dir,
                'sample_size' : self.sample_size,
                'seed' : self.seed
            }
        return str(output)

    def get_sample(self):
        """
        Предполагается реализация в наследниках
        """
        pass

    def get_all(self):
        """
        Предполагается реализация в наследниках
        """
        pass

    def save_sample(self, out_dir: str, f_format: str) -> None:
        """
        Сохраняет выборку данных в указанную директорию.

        :param out_dir: Выходная директория для сохранения данных.
        :param f_format: Формат файла для сохранения (csv или xlsx).
        """

        # Проверка корректности выходных параметров
        if isinstance(out_dir, str) and isinstance(f_format, str) \
           and f_format in ('csv', 'xlsx') and os.path.exists(out_dir):
            self.out_dir = out_dir
        else:
            raise ValueError(
                f'Выходная директория должна быть строкой следуещего вида: "{os.getcwd()}"! Формат файла должен быть либо "csv", либо "xlsx"!'
            )

        # Формируем имя файла
        filename = f'sample_size_{self.sample_size}_seed_{self.seed[0]}_used_{self.seed[1]}.{f_format}'
        bin_filename = f'indexes_size_{self.sample_size}_seed_{self.seed[0]}_used_{self.seed[1]}'
        out = os.path.join(self.out_dir, filename)
        bin_out = os.path.join(self.out_dir, bin_filename)

        if not os.path.exists(out):
            # Сохранение данных в зависимости от формата
            if f_format == 'csv':
                self.sample.to_csv(out)
            else:
                self.sample.to_excel(out)
            # Сохранение индексов выборки
            with open(f'{bin_out}.pickle', 'wb') as f:
                pickle.dump(self.orig_indexes, f)
            print('Данные успешно сохранены')
        else:
            print('Данные уже существуют')

    def reset_seed(self, new_seed: int) -> None:
        """
        Сбрасывает семя для генерации случайных чисел.

        :param new_seed: Новое семя для генерации случайных чисел.
        """

        # Проверка корректности нового семени
        if isinstance(new_seed, int) and 0 <= new_seed <= 2 ** 32 - 1:
            # (seed: int, used_in_sample_creation)
            self.seed = [new_seed, False]
            np.random.seed(self.seed[0])
            # Перегенерация случайных индексов
            self.orig_indexes = np.random.randint(0, self._len, size=self.sample_size)
            print('Семя успешно изменено! Для создания новой выборки на его основе нужно заново вызвать метод с следующими параметрами: get_sample(create_new=True)')
        else:
            raise ValueError('Семя случайной генерации должно лежать в диапазоне: "0 < seed <= 2 ** 32 - 1"!')


class AllBeautyLoader(AmazonReviewsLoader):
    def get_sample(self, create_new: bool = False, reset_index: bool = False) -> pd.DataFrame:
        """
        Получает выборку данных.

        :param create_new: Флаг для создания новой выборки (по умолчанию False).
        :param reset_index: Флаг для сброса индекса DataFrame (по умолчанию False).
        :return: DataFrame с выборкой данных.
        """

        if self.sample is None or create_new:
            self.seed[1] = True
            data = []
            # Загрузка данных по случайным индексам
            for index in tqdm(self.orig_indexes, desc="Загрузка данных"):
                row = json.loads(self._data[index])
                # Удаление поля 'images'
                del row['images']
                data.append(row)

            # Преобразование данных в DataFrame
            self.sample = pd.json_normalize(data).set_index(self.orig_indexes)

        if reset_index:
            # Сброс индекса DataFrame
            self.sample = self.sample.reset_index()

        return self.sample

    def get_all(self) -> pd.DataFrame:
        """
        Получает все данные.

        :return: DataFrame со всеми данными.
        """

        self.seed[1] = True
        data = []
        # Загрузка всех данных
        for index in tqdm(range(self._len), desc="Загрузка данных"):
            row = json.loads(self._data[index])
            # Удаление поля 'images'
            del row['images']
            data.append(row)

        # Преобразование данных в DataFrame
        self.sample = pd.json_normalize(data)
        self.sample_size = self._len

        return self.sample


class AllBeautyLoaderMeta(AmazonReviewsLoader):
    def get_sample(self, create_new: bool = False, reset_index: bool = False) -> pd.DataFrame:
        """
        Получает выборку данных.

        :param create_new: Флаг для создания новой выборки (по умолчанию False).
        :param reset_index: Флаг для сброса индекса DataFrame (по умолчанию False).
        :return: DataFrame с выборкой данных.
        """

        if self.sample is None or create_new:
            self.seed[1] = True
            data = []
            # Загрузка данных по случайным индексам
            for index in tqdm(self.orig_indexes, desc="Загрузка данных"):
                row = json.loads(self._data[index])
                # Удаление поля 'images', 'videos'
                del row['images'], row['videos']
                data.append(row)

            # Преобразование данных в DataFrame
            self.sample = pd.json_normalize(data).set_index(self.orig_indexes)

        if reset_index:
            # Сброс индекса DataFrame
            self.sample = self.sample.reset_index()

        return self.sample

    def get_all(self) -> pd.DataFrame:
        """
        Получает все данные.

        :return: DataFrame со всеми данными.
        """

        self.seed[1] = True
        data = []
        # Загрузка всех данных
        for index in tqdm(range(self._len), desc="Загрузка данных"):
            row = json.loads(self._data[index])
            # Удаление поля 'images', 'videos'
            del row['images'], row['videos'], row['features'], row['description'], row['price'], row['bought_together'], row['details'], row['categories'], row['main_category']
            data.append(row)

        # Преобразование данных в DataFrame
        self.sample = pd.json_normalize(data)
        self.sample_size = self._len

        return self.sample
