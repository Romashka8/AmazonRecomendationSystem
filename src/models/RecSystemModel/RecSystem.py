import warnings
warnings.filterwarnings('ignore')

import os

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import KNN


class AmazonRecSystem:
    def __init__(self, data):
        
        """
        Инициализация рекомендательной системы
        :param data: DataFrame с данными пользователей
        """
        
        self.data = data
        self.n_users, self.n_items = data.shape
        data_num, data_obj = self.data.select_dtypes(exclude='object'), self.data.select_dtypes(include='object')
        self.m, self.s = np.mean(data_num, axis=0), np.std(data_num, axis=0)
        data_num = (data_num - self.m) / self.s
        self.data = pd.concat([data_obj, data_num], axis=1)
        self.knn = KNN.load_knn('fitted_knn/knn_embedded.pkl')
        
    def select_candidates(self, user_id, strategy='unseen', n=100):
        
        """
        Отбор кандидатов для рекомендации
        
        :param user_id: ID целевого пользователя
        :param strategy: стратегия отбора ('unseen', 'popular', 'random', 'knn')
        :param n: количество возвращаемых кандидатов
        :return: список индексов отобранных кандидатов
        """
        
        if strategy == 'unseen':
            # Выбор непросмотренных пользователем объектов(asin товара)
            interacted = self.data[self.data['user_id'] == user_id]['asin'].index.tolist()
            candidates = self.data.drop(interacted, axis=0).index.tolist()
            
        elif strategy == 'popular':
            # Выбор самых популярных объектов(helpful_vote)
            candidates = self.data.sort_values(by='helpful_vote', ascending=False).index.tolist()
            
        elif strategy == 'random':
            # Случайный выбор объектов
            candidates = self.data.index.tolist()
            np.random.shuffle(candidates)

        elif strategy == 'knn':
            # Ищем k похожих пользователей
            distances, indices = self.knn.kneighbors(
                self.data[self.data['user_id'] == user_id].drop(['user_id', 'asin'], axis=1).values.reshape(1, -1), 
                n_neighbors=n+1
            )
            similar_users = self.data.index[indices[0][1:]].tolist()  # Исключаем самого пользователя
            drop_asin = self.data[self.data['user_id'] == user_id].index.tolist()
            candidates = [user for user in similar_users if user not in drop_asin]
            
        return candidates[:n]

    def calculate_similarity(self, target_id, candidate_ids, mode='user'):
        """
        Расчет сходства между пользователями
        
        :param target_id: ID целевого пользователя
        :param candidate_ids: список ID кандидатов
        :param mode: режим расчета, emb добавляет эмбэдинги отзывов ('user' или 'emb')
        :return: словарь с оценками сходства
        """
        similarities = {}
        
        # Косинусное сходство между пользователями
        target_id = self.data[self.data['user_id'] == target_id].index.tolist()[0]
        target_vector = self.data.drop(['user_id', 'asin'], axis=1).loc[target_id].values.reshape(1, -1)
        for candidate_id in candidate_ids:
            candidate_vector = self.data.drop(['user_id', 'asin'], axis=1).loc[candidate_id].values.reshape(1, -1)
            similarities[candidate_id] = cosine_similarity(target_vector, candidate_vector)[0][0]
        
        return similarities

    def recommend(self, user_id, n_recommend=5, strategy='unseen'):
        """
        Генерация рекомендаций
        
        :param user_id: ID пользователя
        :param n_recommend: количество рекомендаций
        :param strategy: стратегия отбора кандидатов
        :param mode: режим расчета сходства
        :return: список рекомендованных ID объектов
        """
        # Отбор кандидатов
        candidates = self.select_candidates(user_id, strategy=strategy)
        
        # Расчет сходства
        similarities = self.calculate_similarity(user_id, candidates)
        
        # Сортировка и выбор топ-N
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:n_recommend]]


if __name__ == '__main__':

    data = pd.read_csv('/home/roman/Документы/AmazonRecomendationSystem/data/interim/reviews_final.csv').drop(['Unnamed: 0'], axis=1)
    goods = pd.read_csv('/home/roman/Документы/AmazonRecomendationSystem/data/interim/goods_for_reviews.csv').drop(['Unnamed: 0'], axis=1)
    
    test = AmazonRecSystem(data)

    users_idx = ['AHFYD2BAJG7VV76FMGOPGPWAXN4Q', 'AFGMBVIA3XFIKNP7S2HRCKGJPXKA', 'AG56FD5DU43FVRNNCNDOAFW7DTGQ']

    for user_id in users_idx:
    
        recs = test.recommend(user_id, n_recommend=10, strategy='knn')
        good_idx = set(data.iloc[recs]['asin'].values.tolist())
        goods_idx = [goods[goods['asin'] == asin].index[0] for asin in good_idx]
        print('-' * 100)
        print(f'Recomendations for user {user_id}: \n')
        print(goods.iloc[goods_idx])
        print('\n')