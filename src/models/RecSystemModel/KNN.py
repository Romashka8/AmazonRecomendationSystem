import pickle
from sklearn.neighbors import NearestNeighbors


def load_knn(load_path):

    with open(load_path, 'rb') as f:
        model = pickle.load(f)

    return model


class TrainNearestNeighbors:

    def __init__(self, data, train_columns, save_path='knn_model.pkl'):

        self.model = self._train(data[train_columns])
        self._save_trained(save_path)
    
    def _train(self, X, metric='cosine'):
        
        knn = NearestNeighbors(metric=metric)
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0))
        knn.fit(X)

        return knn

    def _save_trained(self, save_path):

        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')

    import pandas as pd
    import numpy as np


    MAIN_PATH = '/home/roman/Документы/AmazonRecomendationSystem/data/interim'

    cols = [
        'rating', 'helpful_vote', 'verified_purchase',
        'average_rating', 'rating_number', 'tonality'
    ]

    emb_cols = cols + [f'embeded_feature_{i}' for i in range(768)]
    
    reviews = pd.read_csv(MAIN_PATH + '/reviews_final.csv').drop(['user_id', 'asin'], axis=1)

    knn_embedded = TrainNearestNeighbors(reviews, emb_cols, 'fitted_knn/knn_embedded.pkl')