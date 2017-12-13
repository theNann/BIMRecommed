from scikits.crab.models import MatrixPreferenceDataModel
from scikits.crab.metrics import pearson_correlation
from scikits.crab.similarities import UserSimilarity
from scikits.crab.recommenders.knn import UserBasedRecommender


def recommend():
    movies_data = {
        1: {1: 3.0, 2: 4.0, 3: 3.5, 4: 5.0, 5: 3.0},
        2: {1: 3.0, 2: 4.0, 3: 2.0, 4: 3.0, 5: 3.0, 6: 2.0},
        3: {2: 3.5, 3: 2.5, 4: 4.0, 5: 4.5, 6: 3.0},
        4: {1: 2.5, 2: 3.5, 3: 2.5, 4: 3.5, 5: 3.0, 6: 3.0},
        5: {2: 4.5, 3: 1.0, 4: 4.0},
        6: {1: 3.0, 2: 3.5, 3: 3.5, 4: 5.0, 5: 3.0, 6: 1.5},
        7: {1: 2.5, 2: 3.0, 4: 3.5, 5: 4.0}
    }
    model = MatrixPreferenceDataModel(movies_data)
    print(model)
    pass


if __name__ == '__main__':
    recommend()
