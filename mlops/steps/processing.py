import sys
import pandas as pd
from movie_detector.data.load import MovieData
from movie_detector.data.preprocessing import text_processor, set_genres, set_title_embeddings
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dataset = MovieData('/opt/ml/processing/input/imdb_movies.csv', english_only = False)
    print('Dataset Loaded.')
    dataset.data = text_processor(dataset.data, cols=['title', 'original_title'])
    dataset.data = dataset.data[['title', 'genre']]
    dataset.data = set_genres(dataset.data)
    print('Preprocessing done.')
    dataset.data = set_title_embeddings(dataset.data)
    print('Embeddings done.')
    X, y = (
        dataset.data[[c for c in dataset.data if c.startswith('embedding')]],
        dataset.data[[c for c in dataset.data if not c.startswith('embedding')]]
    )

    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    print('Data split done.')
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    y_train_df = pd.DataFrame(y_train, columns=y.columns)
    X_val_df = pd.DataFrame(X_val, columns=X.columns)
    y_val_df = pd.DataFrame(y_val, columns=y.columns)

    X_train_df.to_csv('/opt/ml/processing/output/train/X_train.csv', index=False)
    y_train_df.to_csv('/opt/ml/processing/output/train/y_train.csv', index=False)
    X_val_df.to_csv('/opt/ml/processing/output/validation/X_val.csv', index=False)
    y_val_df.to_csv('/opt/ml/processing/output/validation/y_val.csv', index=False)
    print('Files written.')
    print('Processing done.')
    sys.exit(0)