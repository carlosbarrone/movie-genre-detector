import pandas as pd
import awswrangler as wr

class MovieData:
    def __init__(self, path: str = None, payload: dict = None, english_only=True) -> None:
        """
        Initialize the MovieData class.

        :param path: Path to the data csv file with movies and genres.
        :param payload: Dictionary from JSON payload for inference and predictions.
        :param english_only: Filter OUT movies that do not include English as language.
        Only one of 'path' or 'payload' should be provided for loading the dataset.
        """
        self.path = path
        self.payload = payload
        self.english_only = english_only
        self.data = None
        self.load_dataset()

    def load_dataset(self):
        if self.path and self.payload:
            raise ValueError("Only one of 'path' or 'payload' should be provided.")
        if self.path:
            if self.path.startswith('s3://'):
                self.data = wr.s3.read_csv(self.path, dtype={'id': str, 'title': str, 'language': str, 'genre': str, 'year': str})
            else:
                self.data = pd.read_csv(self.path, dtype={'id': str, 'title': str, 'language': str, 'genre': str, 'year': str})
        elif self.payload:
            self.payload['genres'] = [', '.join(genre) for genre in self.payload['genres']]
            self.data = pd.DataFrame(self.payload).rename(columns={'titles': 'title', 'genres': 'genre'})
        self.data = self.data.query("language.notna()")
        if self.english_only:
            self.data = self.data.query('language.str.contains("English")')