import pandas as pd


class MovieData:
    def __init__(self, path: str):
        self.path = path
        self.data = None

    def load_dataset(self):
        df = (
            pd
            .read_csv(self.path, dtype={'id': str, 'title': str, 'language': str, 'genre': str, 'year': str})
            .query("language.notna()")
            .query('language.str.contains("English")')
        )[['title', 'genre']]
        self.data = pd.concat(
            [df, df['genre'].str.lower().str.get_dummies(sep=', ')],
            axis=1
        ).drop(columns=['genre'])