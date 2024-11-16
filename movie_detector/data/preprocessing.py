import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from pandas import DataFrame
from typing import Union, List
import warnings


def process(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_processor(text: Union[str, List[str], DataFrame], cols: List[str] = None) -> Union[str, List[str], DataFrame]:
    if isinstance(text, list):
        text = [process(t) for t in text]
    if isinstance(text, str):
        text = [process(text)]
    if cols and isinstance(text, DataFrame):
        for col in cols:
            text[col] = text[col].str.strip().str.replace(r"\s+", " ", regex=True)
    return text


def set_genres(df: DataFrame):
    if 'genre' not in df.columns:
        warnings.warn('Genres not found in DataFrame', UserWarning)
    else:
        return pd.concat([df,df['genre'].str.lower().str.get_dummies(sep=', ')], axis=1).drop(columns=['genre'])


def set_title_embeddings(df: DataFrame):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(
        df['title'].tolist(),
        batch_size=32,
        show_progress_bar=True
    )
    embeddings_df = pd.DataFrame(
        embeddings,
        columns=[f'embedding_{i + 1}' for i in range(embeddings.shape[1])]
    )
    return pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)