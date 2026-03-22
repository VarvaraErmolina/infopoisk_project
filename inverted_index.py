"""
Модуль для построения обратных индексов и поиска по корпусу.

Поддерживаются индексы:
    - frequency
    - bm25
    - word2vec
    - fasttext

Ожидается, что на вход подается CSV-файл с колонкой `preprocessed_text`,
полученной после предобработки корпуса.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing_data import PreprocessingConfig, RussianTextPreprocessor


@dataclass(slots=True)
class SearchConfig:
    text_column: str = "preprocessed_text"
    k1: float = 1.5
    b: float = 0.75

    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    workers: int = 4
    sg: int = 0


class QueryPreprocessor:
    def __init__(self, config: SearchConfig | None = None) -> None:
        preprocessing_config = PreprocessingConfig(
            text_column="text",
            language="russian",
            min_token_length=2,
            drop_digits=True,
            lowercase=True,
        )
        self.preprocessor = RussianTextPreprocessor(preprocessing_config)
        self.config = config or SearchConfig()

    def preprocess(self, query: str) -> list[str]:
        result = self.preprocessor.preprocess_text(query)
        return result["lemmas"]


class CorpusReader:
    @staticmethod
    def read_csv(input_path: str | Path) -> pd.DataFrame:
        return pd.read_csv(input_path)


class LibraryFrequencyInvertedIndex:
    """Частотный обратный индекс на базе CountVectorizer"""

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        self.term_document_matrix = None
        self.feature_names: np.ndarray | None = None
        self.inverted_index: dict[str, dict[int, int]] = {}

    def build(self, documents: list[str]) -> None:
        self.term_document_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.inverted_index = self._build_inverted_index_from_matrix()

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        scores: defaultdict[int, float] = defaultdict(float)

        for token in query_tokens:
            postings = self.inverted_index.get(token, {})
            for doc_id, frequency in postings.items():
                scores[doc_id] += frequency

        return dict(scores)

    def _build_inverted_index_from_matrix(self) -> dict[str, dict[int, int]]:
        inverted_index: dict[str, dict[int, int]] = {}

        if self.term_document_matrix is None or self.feature_names is None:
            return inverted_index

        matrix_csc = self.term_document_matrix.tocsc()

        for term_index, term in enumerate(self.feature_names):
            column = matrix_csc[:, term_index]
            doc_ids = column.nonzero()[0]
            frequencies = column.data
            inverted_index[term] = {
                int(doc_id): int(frequency)
                for doc_id, frequency in zip(doc_ids, frequencies)
            }

        return inverted_index


class LibraryBM25InvertedIndex:
    """BM25-индекс на базе rank_bm25"""

    def __init__(self) -> None:
        self.tokenized_documents: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.inverted_index: dict[str, dict[int, int]] = {}

    def build(self, tokenized_documents: list[list[str]]) -> None:
        self.tokenized_documents = tokenized_documents
        self.bm25 = BM25Okapi(tokenized_documents)
        self.inverted_index = self._build_inverted_index(tokenized_documents)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        if self.bm25 is None:
            raise ValueError("Индекс BM25 еще не построен.")

        scores_array = self.bm25.get_scores(query_tokens)
        return {
            int(doc_id): float(score)
            for doc_id, score in enumerate(scores_array)
            if score > 0
        }

    @staticmethod
    def _build_inverted_index(
            tokenized_documents: list[list[str]],
    ) -> dict[str, dict[int, int]]:
        inverted_index: dict[str, dict[int, int]] = defaultdict(dict)

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens)
            for term, frequency in token_counts.items():
                inverted_index[term][doc_id] = frequency

        return dict(inverted_index)


class Word2VecSemanticIndex:
    """Семантический индекс на основе Word2Vec"""

    def __init__(
            self,
            vector_size: int = 100,
            window: int = 5,
            min_count: int = 1,
            workers: int = 4,
            sg: int = 0,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg

        self.model: Word2Vec | None = None
        self.document_vectors: np.ndarray | None = None
        self.tokenized_documents: list[list[str]] = []

    def build(self, tokenized_documents: list[list[str]]) -> None:
        self.tokenized_documents = tokenized_documents

        if not tokenized_documents:
            self.document_vectors = np.empty((0, self.vector_size), dtype=np.float64)
            return

        self.model = Word2Vec(
            sentences=tokenized_documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
        )

        vectors = [self._get_text_vector(tokens) for tokens in tokenized_documents]
        self.document_vectors = np.vstack(vectors)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        if self.model is None or self.document_vectors is None:
            raise ValueError("Word2Vec-индекс еще не построен.")

        if self.document_vectors.shape[0] == 0:
            return {}

        query_vector = self._get_text_vector(query_tokens)
        if np.allclose(query_vector, 0):
            return {}

        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            self.document_vectors,
        )[0]

        return {
            int(doc_id): float(score)
            for doc_id, score in enumerate(similarities)
            if score > 0
        }

    def _get_text_vector(self, tokens: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros(self.vector_size, dtype=np.float64)

        word_vectors = [
            self.model.wv[token]
            for token in tokens
            if token in self.model.wv
        ]

        if not word_vectors:
            return np.zeros(self.vector_size, dtype=np.float64)

        return np.mean(word_vectors, axis=0)


class FastTextSemanticIndex:
    """Семантический индекс на основе FastText"""

    def __init__(
            self,
            vector_size: int = 100,
            window: int = 5,
            min_count: int = 1,
            workers: int = 4,
            sg: int = 0,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg

        self.model: FastText | None = None
        self.document_vectors: np.ndarray | None = None
        self.tokenized_documents: list[list[str]] = []

    def build(self, tokenized_documents: list[list[str]]) -> None:
        self.tokenized_documents = tokenized_documents

        if not tokenized_documents:
            self.document_vectors = np.empty((0, self.vector_size), dtype=np.float64)
            return

        self.model = FastText(
            sentences=tokenized_documents,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
        )

        vectors = [self._get_text_vector(tokens) for tokens in tokenized_documents]
        self.document_vectors = np.vstack(vectors)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        if self.model is None or self.document_vectors is None:
            raise ValueError("FastText-индекс еще не построен.")

        if self.document_vectors.shape[0] == 0:
            return {}

        query_vector = self._get_text_vector(query_tokens)
        if np.allclose(query_vector, 0):
            return {}

        similarities = cosine_similarity(
            query_vector.reshape(1, -1),
            self.document_vectors,
        )[0]

        return {
            int(doc_id): float(score)
            for doc_id, score in enumerate(similarities)
            if score > 0
        }

    def _get_text_vector(self, tokens: list[str]) -> np.ndarray:
        if self.model is None:
            return np.zeros(self.vector_size, dtype=np.float64)

        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float64)

        word_vectors = [self.model.wv[token] for token in tokens]

        if not word_vectors:
            return np.zeros(self.vector_size, dtype=np.float64)

        return np.mean(word_vectors, axis=0)


class ManualFrequencyInvertedIndex:
    """Ручной частотный индекс"""

    def __init__(self) -> None:
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)

    def build(self, tokenized_documents: list[list[str]]) -> None:
        self.inverted_index = defaultdict(dict)

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens)
            for term, frequency in token_counts.items():
                self.inverted_index[term][doc_id] = frequency

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        scores: defaultdict[int, float] = defaultdict(float)

        for term in query_tokens:
            postings = self.inverted_index.get(term, {})
            for doc_id, frequency in postings.items():
                scores[doc_id] += frequency

        return dict(scores)


class ManualBM25InvertedIndex:
    """Ручной BM25-индекс"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        self.document_lengths: dict[int, int] = {}
        self.document_frequencies: dict[str, int] = {}
        self.documents_count: int = 0
        self.average_document_length: float = 0.0

    def build(self, tokenized_documents: list[list[str]]) -> None:
        self.inverted_index = defaultdict(dict)
        self.document_lengths = {}
        self.document_frequencies = {}
        self.documents_count = len(tokenized_documents)

        total_length = 0

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts = Counter(tokens)
            document_length = len(tokens)
            self.document_lengths[doc_id] = document_length
            total_length += document_length

            for term, frequency in token_counts.items():
                self.inverted_index[term][doc_id] = frequency

        self.average_document_length = (
            total_length / self.documents_count if self.documents_count > 0 else 0.0
        )

        for term, postings in self.inverted_index.items():
            self.document_frequencies[term] = len(postings)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        scores: defaultdict[int, float] = defaultdict(float)

        for term in query_tokens:
            postings = self.inverted_index.get(term, {})
            if not postings:
                continue

            document_frequency = self.document_frequencies[term]
            idf = self._compute_idf(document_frequency)

            for doc_id, term_frequency in postings.items():
                document_length = self.document_lengths[doc_id]
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                        1 - self.b + self.b * document_length / self.average_document_length
                )
                scores[doc_id] += idf * numerator / denominator

        return dict(scores)

    def _compute_idf(self, document_frequency: int) -> float:
        numerator = self.documents_count - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log(1 + numerator / denominator)


class SearchEngine:
    """Поисковый движок"""

    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.query_preprocessor = QueryPreprocessor(self.config)

        self.frequency_index = LibraryFrequencyInvertedIndex()
        self.bm25_index = LibraryBM25InvertedIndex()
        self.word2vec_index = Word2VecSemanticIndex(
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.workers,
            sg=self.config.sg,
        )
        self.fasttext_index = FastTextSemanticIndex(
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.workers,
            sg=self.config.sg,
        )

        self.dataframe: pd.DataFrame | None = None
        self.documents_as_strings: list[str] = []
        self.documents_as_tokens: list[list[str]] = []

    def fit(self, dataframe: pd.DataFrame) -> None:
        self._validate_dataframe(dataframe)

        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.documents_as_strings = self._extract_documents_as_strings(self.dataframe)
        self.documents_as_tokens = [text.split() for text in self.documents_as_strings]

        self.frequency_index.build(self.documents_as_strings)
        self.bm25_index.build(self.documents_as_tokens)
        self.word2vec_index.build(self.documents_as_tokens)
        self.fasttext_index.build(self.documents_as_tokens)

    def search(
            self,
            query: str,
            index_type: str = "bm25",
            top_k: int = 10,
    ) -> pd.DataFrame:
        if self.dataframe is None:
            raise ValueError("Сначала нужно вызвать fit() и построить индексы.")

        start_time = time.perf_counter()

        query_tokens = self.query_preprocessor.preprocess(query)

        if not query_tokens:
            return pd.DataFrame(
                columns=["doc_id", "score", "search_time_seconds", *self.dataframe.columns]
            )

        if index_type == "frequency":
            scores = self.frequency_index.search(query_tokens)
        elif index_type == "bm25":
            scores = self.bm25_index.search(query_tokens)
        elif index_type == "word2vec":
            scores = self.word2vec_index.search(query_tokens)
        elif index_type == "fasttext":
            scores = self.fasttext_index.search(query_tokens)
        else:
            raise ValueError(
                "index_type должен быть 'frequency', 'bm25', 'word2vec' или 'fasttext'."
            )

        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        search_time_seconds = time.perf_counter() - start_time

        if not ranked_results:
            return pd.DataFrame(
                columns=["doc_id", "score", "search_time_seconds", *self.dataframe.columns]
            )

        doc_ids = [doc_id for doc_id, _ in ranked_results]
        doc_scores = [score for _, score in ranked_results]

        result = self.dataframe.iloc[doc_ids].copy()
        result.insert(0, "doc_id", doc_ids)
        result.insert(1, "score", doc_scores)
        result.insert(2, "search_time_seconds", search_time_seconds)

        return result

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        if self.config.text_column not in dataframe.columns:
            raise ValueError(
                f"Колонка '{self.config.text_column}' не найдена во входном DataFrame."
            )

    def _extract_documents_as_strings(self, dataframe: pd.DataFrame) -> list[str]:
        return dataframe[self.config.text_column].fillna("").astype(str).tolist()


class InvertedIndexPipeline:
    def __init__(self, config: SearchConfig | None = None) -> None:
        self.config = config or SearchConfig()
        self.search_engine = SearchEngine(self.config)

    def fit_from_csv(self, input_path: str | Path) -> SearchEngine:
        dataframe = CorpusReader.read_csv(input_path)
        self.search_engine.fit(dataframe)
        return self.search_engine


def build_search_engine(
        input_path: str | Path,
        text_column: str = "preprocessed_text",
        k1: float = 1.5,
        b: float = 0.75,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 0,
) -> SearchEngine:
    config = SearchConfig(
        text_column=text_column,
        k1=k1,
        b=b,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
    )
    pipeline = InvertedIndexPipeline(config)
    return pipeline.fit_from_csv(input_path)
