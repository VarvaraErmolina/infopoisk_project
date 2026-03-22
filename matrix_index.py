"""
Для матричных операций используются numpy и scipy.
Ожидается, что на вход подается CSV-файл с колонкой `preprocessed_text`,
полученной после предобработки корпуса.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from preprocessing_data import RussianTextPreprocessor


@dataclass(slots=True)
class MatrixSearchConfig:
    """Конфигурация матричной индексации и поиска.

    Attributes:
        text_column: Название колонки с предобработанным текстом.
        k1: Параметр BM25.
        b: Параметр BM25.
    """

    text_column: str = "preprocessed_text"
    k1: float = 1.5
    b: float = 0.75


class MatrixCorpusReader:
    """Чтение корпуса из CSV-файла."""

    @staticmethod
    def read_csv(input_path: str | Path) -> pd.DataFrame:
        return pd.read_csv(input_path)


class MatrixVocabularyBuilder:
    """Построение словаря термов"""

    @staticmethod
    def build(tokenized_documents: list[list[str]]) -> dict[str, int]:
        """Построить словарь термов по корпусу."""
        unique_terms = sorted({term for doc in tokenized_documents for term in doc})
        return {term: term_id for term_id, term in enumerate(unique_terms)}


class MatrixFrequencyInvertedIndex:
    """Частотный обратный индекс через матрицу term-document."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.terms: list[str] = []
        self.term_document_matrix: csr_matrix | None = None
        self.documents_count: int = 0

    def build(self, tokenized_documents: list[list[str]]) -> None:
        """Построить матричный частотный индекс."""
        self.vocabulary = MatrixVocabularyBuilder.build(tokenized_documents)
        self.terms = [term for term, _ in sorted(self.vocabulary.items(), key=lambda x: x[1])]
        self.documents_count = len(tokenized_documents)
        self.term_document_matrix = self._build_term_document_matrix(tokenized_documents)

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Найдем документы по частотному матричному индексу."""
        if self.term_document_matrix is None:
            raise ValueError("Частотный индекс еще не построен.")

        valid_term_ids = [
            self.vocabulary[token]
            for token in query_tokens
            if token in self.vocabulary
        ]

        if not valid_term_ids:
            return {}

        # берем строки матрицы, соответствующие термам запроса,
        # и суммируем их по строкам документов.
        selected_rows = self.term_document_matrix[valid_term_ids, :]
        scores_array = np.asarray(selected_rows.sum(axis=0)).ravel()

        return {int(doc_id): float(score)
            for doc_id, score in enumerate(scores_array)
            if score > 0
        }

    def _build_term_document_matrix(
        self,
        tokenized_documents: list[list[str]],
    ) -> csr_matrix:
        """Строим разреженную матрицу term-document."""
        row_indices: list[int] = []
        col_indices: list[int] = []
        data: list[int] = []

        for doc_id, tokens in enumerate(tokenized_documents):
            # частоты внутри каждого документа
            token_counts: dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            # заполняем инфу для разреж матрицы
            for term, frequency in token_counts.items():
                row_indices.append(self.vocabulary[term])
                col_indices.append(doc_id)
                data.append(frequency)

        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.vocabulary), len(tokenized_documents)),
            dtype=np.float64,
        )


class MatrixBM25InvertedIndex:
    """BM25-индекс через матрицы.

    Хранятся:
    - матрица term-document с частотами термов;
    - вектор длин документов;
    - вектор document frequency;
    - вектор idf.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.vocabulary: dict[str, int] = {}
        self.terms: list[str] = []
        self.term_document_matrix: csr_matrix | None = None
        self.document_lengths: np.ndarray | None = None
        self.document_frequencies: np.ndarray | None = None
        self.idf_vector: np.ndarray | None = None
        self.documents_count: int = 0
        self.average_document_length: float = 0.0

    def build(self, tokenized_documents: list[list[str]]) -> None:
        """Построим матричный BM25-индекс."""
        self.vocabulary = MatrixVocabularyBuilder.build(tokenized_documents)
        self.terms = [term for term, _ in sorted(self.vocabulary.items(), key=lambda x: x[1])]
        self.documents_count = len(tokenized_documents)
        self.term_document_matrix = self._build_term_document_matrix(tokenized_documents)
        self.document_lengths = np.array(
            [len(tokens) for tokens in tokenized_documents],
            dtype=np.float64,
        )
        self.average_document_length = (
            float(self.document_lengths.mean()) if self.documents_count > 0 else 0.0
        )
        self.document_frequencies = self._compute_document_frequencies()
        self.idf_vector = self._compute_idf_vector()

    def search(self, query_tokens: list[str]) -> dict[int, float]:
        """Найдем документы по матричному BM25."""
        self._validate_index()

        valid_term_ids = [self.vocabulary[token] for token in query_tokens if token in self.vocabulary]
        if not valid_term_ids:
            return {}

        scores = np.zeros(self.documents_count, dtype=np.float64)

        for term_id in valid_term_ids:
            term_frequencies = self.term_document_matrix.getrow(term_id).toarray().ravel()
            non_zero_mask = term_frequencies > 0 # считаем только для тех доков, где есть терм
            if not np.any(non_zero_mask):
                continue

            idf = self.idf_vector[term_id]
            document_lengths = self.document_lengths[non_zero_mask]
            term_frequencies_non_zero = term_frequencies[non_zero_mask]

            numerator = term_frequencies_non_zero * (self.k1 + 1)
            denominator = term_frequencies_non_zero + self.k1 * (1 - self.b + self.b * document_lengths / self.average_document_length)

            scores[non_zero_mask] += idf * numerator / denominator

        return {
            int(doc_id): float(score)
            for doc_id, score in enumerate(scores)
            if score > 0
        }


    def _build_term_document_matrix(
        self,
        tokenized_documents: list[list[str]],
    ) -> csr_matrix:
        """Построить матрицу term-document с частотами термов."""
        row_indices: list[int] = []
        col_indices: list[int] = []
        data: list[int] = []

        for doc_id, tokens in enumerate(tokenized_documents):
            token_counts: dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            for term, frequency in token_counts.items():
                row_indices.append(self.vocabulary[term])
                col_indices.append(doc_id)
                data.append(frequency)

        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.vocabulary), len(tokenized_documents)),
            dtype=np.float64,
        )

    def _compute_document_frequencies(self) -> np.ndarray:
        """Считаем, во скольких документах встречается каждый термин"""
        if self.term_document_matrix is None:
            raise ValueError("BM25-индекс еще не построен.")
        
        return np.asarray(self.term_document_matrix.getnnz(axis=1), dtype=np.float64)

    def _compute_idf_vector(self) -> np.ndarray:
        """Считаем вектор IDF для всех терминов корпуса"""
        numerator = self.documents_count - self.document_frequencies + 0.5
        denominator = self.document_frequencies + 0.5
        return np.log1p(numerator / denominator)

    def _validate_index(self) -> None:
        """Проверим, что BM25-индекс полностью построен."""
        if self.term_document_matrix is None:
            raise ValueError("BM25-индекс еще не построен.")
        if self.document_lengths is None:
            raise ValueError("Не посчитаны длины документов.")
        if self.document_frequencies is None:
            raise ValueError("Не посчитаны document frequencies.")
        if self.idf_vector is None:
            raise ValueError("Не посчитан вектор IDF.")


class MatrixSearchEngine:
    """Поисковый движок поверх матричных индексов."""

    def __init__(self, config: MatrixSearchConfig | None = None) -> None:
        self.config = config or MatrixSearchConfig()
        self.query_preprocessor = RussianTextPreprocessor()
        self.frequency_index = MatrixFrequencyInvertedIndex()
        self.bm25_index = MatrixBM25InvertedIndex(
            k1=self.config.k1,
            b=self.config.b,
        )
        self.dataframe: pd.DataFrame | None = None
        self.tokenized_documents: list[list[str]] = []

    def fit(self, dataframe: pd.DataFrame) -> None:
        """Построить оба матричных индекса по корпусу."""
        self._validate_dataframe(dataframe)
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.tokenized_documents = self._extract_tokenized_documents(self.dataframe)

        self.frequency_index.build(self.tokenized_documents)
        self.bm25_index.build(self.tokenized_documents)

    def search(
        self,
        query: str,
        index_type: str = "bm25",
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Найдем top-k документов по запросу."""
        if self.dataframe is None:
            raise ValueError("Сначала нужно вызвать fit() и построить индексы.")

        query_tokens = self.query_preprocessor.preprocess_text(query)["lemmas"]
        if not query_tokens:
            return self.dataframe.head(0).copy()

        if index_type == "frequency":
            scores = self.frequency_index.search(query_tokens)
        elif index_type == "bm25":
            scores = self.bm25_index.search(query_tokens)
        else:
            raise ValueError("index_type должен быть 'frequency' или 'bm25'.")

        ranked_results = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        if not ranked_results:
            return self.dataframe.head(0).copy()

        doc_ids = [doc_id for doc_id, _ in ranked_results]
        doc_scores = [score for _, score in ranked_results]

        result = self.dataframe.iloc[doc_ids].copy()
        result.insert(0, "doc_id", doc_ids)
        result.insert(1, "score", doc_scores)
        return result

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Проверить, что в корпусе есть нужная колонка."""
        if self.config.text_column not in dataframe.columns:
            raise ValueError(
                f"Колонка '{self.config.text_column}' не найдена во входном DataFrame."
            )

    def _extract_tokenized_documents(
        self,
        dataframe: pd.DataFrame,
    ) -> list[list[str]]:
        """Преобразуем колонку в список токенизированных документов"""
        return [
            str(text).split() if pd.notna(text) else []
            for text in dataframe[self.config.text_column]
        ]


class MatrixIndexPipeline:
    """Пайплайн чтения корпуса и построения матричных индексов."""

    def __init__(self, config: MatrixSearchConfig | None = None) -> None:
        self.config = config or MatrixSearchConfig()
        self.search_engine = MatrixSearchEngine(self.config)

    def fit_from_csv(self, input_path: str | Path) -> MatrixSearchEngine:
        """Прочитаем корпус и построить матричные индексы."""
        dataframe = MatrixCorpusReader.read_csv(input_path)
        self.search_engine.fit(dataframe)
        return self.search_engine



def build_matrix_search_engine(
    input_path: str | Path,
    text_column: str = "preprocessed_text",
    k1: float = 1.5,
    b: float = 0.75,
) -> MatrixSearchEngine:
    """Основная точка входа для пользователя.

    Args:
        input_path: Путь к CSV-файлу с корпусом.
        text_column: Название колонки с предобработанным текстом.
        k1: Параметр BM25.
        b: Параметр BM25.

    Returns:
        Готовый поисковый движок с матричными индексами.
    """
    config = MatrixSearchConfig(text_column=text_column, k1=k1, b=b)
    pipeline = MatrixIndexPipeline(config)
    return pipeline.fit_from_csv(input_path)


if __name__ == "__main__":
    engine = build_matrix_search_engine(
        input_path="woman_ru_9_topics_preprocessed.csv",
        text_column="preprocessed_text",
    )

    results = engine.search(
        query="Мне изменил муж",
        index_type="bm25",
        top_k=5,
    )
    print("Результаты поиска:")
    for _, row in results.iterrows():
        print("=" * 80)
        print(f"doc_id: {row['doc_id']}")
        print(f"score : {row['score']:.4f}")
        print()
        print(row["text"])
        print()