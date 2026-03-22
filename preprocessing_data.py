"""Модуль подготавливает русский корпус к индексации и поиску.

Точка входа:
    preprocess_corpus(...)

Вход:
    CSV-файл, в котором есть колонка с текстом.

Выход:
    CSV-файл с исходным текстом, очищенным текстом, токенами,
    леммами и итоговой нормализованной строкой.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymystem3 import Mystem
from stop_words import get_stop_words
from tqdm import tqdm


@dataclass(slots=True)
class PreprocessingConfig:
    """Конфигурация предобработки корпуса.

    Attributes:
        text_column: Название колонки с исходным текстом.
        language: Язык стоп-слов.
        min_token_length: Минимальная длина токена.
        drop_digits: Удаляем токены, состоящие только из цифр.
        lowercase: Приводим текст к нижнему регистру.
    """

    text_column: str = "text"
    language: str = "russian"
    min_token_length: int = 2
    drop_digits: bool = True
    lowercase: bool = True


class RussianTextPreprocessor:
    """Класс для предобработки русских текстов.

    Пайплайн включает:
    1. приведение к нижнему регистру
    2. удаление пунктуации и лишних символов
    3. лемматизацию через Mystem
    4. удаление стоп-слов
    5. доп. фильтрацию токенов
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.mystem = Mystem()
        self.stop_words = set(get_stop_words(self.config.language))

    def preprocess_text(self, text: str) -> dict[str, object]:
        """Предобработать один текст.

        Args:
            text: Исходный текст.

        Returns:
            Словарь с очищенным текстом, леммами и итоговой строкой.
        """
        clean_text = self._normalize_text(text)
        lemmas = self._lemmatize_text(clean_text)
        filtered_lemmas = self._remove_stop_words(lemmas)
        filtered_lemmas = self._filter_tokens(filtered_lemmas)

        return {
            "clean_text": clean_text,
            "tokens": filtered_lemmas,
            "lemmas": filtered_lemmas,
            "preprocessed_text": " ".join(filtered_lemmas),
        }

    def preprocess_series(self, texts: Iterable[str]) -> pd.DataFrame:
        """Предобработать последовательность текстов.

        Args:
            texts: Итерируемый объект с текстами.

        Returns:
            DataFrame с результатами предобработки.
        """
        results: list[dict[str, object]] = []

        # для отслеживания прогресса добавили tqdm
        for text in tqdm(texts, desc="Предобработка текстов"):
            safe_text = "" if pd.isna(text) else str(text)
            results.append(self.preprocess_text(safe_text))

        return pd.DataFrame(results)

    def _normalize_text(self, text: str) -> str:
        """Нормализуем текст перед лемматизацией
        Args:
            text: текст.

        Returns:
            Очищенный текст """
        if self.config.lowercase:
            text = text.lower()

        text = re.sub(r"[^а-яa-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _lemmatize_text(self, text: str) -> list[str]:
        """Лемматизируем текст через Mystem.

        Args:
            text: Очищенный текст.

        Returns:
            Список лемм без пустых строк и пробельных токенов.
        """
        if not text:
            return []

        lemmas = self.mystem.lemmatize(text)
        return [lemma.strip() for lemma in lemmas if lemma.strip()]

    def _remove_stop_words(self, tokens: list[str]) -> list[str]:
        """Удаляем стоп-слова."""
        return [token for token in tokens if token not in self.stop_words]

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Применяем дополнительную фильтрацию токенов."""
        filtered_tokens: list[str] = []

        for token in tokens:
            if len(token) < self.config.min_token_length:
                continue
            if self.config.drop_digits and token.isdigit():
                continue
            filtered_tokens.append(token)

        return filtered_tokens


class CorpusPreprocessingPipeline:
    """Готовый Пайплайн чтения, предобработки и сохранения корпуса."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.preprocessor = RussianTextPreprocessor(self.config)

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        sample_fraction: float | None = None,
    ) -> pd.DataFrame:
        """Запускаем предобработку корпуса.

        Args:
            input_path: Путь к входному CSV-файлу.
            output_path: Путь к выходному CSV-файлу.

        Returns:
            DataFrame с результатами предобработки.
        """
        dataframe = self._read_dataframe(input_path, sample_fraction)
        self._validate_input(dataframe)

        processed = self.preprocessor.preprocess_series(
            dataframe[self.config.text_column]
        )
        result = pd.concat([dataframe.reset_index(drop=True), processed], axis=1)

        self._save_dataframe(result, output_path)
        return result

    @staticmethod
    def _read_dataframe(
        input_path: str | Path,
        sample_fraction: float | None = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Читаем корпус из CSV.

        Args:
            input_path: путь к CSV файлу
            sample_fraction: доля строк, которую нужно оставить (например 0.2)
            random_state: фиксируем сид для воспроизводимости

        Returns:
            DataFrame с корпусом
        """

        df = pd.read_csv(input_path)
        if sample_fraction is not None:
            df = df.sample(frac=sample_fraction, random_state=random_state)
        return df

    def _validate_input(self, dataframe: pd.DataFrame) -> None:
        """Проверяем, что в таблице есть нужная текстовая колонка."""
        if self.config.text_column not in dataframe.columns:
            raise ValueError(
                f"Колонка '{self.config.text_column}' не найдена во входном файле."
            )

    @staticmethod
    def _save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
        """Сохраняем результат предобработки в CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)



def preprocess_corpus(
    input_path: str | Path,
    output_path: str | Path = "data/corpus_preprocessed.csv",
    text_column: str = "text",
    sample_fraction: float | None = None,
) -> pd.DataFrame:
    """Основная точка входа для предобработки корпуса.

    Args:
        input_path: Путь к исходному CSV-файлу.
        output_path: Путь к обработанному CSV-файлу.
        text_column: Название текстовой колонки.

    Returns:
        DataFrame с предобработанным корпусом.
    """
    config = PreprocessingConfig(text_column=text_column)
    pipeline = CorpusPreprocessingPipeline(config)
    return pipeline.run(
        input_path=input_path,
        output_path=output_path,
        sample_fraction=sample_fraction)


if __name__ == "__main__":
    preprocess_corpus(
        input_path="woman.ru – 9 topic.csv",
        output_path="woman_ru_9_topics_preprocessed.csv",
        text_column="text",
        sample_fraction=0.3
    )
