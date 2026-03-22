from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from inverted_index import SearchEngine
from matrix_index import MatrixSearchEngine
from preprocessing_data import RussianTextPreprocessor

app = Flask(__name__)

RAW_DATA_PATH = Path("woman.ru – 9 topic.csv")
TEXT_COLUMN = "text"


def ensure_preprocessed(
    dataframe: pd.DataFrame,
    text_column: str = "text",
) -> pd.DataFrame:
    """Проверяет, есть ли колонка preprocessed_text.
    Если нет — запускает предобработку.
    """
    if "preprocessed_text" in dataframe.columns:
        return dataframe

    if text_column not in dataframe.columns:
        raise ValueError(
            f"В файле нет ни 'preprocessed_text', ни исходной колонки '{text_column}'."
        )

    preprocessor = RussianTextPreprocessor()
    processed = preprocessor.preprocess_series(dataframe[text_column])
    result = pd.concat([dataframe.reset_index(drop=True), processed], axis=1)
    return result


def load_or_create_preprocessed_dataframe(
    data_path: str | Path = RAW_DATA_PATH,
    text_column: str = TEXT_COLUMN,
) -> pd.DataFrame:
    """Загружает предобработанный корпус или создает его автоматически."""
    data_path = Path(data_path)
    preprocessed_path = data_path.with_name(f"{data_path.stem}_preprocessed.csv")

    if preprocessed_path.exists():
        print(f"Найден предобработанный файл, загружаем его: {preprocessed_path}")
        return pd.read_csv(preprocessed_path)

    print("Предобработанный файл не найден, запускаем предобработку...")
    dataframe = pd.read_csv(data_path)
    dataframe = ensure_preprocessed(dataframe, text_column=text_column)
    dataframe.to_csv(preprocessed_path, index=False)
    print(f"Предобработка завершена и сохранена в {preprocessed_path}")
    return dataframe


def build_engines(dataframe: pd.DataFrame) -> tuple[SearchEngine, MatrixSearchEngine]:
    """Создает и обучает оба движка один раз при запуске приложения."""
    print("Инициализация inverted engine...")
    inverted_engine = SearchEngine()
    inverted_engine.fit(dataframe)

    print("Инициализация matrix engine...")
    matrix_engine = MatrixSearchEngine()
    matrix_engine.fit(dataframe)

    print("Оба движка готовы.")
    return inverted_engine, matrix_engine


print("Запуск приложения. Подготовка данных и индексов...")
DATAFRAME = load_or_create_preprocessed_dataframe()
INVERTED_ENGINE, MATRIX_ENGINE = build_engines(DATAFRAME)
print("Приложение готово к поиску.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search_page():
    return render_template("search.html")


@app.route("/results", methods=["GET"])
def results():
    query = request.args.get("query", "").strip()
    engine_name = request.args.get("engine", "inverted")
    index_type = request.args.get("index", "bm25")

    try:
        top_k = int(request.args.get("top_k", 5))
    except ValueError:
        top_k = 5

    if not query:
        return render_template(
            "results.html",
            query="",
            engine_name=engine_name,
            index_type=index_type,
            top_k=top_k,
            search_time=0.0,
            results=[],
            error="Введите запрос.",
        )

    try:
        if engine_name == "inverted":
            engine = INVERTED_ENGINE
        elif engine_name == "matrix":
            engine = MATRIX_ENGINE
        else:
            raise ValueError("Некорректный тип движка.")

        start_time = time.time()
        result_df = engine.search(
            query=query,
            index_type=index_type,
            top_k=top_k,
        )
        end_time = time.time()

        results_list = []
        if not result_df.empty:
            for _, row in result_df.iterrows():
                results_list.append(
                    {
                        "doc_id": row["doc_id"] if "doc_id" in row else "",
                        "score": float(row["score"]) if "score" in row else None,
                        "text": row["text"]
                        if "text" in row
                        else row.get("preprocessed_text", ""),
                    }
                )

        return render_template(
            "results.html",
            query=query,
            engine_name=engine_name,
            index_type=index_type,
            top_k=top_k,
            search_time=end_time - start_time,
            results=results_list,
            error=None,
        )

    except Exception as error:
        return render_template(
            "results.html",
            query=query,
            engine_name=engine_name,
            index_type=index_type,
            top_k=top_k,
            search_time=0.0,
            results=[],
            error=f"Ошибка: {error}",
        )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)