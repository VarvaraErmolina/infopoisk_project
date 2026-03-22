import argparse
import time
import pandas as pd
from inverted_index import SearchEngine
from matrix_index import MatrixSearchEngine
from preprocessing_data import RussianTextPreprocessor
from pathlib import Path


def ensure_preprocessed(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    if "preprocessed_text" in df.columns:
        return df

    if text_column not in df.columns:
        raise ValueError(
            f"В файле нет ни 'preprocessed_text', ни исходной колонки '{text_column}'."
        )

    preprocessor = RussianTextPreprocessor()
    processed = preprocessor.preprocess_series(df[text_column])
    result = pd.concat([df.reset_index(drop=True), processed], axis=1)
    return result


def main():
    parser = argparse.ArgumentParser(description="CLI для поиска по корпусу")

    parser.add_argument("--query", type=str, required=True, help="Текст запроса")
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        choices=["inverted", "matrix"],
        help="Тип движка",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        choices=["frequency", "bm25", "word2vec", "fasttext"],
        help="Тип индекса",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="woman.ru – 9 topic.csv",
        help="Путь к CSV",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Название исходной текстовой колонки",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Количество результатов")

    args = parser.parse_args()

    preprocessed_path = Path(args.data).with_name(Path(args.data).stem + "_preprocessed.csv")
    print(preprocessed_path)
    if preprocessed_path.exists():
        print("Найден предобработанный файл, загружаем его...")
        df = pd.read_csv(preprocessed_path)

    else:
        print("Предобработанный файл не найден, запускаем предобработку...")

        df = pd.read_csv(args.data)
        df = ensure_preprocessed(df, text_column=args.text_column)

        df.to_csv(preprocessed_path, index=False)
        print("Предобработка завершена и сохранена в preprocessed_data.csv")

    if args.engine == "inverted":
        engine = SearchEngine()
    else:
        engine = MatrixSearchEngine()

    engine.fit(df)

    start_time = time.time()
    results = engine.search(
        query=args.query,
        index_type=args.index,
        top_k=args.top_k,
    )
    end_time = time.time()

    print("\nРезультаты поиска:\n")
    if results.empty:
        print("Ничего не найдено.")
    else:
        for _, row in results.iterrows():
            print("=" * 80)
            print(f"doc_id: {row['doc_id']}")
            print(f"score : {row['score']:.4f}")
            print()
            print(row["text"] if "text" in row else row["preprocessed_text"])
            print()

    print(f"Время поиска: {end_time - start_time:.4f} секунд")


if __name__ == "__main__":
    main()