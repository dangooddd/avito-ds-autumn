import sklearn_crfsuite
from sklearn_crfsuite import metrics
from .dataset import create_dataset
from pathlib import Path
import pickle


def word2features(text, i):
    """Извлечение признаков для символа в позиции i"""
    char = text[i]
    features = {
        "bias": 1.0,
        f"char={char}": 1.0,
        "char.isalpha": char.isalpha(),
        "char.isdigit": char.isdigit(),
        "char.isupper": char.isupper(),
        "is_first": i == 0,
        "is_last": i == len(text) - 1,
    }

    # Контекстные признаки
    if i > 0:
        prev_char = text[i - 1]
        features.update(
            {
                f"prev_char={prev_char}": 1.0,
                "prev_char.isalpha": prev_char.isalpha(),
            }
        )

    if i < len(text) - 1:
        next_char = text[i + 1]
        features.update(
            {
                f"next_char={next_char}": 1.0,
                "next_char.isalpha": next_char.isalpha(),
            }
        )

    # Биграммы для контекста
    if i > 0:
        features[f"bigram={text[i - 1 : i + 1]}"] = 1.0

    return features


def text2features(text):
    return [word2features(text, i) for i in range(len(text))]


def text2labels(text_with_spaces):
    """Преобразование текста с пробелами в метки"""
    labels = []
    clean_text = text_with_spaces.replace(" ", "")

    j = 0
    for i, char in enumerate(text_with_spaces):
        if char != " ":
            # Пробел после символа?
            if i < len(text_with_spaces) - 1 and text_with_spaces[i + 1] == " ":
                labels.append("SPACE")
            else:
                labels.append("NO_SPACE")
            j += 1

    return clean_text, labels


# Обучение модели
def train_crf_model(model, training_texts):
    # Подготовка данных
    X, y = [], []
    for text in training_texts:
        clean_text, labels = text2labels(text)
        features = text2features(clean_text)
        X.append(features)
        y.append(labels)

    model.fit(X, y)
    return model


def train(save_path: Path, start, num):
    model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,  # L1 регуляризация
        c2=0.1,  # L2 регуляризация
        max_iterations=100,
        all_possible_transitions=True,
    )

    dataset = list(create_dataset().skip(start).take(num))
    texts = [ex["text"] for ex in dataset]
    model = train_crf_model(model, texts)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    from .predict import restore_spaces

    save_path = Path("data/crf/model.pkl")
    start = 500
    num = 20000
    model = train(save_path, start, num)

    test_cases = [
        "купитьайфон14про",
        "яидупоулице",
        "всталивышел",
        "хочешьсладкихапельсинов",
    ]

    for text in test_cases:
        restored = restore_spaces(model, text)
        print(f"'{text}' → '{restored}'")
