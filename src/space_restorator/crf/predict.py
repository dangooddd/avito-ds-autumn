from .train import text2features


def restore_spaces(crf_model, text_without_spaces):
    features = text2features(text_without_spaces)
    predictions = crf_model.predict([features])[0]

    result = []
    for char, label in zip(text_without_spaces, predictions):
        result.append(char)
        if label == "SPACE":
            result.append(" ")

    return "".join(result)
