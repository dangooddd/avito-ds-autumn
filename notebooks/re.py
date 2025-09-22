# %%
import re


def add_spaces_around_words(text, words):
    escaped_words = [re.escape(w) for w in words]
    pattern = r"(" + "|".join(escaped_words) + r")"

    def replacer(match):
        word = match.group(0)
        start, end = match.start(), match.end()

        left_ok = (start == 0) or (text[start - 1] == " ")
        right_ok = (end == len(text)) or (text[end] == " ")

        left_space = "" if left_ok else " "
        right_space = "" if right_ok else " "

        return f"{left_space}{word}{right_space}"

    result = re.sub(pattern, replacer, text)
    return result


text = "Этопримерслова табличка иещёслово"
words = {"пример", "табличка", "слово"}

result = add_spaces_around_words(text, words)
print(result)
