import deepl
import os

# Saved in environment variable $DEEPL_AUTH_KEY
print(os.getenv("DEEPL_AUTH_KEY"))
translator = deepl.Translator(os.getenv("DEEPL_AUTH_KEY"))
result = translator.translate_text(["お元気ですか？", "¿Cómo estás?"], target_lang="EN-GB")
print(result[0].text)  # "How are you?"
print(result[0].detected_source_lang)  # "JA"
print(result[1].text)  # "How are you?"
print(result[1].detected_source_lang)  # "ES"