import deepl
import os

# Saved in environment variable $DEEPL_AUTH_KEY
print(os.getenv("DEEPL_AUTH_KEY"))
translator = deepl.Translator(os.getenv("DEEPL_AUTH_KEY"))
lang="FI"
"""
result = translator.translate_text(["お元気ですか？", "¿Cómo estás?"], target_lang="EN-GB")
print(result[0].text)  # "How are you?"
print(result[0].detected_source_lang)  # "JA"
print(result[1].text)  # "How are you?"
print(result[1].detected_source_lang)  # "ES"
"""

translator.translate_document_from_filepath(
    "/scratch/project_2009498/dippa/deepl/with_annotations.txt",
    f"translated_with_annotations{lang}.txt",
    target_lang=lang
)
