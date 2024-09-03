import os
import deepl

translation_list="/scratch/project_2009498/dippa/deepl/to_be_translated.txt"
save_directory="/scratch/project_2009498/dippa/deepl/translated"
lang="fi"
translator = deepl.Translator(os.getenv("DEEPL_AUTH_KEY"))

lang_map ={"en":"EN-GB", "fi": "FI"}

with open(translation_list) as f:
    paths = [i.rstrip() for i in f.readlines()]

for path in paths:
    p, f = os.path.split(path)
    split = p.split("/")[-2]
    assert split in ["train","dev","test"]
    save_path=os.path.join(save_directory,lang,split)
    save_file=os.path.join(save_path,f)
    if not os.path.isdir(save_path):
        print(f"Making {save_path}")
        os.makedirs(save_path)
    if os.path.isfile(save_file):
        print(f"Already done {save_file}")
    else:
        print(f"Translating {path} to {lang_map[lang]} and saving to {save_file}.")

        translator.translate_document_from_filepath(
        path,
        save_file,
        target_lang=lang_map[lang]
        )   

    
    
