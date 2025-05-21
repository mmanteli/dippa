import os
os.environ['HF_HOME'] = '/scratch/project_2009498/cache/'

from piidetector import PiiDetector
from transformers import AutoModelForPreTraining, AutoTokenizer
import sys
import json
import spacy
import argparse
import re


# "dual key" dictionary
path_for_data = lambda split: {"fi":f"/scratch/project_2009498/dippa/deepl/translated_with_annotations/fi/{split}",
                 "en":f"/scratch/project_2009498/dippa/deepl/translated_with_annotations/en/{split}",
                 "es":f"/scratch/project_2009498/dippa/deepl/translated_with_annotations/es/{split}",
                 }
lemmatizer_map = {"fi":"fi_core_news_lg",
              "en": False,
              "es": "es_core_news_md"}

# this for models with long names
model_map = {"cambridgeltl":"cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
             "dilbert":"beatrice-portelli/DiLBERT",
             "bert-fi":"TurkuNLP/bert-base-finnish-cased-v1",
             "biomed-roberta": "allenai/biomed_roberta_base",
             "bert-en": "google-bert/bert-base-cased",
             "bert-es": "dccuchile/bert-base-spanish-wwm-uncased",
             "biomed-es":"PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
             "bioclin-bert":"emilyalsentzer/Bio_ClinicalBERT",
             "pubmed-bert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}


def reverse_annotations(text):
    """
    Reverse MEDDOCAN + deepL annotations
    E.g. Name: [#3 Joseo] 
    to
    Name: Joseo
    """

    i = re.finditer(r'\[\#[0-9]{1,2}[ ].+?\]', text)
    ind = [[m.start(0),m.end(0)] for m in i]
    current_index=0
    parsed_text=""
    annotations=[]
    for start, end in ind:
        parsed_text = parsed_text + text[current_index:start]
        annot = text[start:end].split(" ")
        pii_class = annot[0].replace("[","")
        annot_text = " ".join(annot[1:]).replace("]","")
        parsed_text += annot_text
        current_index = end
        annotations.append([pii_class, len(parsed_text)-len(annot_text), len(parsed_text)])

    return parsed_text

def extract(options):
    data_path = path_for_data(options.split)[options.lang]
    MODEL_NAME = model_map.get(options.model, options.model)
    TOKENIZER_NAME = MODEL_NAME if MODEL_NAME!="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" else "xlm-roberta-large"
    LEMMATIZER = lemmatizer_map[options.lang]
    use_context=True # should be set to True always, False for testing

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModelForPreTraining.from_pretrained(MODEL_NAME)
    if LEMMATIZER:
        lemmatizer = spacy.load(LEMMATIZER)
    else:
        lemmatizer=LEMMATIZER

    pd = PiiDetector(model, tokenizer, lemmatizer, 0, use_context=use_context)#, tokenizer_type="WordPiece")#, tokenizer_type="BPE")

    save_path = f"/scratch/project_2009498/dippa/zeroshot/scores_by_token/{options.lang}/{options.split}/{options.model}"
    os.makedirs(save_path, exist_ok=True)

    for file in os.scandir(data_path):  
        print(f"In file {file}")
        save_file = save_path+"/"+file.name.replace(".txt","")
        if os.path.isfile(save_path+"/"+file.name.replace(".txt", "_by_token.tsv")):
            print(f"\tfile {file.name} already done!")
            continue   # skip if already done
        with open(file, "r") as f:
            text_with_annotations = f.read()
            text = reverse_annotations(text_with_annotations)
            print(text)
        try:
            pd.score_pii(str(text), save_file)
        except Exception as e:
            print(f'Error with file {file.name}')
            print(e)
            print("\n-----------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Extract scores for texts")
    
    # Add required arguments
    parser.add_argument('--lang', required=True, type=str, help='Language code')
    parser.add_argument('--model', required=True, type=str, help='Model name')
    parser.add_argument('--split', required=True, type=str, help="Data split")

    # Parse the arguments
    options = parser.parse_args()
    print(options)

    # run
    extract(options)

if __name__=="__main__":
    main()