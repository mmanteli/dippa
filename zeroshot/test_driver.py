from piidetector import PiiDetector
import transformers
import sys
import json
import spacy

context = sys.argv[1]
use_context = {"False":False, "True":True}[context]
#MODEL_NAME = "xlm-roberta-base"
MODEL_NAME="TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_NAME)
lemmatizer = spacy.load("fi_core_news_lg")

pd = PiiDetector(model, tokenizer, lemmatizer, 1e-3, use_context=use_context)#, tokenizer_type="BPE")

text = "Tässä on Amanda Myntti. Amandalle voit laittaa viestiä osoitteeseen amanda.myntti@gmail.com"

print(json.dumps(pd.print_pii(text, debug=True)))
output = print(pd.find_pii(text))
print(json.dumps(output))

