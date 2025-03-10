import os
os.environ['HF_HOME'] = '/scratch/project_2009498/cache'

from piidetector import PiiDetector
import transformers
import sys
import json
import spacy


context = sys.argv[1]
use_context = {"False":False, "True":True}[context]
#MODEL_NAME = "xlm-roberta-base"
MODEL_NAME="TurkuNLP/bert-base-finnish-cased-v1"
#MODEL_NAME="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
#MODEL_NAME="beatrice-portelli/DiLBERT"
#MODEL_NAME="allenai/biomed_roberta_base"
TOKENIZER_NAME = MODEL_NAME if MODEL_NAME!="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" else "xlm-roberta-large"

tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_NAME)
lemmatizer = spacy.load("fi_core_news_lg")

f = open("/scratch/project_2009498/dippa/deepl/translated/fi/test/S0210-56912006000300007-2.txt", "r")
#f = open("/scratch/project_2009498/dippa/deepl/translated/fi/dev/S0211-69952011000400013-1.txt", "r")
text = f.read()
#text = "Tässä on Amanda Myntti. Amandalle voit laittaa viestiä osoitteeseen amanda.myntti@gmail.com"

#print(text)
pd = PiiDetector(model, tokenizer, lemmatizer, 1e-4, use_context=use_context)#, tokenizer_type="WordPiece")#, tokenizer_type="BPE")

print(json.dumps(pd.redact_pii(text)))#, debug=True)))
#output = print(pd.find_pii(text))
#print(json.dumps(output))

