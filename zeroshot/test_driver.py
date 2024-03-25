from piimasker import PiiMasker
import transformers
import sys

context = sys.argv[1]
use_context = {"False":False, "True":True}[context]
#MODEL_NAME = "xlm-roberta-base"
MODEL_NAME="TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_NAME)

pf = PiiMasker(model, tokenizer, 1e-4, use_context=use_context, tokenizer_type="WordPiece")

text = "Moi, olen Amanda, mulle voit laittaa viestiä osoitteeseen amanda@outlook.com"

print(pf.print_pii(text))#, debug=True))
#output = print(pf.find_pii(text))

