from piifinder import PiiFinder
import transformers

MODEL_NAME = "xlm-roberta-base"
#MODEL_NAME="TurkuNLP/bert-base-finnish-cased-v1"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_NAME)

pf = PiiFinder(model, tokenizer, 1e-2, "BPE")

text = "Moi, olen Amanda, mulle voit laittaa viestiä osoitteeseen example@outlook.com"

print(pf.print_pii(text, debug=True))
#print(pf.find_pii(text))