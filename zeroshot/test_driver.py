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
#MODEL_NAME="TurkuNLP/bert-base-finnish-cased-v1"
#MODEL_NAME="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
#MODEL_NAME="beatrice-portelli/DiLBERT"
#MODEL_NAME="allenai/biomed_roberta_base"
#MODEL_NAME="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer_map = {"cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR":"xlm-roberta-large",
                 "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es":"PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                 #"allenai/biomed_roberta_base": "xlm-roberta-base"
                 }
TOKENIZER_NAME = tokenizer_map.get(MODEL_NAME, MODEL_NAME)

tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = transformers.AutoModelForPreTraining.from_pretrained(MODEL_NAME)
lemmatizer = spacy.load("es_core_news_md")

text = """
Datos del paciente.
Nombre:  Guillermo.
Apellidos: Pedroche Roa.
NHC: 3587642.
NASS: 2324.
Fecha de Ingreso: 21/12/2016.
Médico: Roberto Llarena Ibarguren  NºCol: 20 20 94583 (AS), 38.
Informe clínico del paciente: Varón de 64 años que consultó en mayo de 2006 por disestesia facial y ptosis palpebral izquierda y proptosis del globo ocular del mismo lado.
Se alcanza un nadir de PSA en julio de 2006 con cifras de 6-8 ng/ml, tras las dos primeras administraciones de docetaxel. Remitido por: Roberto Llarena Ibarguren Apartado de Correos 20134 48080 Bilbao. (España) rllarena@eusklanet.net
"""


#f = open("/scratch/project_2009498/dippa/deepl/translated/fi/test/S0210-56912006000300007-2.txt", "r")
#f = open("/scratch/project_2009498/dippa/deepl/translated/fi/dev/S0004-06142007000900017-1.txt", "r")
#text = f.read()
#text = "Tässä on Amanda Myntti (amanda). \nAmandalle voit laittaa viestiä osoitteeseen amanda.myntti@gmail.com"

#print(text)
pd = PiiDetector(model, tokenizer, lemmatizer, 1e-4, use_context=use_context)#, tokenizer_type="WordPiece")

print(json.dumps(pd.redact_pii(text,"scores/es/bert-es/S0004-06142007000900017-1.txt")))#, debug=True)))
#output = print(pd.find_pii(text))
#print(json.dumps(output))

