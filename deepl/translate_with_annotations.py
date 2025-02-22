import os
import deepl

translation_list="/scratch/project_2009498/dippa/deepl/to_be_translated.txt"
save_directory="/scratch/project_2009498/dippa/deepl/translated_with_annotations"
lang="en"
key = os.getenv("DEEPL_AUTH_KEY")
assert key is not None
translator = deepl.Translator(os.getenv("DEEPL_AUTH_KEY"))

lang_map ={"en":"EN-GB", "fi": "FI"}

annotation_map = {
    "NOMBRE_SUJETO_ASISTENCIA": "#1",
    "EDAD_SUJETO_ASISTENCIA": "#2",
    "SEXO_SUJETO_ASISTENCIA": "#3",
    "FAMILIARES_SUJETO_ASISTENCIA": "#4",
    "NOMBRE_PERSONAL_SANITARIO": "#5",
    "FECHAS": "#6",
    "PROFESION": "#7",
    "HOSPITAL": "#8",
    "ID_CENTRO_DE_SALUD": "#9",
    "INSTITUCION": "#9",
    "CALLE": "#11",
    "TERRITORIO": "#12",
    "PAIS": "#13",
    "NUMERO_TELEFONO": "#14",
    "NUMERO_FAX": "#15",
    "CORREO_ELECTRONICO": "#16",
    "ID_SUJETO_ASISTENCIA": "#17",
    "ID_CONTACTO_ASISTENCIAL": "#18",
    "ID_ASEGURAMIENTO": "#19",
    "ID_TITULACION_PERSONAL_SANITARIO": "#20",
    "ID_EMPLEO_PERSONAL_SANITARIO": "#21",
    "IDENTIF_VEHÍCULOS_NRSERIE_PLACAS": "#22",
    "IDENTIF_DISPOSITIVOS_NRSERIE": "#23",
    "DIREC_PROT_INTERNET":"#24",
    "URL_WEB": "#25",
    "IDENTIF_BIOMETRICOS": "#26",
    "NUMERO_IDENTIF": "#27",
    "OTROS_SUJETO_ASISTENCIA": "#28",
    "CENTRO_SALUD": "#29"
}




def apply_annotations(text_file_path, annotations_file_path, output_file_path):
    # Read the text file
    with open(text_file_path, 'r') as text_file:
        text = text_file.read()
    
    # Read the annotations file
    annotations = []
    with open(annotations_file_path, 'r') as ann_file:
        for line in ann_file:
            #print(line.split('\t'))
            parts = line.split('\t')
            if len(parts) == 3:
                index = parts[0]
                (annotation_type, index_start, index_end) = (i for i in parts[1].split(" "))
                annotation_type = annotation_map[annotation_type]
                annotated_text = parts[2].replace("\n","")
                annotations.append((index, annotation_type, int(index_start), int(index_end), annotated_text))
    
    # Sort annotations by the start index (to avoid overlap issues when applying them)
    annotations.sort(key=lambda x: x[2])  # Sort by index_start
    #print(annotations)

    # Apply the annotations to the text
    offset = 0 # This accounts for the changes in the text length due to annotations
    for index, annotation_type, index_start, index_end, annotated_text in annotations:
        # Adjust the indices due to the changes in text length as annotations are applied
        start_adjusted = index_start + offset
        end_adjusted = index_end + offset
        
        # Replace the segment with the annotation
        text = text[:start_adjusted] + f"[{annotation_type} {annotated_text}]" + text[end_adjusted:]
        
        # Update the offset (length of the annotation replaces the old text length)
        offset += len(f"[{annotation_type}] {annotated_text}") - (end_adjusted - start_adjusted)
    
    # Save the annotated text to a new file
    with open(output_file_path, 'w') as output_file:
        output_file.write(text)


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
        print(f"Translating {path} to {lang_map[lang]} with annotations and saving to {save_file}.")

        temp_path = f'/scratch/project_2009498/tmp/{lang}/{split}/{f}'
        os.makedirs(f'/scratch/project_2009498/tmp/{lang}/{split}', exist_ok=True)
        annot_path = path.replace(".txt", ".ann")

        apply_annotations(path, annot_path, temp_path)

        translator.translate_document_from_filepath(
        temp_path,
        save_file,
        target_lang=lang_map[lang]
        )   

    
    
