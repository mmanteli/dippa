from pprint import PrettyPrinter
import logging
import datasets
import transformers
import torch
import seqeval
import evaluate
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from seqeval.metrics import classification_report
from collections import defaultdict

print(torch.cuda.is_available()) # this does not find cuda aaaah how to fix, no wonder everything has been slow lol

logging.disable(logging.INFO)
pprint = PrettyPrinter(compact=True).pprint

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name',
                    help='Pretrained model name')
    ap.add_argument('--train',nargs='+', metavar='FILE', required=True,
                    help='train file')
    ap.add_argument('--test', metavar='FILE', required=True,
                    help='test file')
    ap.add_argument('--dev', required=True,
                    help='dev file')
    ap.add_argument('--label_names', default = ["PII","O"], 
                    help="labels to use")
    ap.add_argument('--batch', type=int, default=8,
                    help="The batch size for the model")
    ap.add_argument('--epochs', type=int, default=3,
                    help="The number of epochs to train for")
    ap.add_argument('--lr', type=float, default=8e-6,
                    help="The learning rate for the model")
    ap.add_argument('--save', type=str, default=None,
                    help="Path to save the model.")
    return ap



def map_text(example):
    example["text"] = example["title"]+ " "+example["selftext"]+" "+example["best_answer"]
    return example


def tokenize_and_align_labels(example):
    # adapted from https://huggingface.co/docs/transformers/custom_datasets#tok_ner
    tokenized = tokenizer(example["tokens"],
                          truncation=True,
                          is_split_into_words=True)
    # TODO
    return tokenized



def train_and_eval_ner(dataset, tokenizer, args):
    

    model = transformers.AutoModelForTokenClassification.from_pretrained(args.model_name, 
                                                                         num_labels=len(args.label_names),
                                                                         id2label=args.id2label, 
                                                                         label2id=args.label2id)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    trainer_args = transformers.TrainingArguments(
        "checkpoints",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=16,
        num_train_epochs=args.epochs,
    )
    
    metric = evaluate.load("seqeval")
    
    
    def compute_metrics(outputs_and_labels):
        outputs, labels = outputs_and_labels
        predictions = outputs.argmax(axis=2)
    
        # Remove ignored indices (special tokens)
        token_predictions = [
            [tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        token_labels = [
            [tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    
        print(classification_report(token_labels, token_predictions))
        
        results = metric.compute(
            predictions=token_predictions,
            references=token_labels,
            #suffix=True, scheme="IOB2"
        )
    
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]
        }
    
    
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer)
    # Argument gives the number of steps of patience before early stopping
    early_stopping = transformers.EarlyStoppingCallback(early_stopping_patience=5)
    
    
    
    class LogSavingCallback(transformers.TrainerCallback):
        def on_train_begin(self, *args, **kwargs):
            self.logs = defaultdict(list)
            self.training = True
    
        def on_train_end(self, *args, **kwargs):
            self.training = False
    
        def on_log(self, args, state, control, logs, model=None, **kwargs):
            if self.training:
                for k, v in logs.items():
                    if k != "epoch" or v not in self.logs[k]:
                        self.logs[k].append(v)
    
    training_logs = LogSavingCallback()
    
    
    
    trainer = transformers.Trainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer = tokenizer,
        callbacks=[early_stopping, training_logs]
    )

    trainer.train()
    
    eval_results = trainer.evaluate(dataset["test"])
    
    pprint(eval_results)
    
    print('Accuracy:', eval_results['eval_accuracy'])
    return trainer.model
    

if __name__ = "__main__":
    args = argparser().parse_args()
    print(args)

    dataset, label2id, id2label = read_dataset(args)
    args.label2id = label2id
    args.id2label = id2label
    dataset=dataset.shuffle()

    print(dataset)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    dataset = dataset.map(tokenize_and_align_labels)

    
    print("training")
    model=train_and_evaluate(dataset, tokenizer, args)
    
    
    if args.save != None:
        torch.save(model, args.save)
        print("saved model")


