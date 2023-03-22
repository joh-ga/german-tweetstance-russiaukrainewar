"""
Author: Johanna Garthe
Script to evaluate finetuned model with test sets
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import load_dataset
from datasets import Features, Value, ClassLabel
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from torch.nn.functional import cross_entropy
import pandas as pd
import numpy as np
import torch
import glob
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----- CONSTANTS ----- #
DIR = " "
LABELS = ['AGAINST', 'FAVOR']
MODEL_FINETUNED_NAME = " "
MODEL_NAME = " "
NUM_LABELS = 2
MAX_LEN = 512

def main():

    # ----- DIRECTORY ----- #
    filepaths = glob.glob(DIR + "*.csv")
    for fpathname in filepaths:
        print(fpathname)
        fpath = os.path.basename(fpathname)
        fname = os.path.splitext(fpath)[0]

        # ----- TRAINING ON A GPU ----- #
        gpu = 0
        torch.cuda.set_device(gpu)
        print(f'cuda device: {torch.cuda.current_device()}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- LOAD MODEL ----- #
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_FINETUNED_NAME, num_labels=NUM_LABELS)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # ----- LOAD AND PREPROCESS DATA ----- #
        def create_dataset(test_file, labels):
            data_files = {'test': test_file}
            ft = Features({
                'text': Value(dtype='string', id=None),
                'labels': ClassLabel(names=labels),
                })
            dataset = load_dataset("csv", data_files=data_files,features=ft)
            return dataset
        
        def preprocess(batches):
            # Padding=True      --> Pad texts with zeros to the size of the longest one in a batch
            # Truncation=True   --> Truncate texts to the model's maximum context size
            return tokenizer(batches["text"], truncation=True, padding=True, max_length=MAX_LEN)
        
        dataset = create_dataset(fpathname, LABELS)
        dataset_encoded = dataset.map(preprocess, batched=True, batch_size=None)

        # ----- INDICATE METRICS ----- #
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            report = classification_report(y_true=labels, y_pred=preds, output_dict=True)
            macro_precision =  report['macro avg']['precision'] 
            macro_recall = report['macro avg']['recall']    
            macro_f1 = report['macro avg']['f1-score']
            weighted_precision =  report['weighted avg']['precision'] 
            weighted_recall = report['weighted avg']['recall']    
            weighted_f1 = report['weighted avg']['f1-score']       
            against_precision = report['0']['precision']
            against_recall = report['0']['recall']
            against_f1 = report['0']['f1-score']
            favor_precision = report['1']['precision']
            favor_recall = report['1']['recall']
            favor_f1 = report['1']['f1-score']
            return {
                "acc": acc,
                "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1,
                "weighted_precision": weighted_precision, "weighted_recall": weighted_recall, "weighted_f1": weighted_f1,
                "against_precision": against_precision, "against_recall": against_recall, "against_f1": against_f1,
                "favor_precision": favor_precision, "favor_recall": favor_recall, "favor_f1": favor_f1}

        # ----- TRAINER ----- #
        trainer = Trainer(
            model=model,
            eval_dataset=dataset_encoded["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics)

        # ----- SAVE RESULT SCORES INTO A CSV FILE----- #
        result_scores = trainer.evaluate()
        pred_scores = {
            'eval_accuracy':[result_scores['eval_acc']],
            'eval_macro_f1':[result_scores['eval_macro_f1']], 'eval_macro_precision':[result_scores['eval_macro_precision']], 'eval_macro_recall':[result_scores['eval_macro_recall']],
            'eval_weighted_f1':[result_scores['eval_weighted_f1']], 'eval_weighted_precision':[result_scores['eval_weighted_precision']], 'eval_weighted_recall':[result_scores['eval_weighted_recall']],    
            'eval_against_f1':[result_scores['eval_against_f1']], 'eval_against_precision':[result_scores['eval_against_precision']], 'eval_against_recall':[result_scores['eval_against_recall']],
            'eval_favor_f1':[result_scores['eval_favor_f1']], 'eval_favor_precision':[result_scores['eval_favor_precision']], 'eval_favor_recall':[result_scores['eval_favor_recall']],
            }
        df = pd.DataFrame(pred_scores)
        transposed = df.transpose().reset_index().rename(columns={'index':'Metrics'})
        transposed.rename(columns={transposed.columns[1]:'Values'}, inplace=True)
        transposed.to_csv('pred_scores_testdata_{}.csv'.format(fname), index=False, header=True)
        print('********* SAVED RESULT SCORES *********')

        # ----- CREATE AND SAVE CONFUSION MATRIX ----- #
        preds_output = trainer.predict(dataset_encoded['test'])

        def plot_confusion_matrix(y_preds, y_true, labels):
            cm = confusion_matrix(y_true, y_preds, normalize="true")
            fig, ax = plt.subplots(figsize=(15, 15))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
            font = {'size':22}
            plt.rc('font', **font)
            plt.title("Normalized confusion matrix")
            plt.savefig('confusion_matrix_testdata_{}.png'.format(fname))

        labels = dataset_encoded['test'].features["labels"].names
        y_valid = np.array(dataset_encoded['test']["labels"])
        y_preds = np.argmax(preds_output.predictions, axis=1)
        plot_confusion_matrix(y_preds, y_valid, labels)
        print('********* CREATED CONFUSION MATRIX *********')

        # ----- SAVE PREDICTION SCORES INTO CSV FILE ----- #
        def label_int2str(row):
            return dataset["test"].features["labels"].int2str(row)

        def forward_pass_with_label(batch):
        # Place all input tensors on the same device as the model
            inputs = {k:v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names}
            with torch.no_grad():
                output = model(**inputs)
                pred_label = torch.argmax(output.logits, axis=-1)
                loss = cross_entropy(output.logits, batch["labels"].to(device),reduction="none")
            # Place outputs on CPU for compatibility with other dataset columns
            return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}
        
        # Convert dataset back to PyTorch tensors
        dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        # Compute loss values
        dataset_encoded["test"] = dataset_encoded["test"].map(forward_pass_with_label, batched=True)
        dataset_encoded.set_format("pandas")
        cols = ["text", "labels", "predicted_label", "loss"]
        df_test = dataset_encoded["test"][:][cols]
        df_test["labels"] = df_test["labels"].apply(label_int2str)
        df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))
        # Save predictions into CSV file sorted by highest loss value
        df_sorted = df_test.sort_values("loss", ascending=False)
        df_sorted.to_csv('./error-analysis-report_testdata_{}.csv'.format(fname), index=False, header=True)
        print("********* CREATED PREDICTION REPORT OF VAL SET *********") 


if __name__ == "__main__":
    main()
    print('********* FINISHED *********')