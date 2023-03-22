"""
Author: Johanna Garthe
Script to fine-tune pretrained language model
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
from datasets import Features, Value, ClassLabel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.nn.functional import cross_entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import logging
logging.basicConfig(level=logging.INFO)

# ----- CONSTANTS ----- #
MODEL_NAME = " "
NUM_LABELS = 2
MAX_LEN = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
N_EPOCHS = 2
seed = 15
set_seed(seed)

# ----- WEIGHTS AND BIAS INTIATION FOR TRACKING ----- #
wandb.login()
config={"epochs": N_EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "seed": seed, "Data": "BT-balanced"}
wandb.init(project="SD_Dataset3",name="{}-lr_{}_epo_{}_bs_{}".format(MODEL_NAME,LEARNING_RATE,N_EPOCHS,BATCH_SIZE), config=config)

# ----- CONSTANTS ----- #
RUN_NAME = "lr_{}_epo_{}_bs_{}".format(LEARNING_RATE,N_EPOCHS,BATCH_SIZE)
SAVE_DIR = "./saved_models"
TRAINFILE = " "
VALFILE= " "
LABELS = ['AGAINST', 'FAVOR']
NUM_CLASSES = 2
OUTPUT_DIR = "./output"
LOGGING_DIR = "./logs"


def main():
    # ----- FOR TRAINING ON A GPU ----- #
    gpu = 0
    torch.cuda.set_device(gpu)
    print(f'cuda device: {torch.cuda.current_device()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----- LOAD MODEL AND TOKENIZER ----- #
    model = (AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, output_attentions = False, output_hidden_states = False,attention_probs_dropout_prob=0.2,hidden_dropout_prob=0.2).to(device))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # ----- LOAD AND PREPROCESS DATA ----- #
    def create_dataset(train_file, dev_file, numclasses, labels):
        data_files = {'train': train_file , 'validation': dev_file}
        ft = Features({
            'text': Value(dtype='string', id=None),
            'labels': ClassLabel(num_classes=numclasses, names=labels),
            })
        dataset = load_dataset("csv", data_files=data_files, features=ft)
        return dataset
    
    def preprocess(batches):
        # Padding=True      --> Pad texts with zeros to the size of the longest one in a batch
        # Truncation=True   --> Truncate texts to the model's maximum context size
        return tokenizer(batches["text"], truncation=True, padding=True, max_length=MAX_LEN)
    
    dataset = create_dataset(TRAINFILE, VALFILE, NUM_CLASSES, LABELS)
    dataset_encoded = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----- METRICS ----- #
    def compute_metrics(pred):
        labels = pred.label_ids
        # Argmax is used to find the class with the largest predicted probability, axis=-1
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

    # ----- HYPERPARAMETERS ----- #
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,                      # Output directory
        num_train_epochs=N_EPOCHS,                  # Total number of training epochs
        per_device_train_batch_size=BATCH_SIZE,     # Batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,      # Batch size for evaluation
        learning_rate=LEARNING_RATE,
        optim='adamw_torch',
        logging_dir=LOGGING_DIR,                    # Directory for storing logs
        logging_first_step = True,
        logging_strategy='steps',
        evaluation_strategy='steps',                # Evaluate every n number of steps. epoch                       
        load_best_model_at_end=True,                # To load or not the best model at the end
        report_to="wandb",                          
        seed=seed,                                  # Seed for consistent results
        push_to_hub=False,                          # Option to push model to the HH 
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,                                 
        tokenizer=tokenizer,                         
        args=training_args,                           
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded['validation'],
        # Early stopping of training after 3 evaluation calls with no improvement of performance of at least 0.001
        #callbacks = [EarlyStoppingCallback(3, 0.001)]
    )

    # ----- TRAIN MODEL ----- #
    print("********* TRAINING STARTS *********")
    trainer.train()

    # ----- SAVE RESULT SCORES TO CSV ----- #
    result_scores = trainer.evaluate()
    pred_scores = {
        'eval_loss':[result_scores['eval_loss']],'eval_accuracy':[result_scores['eval_acc']],
        'eval_macro_f1':[result_scores['eval_macro_f1']], 'eval_macro_precision':[result_scores['eval_macro_precision']], 'eval_macro_recall':[result_scores['eval_macro_recall']],
        'eval_weighted_f1':[result_scores['eval_weighted_f1']], 'eval_weighted_precision':[result_scores['eval_weighted_precision']], 'eval_weighted_recall':[result_scores['eval_weighted_recall']],    
        'eval_against_f1':[result_scores['eval_against_f1']], 'eval_against_precision':[result_scores['eval_against_precision']], 'eval_against_recall':[result_scores['eval_against_recall']],
        'eval_favor_f1':[result_scores['eval_favor_f1']], 'eval_favor_precision':[result_scores['eval_favor_precision']], 'eval_favor_recall':[result_scores['eval_favor_recall']],
        }
    df = pd.DataFrame(pred_scores)
    transposed = df.transpose().reset_index().rename(columns={'index':'Metrics'})
    transposed.rename(columns={transposed.columns[1]:'Values'}, inplace=True)
    transposed.to_csv('pred_scores_eval.csv', index=False, header=True)
    print('********* SAVED RESULT SCORES *********')

    # ----- SAVE PREDICTION REPORT OF EVAL DATASET ----- #
    def forward_pass_with_label(batch):
        # Place all input tensors on the same device as the model
        inputs = {k:v.to(device) for k,v in batch.items() 
                if k in tokenizer.model_input_names}
        with torch.no_grad():
            output = model(**inputs)
            pred_label = torch.argmax(output.logits, axis=-1)
            loss = cross_entropy(output.logits, batch["labels"].to(device), 
                                reduction="none") 
        return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}
    
    def label_int2str(row):
        # Reads label dict of dataset format (against=0, favor=1)
        return dataset_encoded["validation"].features["labels"].int2str(row)

    # Convert dataset back to PyTorch tensors
    dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    # Compute loss values
    dataset_encoded["validation"] = dataset_encoded["validation"].map(forward_pass_with_label, batched=True)
    # Create Dataframe
    dataset_encoded.set_format("pandas")
    cols = ["text", "labels", "predicted_label", "loss"]
    df_val = dataset_encoded["validation"][:][cols]
    df_val["labels"] = df_val["labels"].apply(label_int2str)
    df_val["predicted_label"] = (df_val["predicted_label"].apply(label_int2str))
    # Save as CSV file sorted by highest loss value
    df_sorted = df_val.sort_values("loss", ascending=False)
    df_sorted.to_csv('error-analysis-report_{}.csv'.format(RUN_NAME), index=False, header=True)
    print("********* CREATED PREDICTION REPORT OF EVAL DATASET *********")

    # ----- SAVE MODEL AND TOKENIZER ----- #
    trainer.save_model(SAVE_DIR)
    #model.save_pretrained(SAVE_DIR)
    #tokenizer.save_pretrained(SAVE_DIR)
    #trainer.push_to_hub(commit_message="Training completed!", repo_name= "name")
    print('********* SAVED MODEL *********')


if __name__ == "__main__":
    main()