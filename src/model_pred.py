"""
Author: Johanna Garthe
Script to make predictions using the Pipeline
"""

import glob
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd

MODEL_FND = " "
DIR_DATA = " "

def main():
    # ----- TRAINING ON A GPU ----- #
    gpu = 0
    torch.cuda.set_device(gpu)
    print(f'cuda device: {torch.cuda.current_device()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- DIRECTORY ----- #
    filepaths = glob.glob(DIR_DATA + "*.csv")
    for fpathname in filepaths:
        print(fpathname)
        fpath = os.path.basename(fpathname)
        fname = os.path.splitext(fpath)[0]

        # ----- LOAD FINE-TUNED MODEL ----- #
        finetuned_model = (AutoModelForSequenceClassification.from_pretrained(MODEL_FND).to(device))
        finetuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_FND)
        pipeline_finetuned  = pipeline("text-classification", model=finetuned_model, tokenizer=finetuned_tokenizer, device=0)

        # ----- PREDICTIONS ----- #
        def prediction_test_data(data): #label
            df = pd.DataFrame(data, columns=['Text'])
            prediction = pipeline_finetuned(data)
            pred_label = []
            pred_score = []
            # If label names not saved in config during training
            for item in prediction:
                if item['label'] == "LABEL_0":
                    label_name = "against"
                else:
                    label_name = "favor"
                pred_label.append(label_name)
                pred_score.append(item['score'])
            #df['gold_label'] = label
            df['predicted_label'] = pred_label
            df['prediction_score'] = pred_score
            return df

        file = pd.read_csv(fpathname)
        #gold_labels = file['labels'].tolist()
        texts = file['text'].tolist()
        #pred_results = prediction_test_data(texts,gold_labels)
        pred_results = prediction_test_data(texts)
        pred_results.to_csv('predscores_{}.csv'.format(fname), index=False, header=True)
        print("****** FINISHED ******")

if __name__ == "__main__":
    main()