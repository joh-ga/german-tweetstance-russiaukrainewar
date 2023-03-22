"""
Author: Johanna Garthe
Script to perform back translation for data augmentation
"""

from tqdm import tqdm
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

FILENAME = " "
SOURCE_LANGUAGE = "de"
SOURCE_MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
SOURCE_MODEL = MarianMTModel.from_pretrained(SOURCE_MODEL_NAME)
SOURCE_MODEL_TKN = MarianTokenizer.from_pretrained(SOURCE_MODEL_NAME)
TARGET_LANGUAGE = "en"
TARGET_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
TARGET_MODEL = MarianMTModel.from_pretrained(SOURCE_MODEL_NAME)
TARGET_MODEL_TKN = MarianTokenizer.from_pretrained(SOURCE_MODEL_NAME)
CHUNK_SIZE = 5

def main():
  def chunks(batches, chunk_size):
    for i in range(0, len(batches), chunk_size):
      yield batches[i:i + chunk_size]

  def format_batch_texts(language_code, batch_texts):
    formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]
    return formated_bach

  def perform_translation(batch_texts, model, tokenizer, language):
    formated_batch_texts = format_batch_texts(language, batch_texts)  
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512))
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_texts

  def remove_duplicates(batch_texts, back_translated_batch):
    final_augmentation = [elem for elem in back_translated_batch if elem not in batch_texts]
    return final_augmentation

  def perform_back_translation(batch_texts, original_language, original_model, original_tok ,temporary_language, temporary_model, temporary_tok):
    tmp_translated_batch = perform_translation(batch_texts, temporary_model, temporary_tok, temporary_language)
    back_translated_batch = perform_translation(tmp_translated_batch, original_model, original_tok, original_language)
    return remove_duplicates(batch_texts, back_translated_batch)

  df = pd.read_csv('./{}.csv'.format(FILENAME))
  src_texts = df['text_cleaned'].tolist()
  final_backtranslation = []
  for src_text_batches in tqdm(list(chunks(src_texts, CHUNK_SIZE))):
    final = perform_back_translation(src_text_batches, SOURCE_LANGUAGE, SOURCE_MODEL, SOURCE_MODEL_TKN, TARGET_LANGUAGE, TARGET_MODEL, TARGET_MODEL_TKN)
    final_backtranslation.extend(final)

  df_final = pd.DataFrame(final_backtranslation, columns = ['backtranslation'])
  df_final.to_csv('{}_backtranslation.csv'.format(FILENAME), index=False, header=True)
  print('********* BACKTRANSLATION COMPLETED *********')


if __name__ == "__main__":
    main()