"""
Author: Johanna Garthe
Script to perform splitting of data per target topic in train, test, and dev (80/10/10).
Input = complete dataset
Output = train, val and test files including all target data
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# ----- FILES----- #
DIR = " "
FILENAME = " "

def main():
    train_data_all = []
    test_data_all = []
    val_data_all = []

    def target_dataset_split(dataset, targetname):
        training_data, test_data = train_test_split(dataset, test_size=0.2, random_state=25)
        testing_data, validation_data = train_test_split(test_data, test_size=0.5, random_state=25)
        print(f"No. of training examples: {training_data.shape[0]}")
        print(f"No. of testing examples: {testing_data.shape[0]}")
        print(f"No. of validation examples: {validation_data.shape[0]}")
        train_data_all.append(training_data)
        test_data_all.append(testing_data)
        val_data_all.append(validation_data)
        print('********* Train test split for {} completed *********'.format(targetname))

    dataset = pd.read_csv(DIR + FILENAME + '.csv')
    ad_against = dataset[(dataset['target']=='AD') & (dataset['label']=='AGAINST')] 
    ad_favor = dataset[(dataset['target']=='AD') & (dataset['label']=='FAVOR')] 
    noc_against = dataset[(dataset['target']=='NOC') & (dataset['label']=='AGAINST')]
    noc_favor = dataset[(dataset['target']=='NOCn') & (dataset['label']=='FAVOR')]
    sli_against = dataset[(dataset['target']=='SLI') & (dataset['label']=='AGAINST')]
    sli_favor = dataset[(dataset['target']=='SLI') & (dataset['label']=='FAVOR')]
    us_against = dataset[(dataset['target']=='US') & (dataset['label']=='AGAINST')]
    us_favor = dataset[(dataset['target']=='US') & (dataset['label']=='FAVOR')]

    # ----- SPLIT INTO SETS ----- #
    target_dataset_split(ad_against, 'ad_against')
    target_dataset_split(ad_favor, 'ad_favor')
    target_dataset_split(noc_against, 'noc_against')
    target_dataset_split(noc_favor, 'noc_favor')
    target_dataset_split(sli_against, 'sli_against')
    target_dataset_split(sli_favor, 'sli_favor')
    target_dataset_split(us_against, 'us_against')
    target_dataset_split(us_favor, 'us_favor')

    # Merge Dataframe list of single target data into one Dataframe per train and test split
    train_data_all_appended = pd.concat(train_data_all)
    test_data_all_appended = pd.concat(test_data_all)
    val_data_all_appended = pd.concat(val_data_all)

    # ----- SAVE INTO NEW CSV FILES ----- #
    train_data_all_appended.to_csv(DIR + '{}_train.csv'.format(FILENAME), index=False, header=True)
    test_data_all_appended.to_csv(DIR + '{}_test.csv'.format(FILENAME), index=False, header=True)
    val_data_all_appended.to_csv(DIR + '{}_val.csv'.format(FILENAME), index=False, header=True)
    print('********* COMPLETED *********')


if __name__ == "__main__":
    main()