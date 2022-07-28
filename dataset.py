from datasets import load_dataset, load_from_disk
from config import DATA_DIR
import pandas as pd
import os

from preprocess import preprocess

def download_dataset():
    if os.path.exists(os.path.join(DATA_DIR,"train")) and \
    os.path.exists(os.path.join(DATA_DIR, 'validation')) and \
    os.path.exists(os.path.join(DATA_DIR,'test')):
        print("data is already in the directory")
        return True
    else:
        data = load_dataset('cfilt/iitb-english-hindi')
        data.save_to_disk(DATA_DIR)

def prepare_data(type='train',max_entries=100, keep_in_memory=False):
    if type not in ['train', 'test','validation']:
        print("Invalid type of dataset choice: {}".format(type))
        print("type can be one of: train, test, validation")
        assert False
    if type == 'train':
        data = load_from_disk('data/train', keep_in_memory=keep_in_memory)
    if type == 'test':
        data = load_from_disk('data/test', keep_in_memory=keep_in_memory)
    if type == 'validation':
        data = load_from_disk('data/validation', keep_in_memory=keep_in_memory)

    df = pd.DataFrame(data['translation'][:max_entries])

    df['hi'] = df['hi'].apply(preprocess)
    df['en'] = df['en'].apply(preprocess)
    return df
    

if __name__ == '__main__':
    #download_dataset()
    print(prepare_data(type='train').head())



