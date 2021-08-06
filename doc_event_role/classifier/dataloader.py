import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer
import pytorch_lightning as pl

class DataPreprocess:

    def __init__(self, train_path, dev_path, test_path):
        self.train_data = self.prepare_class_data(self.get_data(train_path))
        self.dev_data = self.prepare_class_data(self.get_data(dev_path))
        self.test_data = self.prepare_class_data(self.get_data(test_path))

    def get_data(self,datapath):
        data = []
        with open(datapath) as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def prepare_class_data(self, data):
        class_data = pd.DataFrame(columns=['DocID','Text','Attack','Kidnapping',
                                        'Bombing','Robbery','Arson','Forced'])
        class_dict = {'attack': 0,
                    'kidnapping': 1,
                    'bombing': 2,
                    'robbery': 3,
                    'arson': 4,
                    'forced work stoppage': 5}
        for doc in data:
            doc_list = [doc['docid'],doc['doctext']]
            class_list = [0]*len(class_dict.keys())
            for template in doc['templates']:
                incident_type = template['incident_type']
                if (incident_type=='bombing / attack') or (incident_type=='attack / bombing'):
                    class_list[class_dict['bombing']] = 1
                    class_list[class_dict['attack']] = 1
                    continue
                class_list[class_dict[incident_type]] = 1
            doc_list += class_list
            class_data.loc[len(class_data)] = doc_list
        return class_data

    def get_splits(self):
        return self.train_data, self.dev_data, self.test_data

class MucDataset(Dataset):

    def __init__(self, data, tokenizer, max_token_len: int = 2000):
        self.tokenizer = tokenizer
        self.data = data
        self.label_columns = ['Attack','Kidnapping','Bombing','Arson','Robbery','Forced'] #data.columns.tolist()[2:]
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['Text']
        labels = data_row[self.label_columns]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        return dict(text=text, input_ids=encoding["input_ids"].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(),
                    labels=torch.FloatTensor(labels))

class MucDataModule(pl.LightningDataModule):

    def __init__(self, train_df, dev_df, test_df, tokenizer, workers=12, batch_size=1, max_token_len=2000):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.dev_df = dev_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.workers = workers

    def setup(self, stage=None):
        
        self.train_dataset = MucDataset(self.train_df,
                                        self.tokenizer,
                                        self.max_token_len)
        
        self.dev_dataset = MucDataset(self.dev_df,
                                      self.tokenizer,
                                      self.max_token_len)

        self.test_dataset = MucDataset(self.test_df,
                                       self.tokenizer,
                                       self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.workers)

