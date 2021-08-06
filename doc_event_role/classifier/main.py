import argparse
import pandas as pd
from dataloader import *
from model import *
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
import os
import json

class EventClassifier:

    def __init__(self, model, data_module, logger, checkpoint_callback, callback_list, epochs, gpu):
        self.model = model
        self.data_module = data_module
        self.logger = logger
        self.checkpoint_callback = checkpoint_callback
        self.callback_list = callback_list
        self.epoch = epochs
        self.gpu = gpu
        self.trainer = pl.Trainer(logger = self.logger, checkpoint_callback=self.checkpoint_callback,
                                  callbacks=self.callback_list, max_epochs=self.epoch, gpus=self.gpu,
                                  progress_bar_refresh_rate=30)


    def train(self):
        self.trainer.fit(model, data_module)

    def predict(self, test_data, tokenizer, max_token_len, label_columns, model_name, save_model_path):
        trained_model = self.model.load_from_checkpoint(save_model_path, n_classes=len(label_columns), label_columns=label_columns, model_name=model_name)
        trained_model.eval()
        trained_model.freeze()

        if self.gpu>0:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        trained_model = trained_model.to(device)

        val_dataset = MucDataset(test_data, tokenizer, max_token_len=max_token_len)

        predictions = []
        labels = []

        for item in tqdm(val_dataset):
            _, prediction = trained_model(item["input_ids"].unsqueeze(dim=0).to(device), 
                                        item["attention_mask"].unsqueeze(dim=0).to(device))
            predictions.append(prediction.flatten())
            if "labels" in item.keys():
                labels.append(item["labels"].int())

        predictions = torch.stack(predictions).detach().cpu()
        if labels!=[]:
            labels = torch.stack(labels).detach().cpu()
            return predictions, labels
        else:
            return predictions, None

    def evaluate(self,test_data, tokenizer, max_token_len, label_columns, model_name, saved_model_path):
        predictions, labels = self.predict(test_data, tokenizer, max_token_len,label_columns, model_name, saved_model_path)
        y_pred = predictions.numpy()
        upper, lower = 1, 0
        y_pred = np.where(y_pred > 0.5, upper, lower)

        if labels!=None:
            y_true = labels.numpy()
            print(classification_report(y_true, y_pred, target_names=label_columns, zero_division=0))

        test_columns = test_data.columns
        test_inference = pd.concat([test_data[['DocID','Text']], pd.DataFrame(y_pred)], axis=1)
        test_inference.columns = test_columns

        return test_inference

class DataConvert:

    def __init__(self, table_data):
        self.table_data = table_data

    def assign_data(self, template_names, role_names):
        output_data = {}
        empty_role = {}
        for role in role_names:
            empty_role[role] = []
        for name in template_names:
            output_data[name] = {}
            for i,entry in enumerate(self.table_data[name]):
                if entry==1:
                    doc_count = len(output_data[name].keys()) + 1
                    docid = "TST-MUC3-" + str(doc_count).zfill(4)
                    output_data[name][docid] = {"doc": self.table_data.loc[i,'Text'],
                                                "roles": empty_role}
            with open('output/'+ name.lower() +'/output.json', 'w', encoding='utf-8') as f:
                json.dump(output_data[name], f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse arguments')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--train_path',  help='Path to train dataset', default='../gtt_data/train.json')
    parser.add_argument('--dev_path',  help='Path to dev dataset', default='../gtt_data/dev.json')
    parser.add_argument('--test_path',  help='Path to test dataset', default='../gtt_data/test.json')
    parser.add_argument('--epochs',  help='Number of Epochs', default=10, type=int)
    parser.add_argument('--batch_size',  help='Batch Size', default=1, type=int)
    parser.add_argument('--max_token_len',  help='Max length of tokens', default=512, type=int)
    parser.add_argument('--workers',  help='Number of workers', default=12, type=int)
    parser.add_argument('--model_path',  help='Model name or path', default='allenai/longformer-base-4096')
    parser.add_argument('--warmup_steps',  help='Number of steps to warmup', default=20, type=int)
    parser.add_argument('--gpu',  help='Number of gpu', default=1, type=int)
    parser.add_argument('--do_train',  help='Train the model', action='store_true')
    parser.add_argument('--save_dir',  help='Train the model', default="checkpoints")
    parser.add_argument('--save_filename',  help='Train the model', default="best-checkpoint")

    args = parser.parse_args()
    # train, dev, test = DataPreprocess(args.train_path, args.dev_path, args.test_path).get_splits()
    # tokenizer = LongformerTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    # label_columns = train.columns.tolist()[2:]
    # steps_per_epoch=len(train) // args.batch_size
    # total_training_steps = steps_per_epoch * args.epochs

    #Temporary paragraph, delete when not using wikievents
    train_f = open("../wikievents/train.json")
    train = json.load(train_f)
    dev_f = open("../wikievents/dev.json")
    dev = json.load(dev_f)
    test_f = open("../wikievents/test.json")
    test = json.load(test_f)
    train_df = pd.DataFrame(columns=['DocID','Text','Attack','Kidnapping','Bombing','Robbery','Arson','Forced'])
    dev_df = pd.DataFrame(columns=['DocID','Text','Attack','Kidnapping','Bombing','Robbery','Arson','Forced'])
    test_df = pd.DataFrame(columns=['DocID','Text','Attack','Kidnapping','Bombing','Robbery','Arson','Forced'])
    for doc_id in train.keys():
        train_df.loc[len(train_df)] = [doc_id,train[doc_id]['doc'],0,0,0,0,0,0]
    for doc_id in dev.keys():
        dev_df.loc[len(dev_df)] = [doc_id,dev[doc_id]['doc'],0,0,0,0,0,0]
    for doc_id in test.keys():
        test_df.loc[len(test_df)] = [doc_id,test[doc_id]['doc'],0,0,0,0,0,0]
    train, dev, test = train_df, dev_df, test_df
    print(train.head)
    tokenizer = LongformerTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    label_columns = ['Attack','Kidnapping','Bombing','Arson','Robbery','Forced']
    steps_per_epoch=len(train) // args.batch_size
    total_training_steps = steps_per_epoch * args.epochs

    data_module = MucDataModule(train, dev, test, tokenizer, workers = args.workers, batch_size=args.batch_size, max_token_len=args.max_token_len)
    model = MucTagger(label_columns, args.model_path, n_classes=len(label_columns), n_warmup_steps=args.warmup_steps, n_training_steps=total_training_steps)
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir, filename=args.save_filename, save_top_k=1, verbose=True, monitor="val_loss", mode="min")
    logger = TensorBoardLogger("lightning_logs", name="muc")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    event_classifier = EventClassifier(model, data_module, logger, checkpoint_callback, [early_stopping_callback], args.epochs, args.gpu)
    saved_model_path = os.path.join(args.save_dir, args.save_filename+'.ckpt')
    if args.do_train:
        event_classifier.train()
        output_data = event_classifier.evaluate(test, tokenizer, args.max_token_len, label_columns, args.model_path, saved_model_path)
    else:
        output_data = event_classifier.evaluate(test, tokenizer, args.max_token_len, label_columns, args.model_path, saved_model_path)
    DataConvert(output_data).assign_data(['Attack','Kidnapping','Bombing'],["perp_individual_id","perp_organization_id","phys_tgt_id","hum_tgt_name","incident_instrument_id"])