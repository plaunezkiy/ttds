from scipy.sparse import dok_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from random import shuffle
import csv
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from utils.processing import tokenize_text, process_tokens

class Classification:
    def preprocess_data(self, data):
        documents = []
        categories = []
        vocab = set()
        # Skip the header
        lines = data.split("\n")[1:]
        shuffle(lines)
        
        for line in lines:
            if not line:
                continue
            tweet_id, category, tweet = line.split("\t")
            tokens = tokenize_text(tweet)
            # processed_tokens = process_tokens(tokens)
            processed_tokens = tokens
            documents.append(processed_tokens)
            categories.append(category)
            vocab.update(processed_tokens)
        
        word2id = {word: i for i, word in enumerate(vocab)}
        cat2id = {cat: i for i, cat in enumerate(set(categories))}

        return documents, categories, vocab, word2id, cat2id

    def convert_to_bow(self, data, word2id):
        matrix_size = (len(data), len(word2id)+1)
        oov_index = len(word2id)
        bow = dok_matrix(matrix_size)
        for doc_id, doc in enumerate(data):
            for word in doc:
                word_id = word2id.get(word, oov_index)
                bow[doc_id, word_id] += 1
        return bow
    
    def export_results(reports):
        # reports: [{system: str, split: str, report: classification_report}]
        with open("data/cw2/classification.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow("system,split,p-pos,r-pos,f-pos,p-neg,r-neg,f-neg,p-neu,r-neu,f-neu,p-macro,r-macro,f-macro".split(","))
            # 
            for report in reports:
                metrics = []
                # 
                for cat in ["positive", "negative", "neutral"]:
                    data = report["report"][cat]
                    for metric in ["precision", "recall", "f1-score"]:
                        metrics.append(round(data[metric], 3))
                # 
                macros = report["report"]["macro avg"]
                for metric in ["precision", "recall", "f1-score"]:
                    metrics.append(round(macros[metric], 3))
                # 
                writer.writerow([report["system"], report["split"], *metrics])


    def train_and_eval(self):
        train_data = open('data/collections/train.txt', encoding="utf-8").read()
        test_data = open('data/collections/test.txt', encoding="utf-8").read()
        train_docs, train_cats, train_vocab, word2id, cat2id = self.preprocess_data(train_data)
        cat_names = []
        for cat,cid in sorted(cat2id.items(),key=lambda x:x[1]):
            cat_names.append(cat)
        # baseline data
        X = train_docs 
        Y = [cat2id[cat] for cat in train_cats]
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_BoW = self.convert_to_bow(X_train, word2id)
        X_dev_BoW = self.convert_to_bow(X_dev, word2id)
        # print cats and ids
        print(cat2id)
        # train SVC
        model = SVC(C=1000, kernel='linear')
        model.fit(X_train_BoW, Y_train)

        reports = []
        # baseline train data report
        Y_train_pred = model.predict(X_train_BoW)
        train_report = classification_report(Y_train, Y_train_pred, output_dict=True, target_names=cat_names)
        reports.append({"system": "baseline", "split": "train", "report": train_report})
        print(classification_report(Y_train, Y_train_pred, target_names=cat_names))
        # baseline dev data report
        Y_dev_pred = model.predict(X_dev_BoW)
        dev_report = classification_report(Y_dev, Y_dev_pred, output_dict=True, target_names=cat_names)
        reports.append({"system": "baseline", "split": "dev", "report": dev_report})
        print(classification_report(Y_dev, Y_dev_pred, target_names=cat_names))
        # print 3 misclassified examples from the dev set
        cnt = 0
        for i, (gold, pred) in enumerate(zip(Y_dev, Y_dev_pred)):
            if gold != pred:
                cnt += 1
                # labels
                print("Gold:", cat_names[gold], "Pred:", cat_names[pred])
                # text
                # print(X_dev[i])
                print(" ".join(X_train[i]))
                print()
            if cnt == 3:
                break
        # baseline test data report
        test_docs, test_cats, _, _, _ = self.preprocess_data(test_data)
        X_test_BoW = self.convert_to_bow(test_docs, word2id)
        Y_test = [cat2id[cat] for cat in test_cats]
        Y_test_pred = model.predict(X_test_BoW)
        test_report = classification_report(Y_test, Y_test_pred, output_dict=True, target_names=cat_names)
        reports.append({"system": "baseline", "split": "test", "report": test_report})
        print(classification_report(Y_test, Y_test_pred, target_names=cat_names))
        # DistilBERT results
        model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")
        # evaluate on all splits
        splits = ["dev", "train", "test"]
        for i, (collection, labels) in enumerate([(X_dev, Y_dev), (X_train, Y_train), (test_docs, Y_test)]):
            tweets = [" ".join(doc) for doc in collection]
            print()
            print(f"Processing {splits[i]} set")
            # do in batches of size 20
            N = 20
            preds = []
            for j in tqdm(range(0, len(tweets), N)):
                inputs = tokenizer(tweets[j:j+N], padding=True, truncation=True, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                ps = torch.argmax(logits, dim=1).tolist()
                if j == 0:
                    print(ps)
                preds.extend(ps)
            # get classification report
            report = classification_report(labels, preds, output_dict=True, target_names=cat_names)
            print(classification_report(labels, preds, target_names=cat_names))
            reports.append({"system": "improved", "split": splits[i], "report": report})
        # export results
        self.export_results(reports)
    
    def finetune(self):
        # Load CSV files
        train_df = pd.read_csv('data/collections/train.txt', sep='\t')
        test_df = pd.read_csv('data/collections/test.txt', sep='\t')
        # Map sentiment to label
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        train_df['label'] = train_df['sentiment'].map(label_map)
        test_df['label'] = test_df['sentiment'].map(label_map)
        # split and create datasets
        test_dataset = Dataset.from_pandas(test_df[['tweet', 'label']])
        train_dataset, val_dataset = train_test_split(train_df, test_size=0.1, random_state=42)
        train_dataset = Dataset.from_pandas(train_dataset[['tweet', 'label']])
        val_dataset = Dataset.from_pandas(val_dataset[['tweet', 'label']])
        # 
        # Load the pre-trained DistilBERT tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Load the pre-trained DistilBERT model for sequence classification
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
        # Tokenize the datasets
        def tokenize_function(examples):
            return tokenizer(examples['tweet'], padding='max_length', truncation=True, max_length=128)
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        # Check if GPU is available and move model to GPU if so
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,  # Increase batch size
            per_device_eval_batch_size=64,   # Larger evaluation batch size
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            fp16=True,
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps if batch size is too large
        )
        # Define the Trainer
        trainer = Trainer(
            model=model,                         # the model to be trained
            args=training_args,                  # training arguments
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            tokenizer=tokenizer,                 # tokenizer to handle the tokenization
        )
        # Train the model
        trainer.train()
        eval_results = trainer.evaluate()
        print(eval_results)
        # Save the model and tokenizer
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")


if __name__ == "__main__":
    c = Classification()
    c.finetune()
    c.train_and_eval()
