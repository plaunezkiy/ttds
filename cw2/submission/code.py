import csv
import heapq
import torch
from tqdm import tqdm
import pandas as pd
from random import shuffle
from datasets import Dataset
from math import log2
import scipy.stats as stats
from scipy.sparse import dok_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from typing import List
from nltk.stem import PorterStemmer
import re


alphanum = r"[a-zA-Z0-9_-]*"
non_alphanum = r"[^a-zA-Z0-9_-]"
stemmer = PorterStemmer()
# load stopwords
with open("stopwords.txt", "r") as stopwords_file:
    stopwords = set(stopwords_file.read().strip().split("\n"))


def tokenize_text(text: str) -> List[str]:
    """
    lowercases everything, converts non-alphanumeric chars into newlines
    returns an array of tokens, split by newline (drops empty strings)
    """
    text = text.lower()
    text = re.sub(non_alphanum, "\n", text)
    return list(filter(lambda token: bool(token), text.split("\n")))


def remove_stopwords(collection: List[str]) -> List[str]:
    """
    Drops all entries that are in the stopword list
    """
    return list(filter(lambda token: token not in stopwords, collection))


def normalize(collection: List[str]) -> List[str]:
    return list(map(lambda token: stemmer.stem(token), collection))


def process_tokens(tokens: List[str]):
    ts = remove_stopwords(tokens)
    ts = normalize(ts)
    return ts



class RankedDocument:
    """
    Instance of a ranked document in a IR system
    """
    def __init__(self, doc_id, rank, relevance_score):
        self.doc_id = doc_id
        self.rank = int(rank)
        self.score = float(relevance_score)
    
    def __repr__(self):
        return f"Doc {self.doc_id} @ {self.rank} with {self.score}"

    def __lt__(self, other):
        return self.rank < other.rank


class IR_EVAL:
    def __init__(self):
        # systems dict - {  system.id: {    query.id: heap[ (doc_id, relevance) ]   } }
        #                                                   ^ position = rank
        self.systems = {}
        # query relevant data dict - {  query.id: {doc.id: relevance}   }
        self.qrel = {}

    def eval(self):
        # export to `ir_eval.csv`
        # system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20
        with open("ir_eval.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"])
            system_eval_data = {}
            # calculate the metrics for each system and query
            for sys_n, queries in self.systems.items():
                ps = []
                rs = []
                r_ps = []
                aps = []
                nDCG_10s = []
                nDCG_20s = []
                for query_n, docs in queries.items():
                    # calculate the metrics
                    # P@10
                    P_10 = round(self.precision(sys_n, query_n, 10), 3)
                    ps.append(P_10)
                    # R@50
                    R_50 = round(self.recall(sys_n, query_n, 50), 3)
                    rs.append(R_50)
                    # r-Precision
                    r_P = round(self.rPrecision(sys_n, query_n), 3)
                    r_ps.append(r_P)
                    # AP
                    AP = round(self.averagePrecision(sys_n, query_n), 3)
                    aps.append(AP)
                    # nDCG@10
                    nDCG_10 = round(self.nDCG(sys_n, query_n, 10), 3)
                    nDCG_10s.append(nDCG_10)
                    # nDCG@20
                    nDCG_20 = round(self.nDCG(sys_n, query_n, 20), 3)
                    nDCG_20s.append(nDCG_20)
                    # save the row of results
                    writer.writerow([sys_n, query_n, P_10, R_50, r_P, AP, nDCG_10, nDCG_20])
                system_eval_data[sys_n] = [ps, rs, r_ps, aps, nDCG_10s, nDCG_20s]
                # calculate mean metrics
                means = []
                for metric in [ps, rs, r_ps, aps, nDCG_10s, nDCG_20s]:
                    mean = sum(metric) / len(metric)
                    means.append(round(mean, 3))
                # save the results
                writer.writerow([sys_n, "mean", *means])
            return system_eval_data


    def load_data(self):
        """
        Load the system results and query relevant data
        """
        # load system results
        data = csv.reader(open("system_results.csv"), delimiter=",")
        next(data, None) # skip the headers
        for row in data:
            sys_n, query_n, doc_n, rank, score = row
            self.systems[sys_n] = self.systems.get(sys_n, {})
            self.systems[sys_n][query_n] = self.systems[sys_n].get(query_n, [])
            self.systems[sys_n][query_n].append(RankedDocument(doc_n, rank, score))
        # convert the lists to max heaps
        for sys_n, queries in self.systems.items():
            for query_n, docs in queries.items():
                # heapify to guarantee the max heap property
                heapq.heapify(docs)
                self.systems[sys_n][query_n] = docs
        # load query relevant data
        data = csv.reader(open("qrels.csv"), delimiter=",")
        next(data, None) # skip the headers
        for row in data:
            query_n, doc_n, relevance = row
            self.qrel[query_n] = self.qrel.get(query_n, {})
            self.qrel[query_n][doc_n] = float(relevance)

    def precision(self, sys_n, query_n, N):
        """
        Calculate the Precision at cutoff N for a given system and query
        Proportion of the relevant documents in top N docs
        """
        # get top N documents for the query for the system
        ret_res = self.systems[sys_n][query_n][:N]
        # get the relevant documents for the query
        q_rel_docs = self.qrel[query_n]
        # count the relevant documents in the top N
        rel_docs = sum([doc.doc_id in q_rel_docs for doc in ret_res])
        return rel_docs / N

    def recall(self, sys_n, query_n, N):
        """
        Calculate the Recall at cutoff N for a given system and query
        Proportion of all relevant documents in top N docs
        """
        # get top N documents for the query for the system
        ret_res = self.systems[sys_n][query_n][:N]
        # get the relevant documents for the query
        q_rel_docs = self.qrel[query_n]
        # count the relevant documents in the top N
        rel_docs = sum([doc.doc_id in q_rel_docs for doc in ret_res])
        return rel_docs / len(q_rel_docs)

    def rPrecision(self, sys_n, query_n):
        """
        Calculate the R-Precision for a given system and query
        For a query with R relevant documents, the R-Precision is P@R
        """
        return self.precision(sys_n, query_n, len(self.qrel[query_n]))

    def averagePrecision(self, sys_n, query_n):
        """
        Calculate the Average Precision
        AP = 1/r * sum(P@k * rel(k))
        r - number of relevant documents
        P@k - Precision at cutoff k
        rel(k) - is the relevance of item at rank k (1/0)
        """
        docs = self.systems[sys_n][query_n]
        precision = 0
        for k in range(1, len(docs)+1):
            p_k = self.precision(sys_n, query_n, k)
            if docs[k-1].doc_id in self.qrel[query_n]:
                precision += p_k
        return precision / len(self.qrel[query_n])

    def DCG(self, sys_n, query_n, N, ideal=False):
        """
        Calculate the Discounted Cumulative Gain at cutoff N
        DCG@N = rel(1) + sum((rel(i) / log2(i)) for i in 2 to N)
        :param ideal: if True, calculate the iDCG, else DCG
        """
        # get the relevance scores for the top N documents
        docs = self.systems[sys_n][query_n][:N]
        rels = [self.qrel[query_n].get(doc.doc_id, 0) for doc in docs]
        if ideal:
            # ALL relevant come first in order of relevance
            # the rest (N - n_rel) have 0 relevance score
            docs = [doc for doc in self.qrel[query_n]] + [*['-1'] * (N - len(self.qrel[query_n]))]
            docs = sorted(docs, key=lambda x: self.qrel[query_n].get(x, 0), reverse=True)
            rels = [self.qrel[query_n].get(doc, 0) for doc in docs]
        rel1 = rels[0]
        # calculate the DCG
        DCG = rel1
        for i in range(2, N+1):
            rel = rels[i-1]
            DCG += (rel / log2(i))
        return DCG

    def nDCG(self, sys_n, query_n, N):
        """
        Calculate the Normalized Discounted Cumulative Gain at cutoff N
        nDCG@N = DCG@N / iDCG@N
        DCG@N = rel(1) + sum((rel(i) / log2(i)) for i in 2 to N) (actual ranking)
        iDCG@N - Ideal DCG at cutoff N (ALL relevant come first in order of relevance)
        """
        DCG = self.DCG(sys_n, query_n, N)
        iDCG = self.DCG(sys_n, query_n, N, ideal=True)
        # avoid division by zero
        if iDCG == 0:
            return 0
        return DCG / iDCG

    def find_best_system(self, eval_data):
        """
        Find whether the most averaging system is the best
        for each metric
        """
        for i, metric in enumerate(["P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"]):
            all_means = [round(sum(data[i])/len(data[i]), 3) for data in eval_data.values()]
            print()
            second_best_sys = [sys_n for i,sys_n in enumerate(eval_data.keys()) if all_means[i] == sorted(set(all_means))[-2]]
            best_sys = [sys_n for i, sys_n in enumerate(eval_data.keys()) if all_means[i] == max(all_means)]
            for b_sys_n in best_sys:
                for sb_sys_n in second_best_sys:
                    # t test
                    result = stats.ttest_rel(eval_data[b_sys_n][i], eval_data[sb_sys_n][i])
                    print(f"{metric} - {b_sys_n} vs {sb_sys_n}: {round(result.pvalue, 3)}")
                    if result.pvalue <= 0.05:
                        print(f"{b_sys_n} is better than {sb_sys_n} for {metric}")


class TextAnalyser:
    def __init__(self):
        self.corpora = {}
    
    def load(self):
        with open("data/collections/bible_and_quran.tsv", "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                corpus, verse = row
                tokens = process_tokens(tokenize_text(verse))
                self.corpora[corpus] = self.corpora.get(corpus, []) + [tokens]
        self.precompute_N()
    
    def precompute_N(self):
        corpus_tokens = {}
        token_index = {}
        # total docs
        N = 0
        
        # Count term frequencies per corpus
        for corpus, documents in self.corpora.items():
            N += len(documents)
            corpus_tokens[corpus] = set()
            
            for doc in documents:
                for term in set(doc):
                    token_index[term] = token_index.get(term, {})
                    token_index[term][corpus] = token_index[term].get(corpus, 0) + 1
                    corpus_tokens[corpus].add(term)
        self.corpus_tokens = corpus_tokens
        self.token_index = token_index

        def get_ns(term, corpus):
            token_index = self.token_index
            # docs that both contain the term AND are in the corpus
            n11 = token_index[term].get(corpus, 0)
            # docs that contain the term but are not in the corpus
            n10 = sum([sum([token_index[term].get(_corp, 0) if _corp != corpus else 0]) for _corp in self.corpora.keys()])
            # docs that don't contain the term but are in the corpus
            n01 = len(self.corpora[corpus]) - n11
            # all docs that contain the term
            n1_ = sum([sum([token_index[term].get(_corp, 0)]) for _corp in self.corpora.keys()])
            # all docs that are in the corpus
            n_1 = len(self.corpora[corpus])
            # docs that don't contain the term
            n0_ = N - n1_
            # docs that are not in the corpus
            n_0 = N - n_1
            # docs that don't contain the term and are not in the corpus
            n00 = N - n_1 - n10
            return n11, n10, n01, n1_, n_1, n0_, n_0, n00, N
        #
        self.get_ns = get_ns

    def calculate_MI(self):
        """
        Calculate Mutual Information for each term in each corpus
        p()
        Returns a dictionary mapping corpus names to lists of (term, MI_score) tuples.
        """
        results = {}
        
        # Calculate MI for each term in each corpus
        for corpus in self.corpora.keys():
            results[corpus] = []
            for term in self.token_index.keys():
                MI = 0
                n11, n10, n01, n1_, n_1, n0_, n_0, n00, N = self.get_ns(term, corpus)
                # Calculate MI score
                for a, b, c in ((n11, n1_, n_1), (n01, n0_, n_1), (n10, n1_, n_0), (n00, n0_, n_0)):
                    try:
                        f = (a / N) * log2((N * a) / (b * c))
                        # if term == "jesu" and corpus == "OT":
                        #     print(f)
                        MI += f
                    except (ValueError, ZeroDivisionError):
                        MI += 0
                results[corpus].append((term, MI))
            
            # Sort terms by MI score in descending order
            results[corpus] = sorted(results[corpus], key=lambda x: x[1], reverse=True)
        
        return results

    def calculate_Chi2(self):
        """
        Calculate Chi-squared for each term in each corpus
        Returns a dictionary mapping corpus names to lists of (term, chi2_score) tuples.
        """
        results = {}
        
        # Calculate Chi-square for each term in each corpus
        for corpus in self.corpora.keys():
            results[corpus] = []
            
            for term in self.token_index.keys():
                n11, n10, n01, n1_, n_1, n0_, n_0, n00, N = self.get_ns(term, corpus)
                chi2 = 0

                numerator = (n11 + n10 + n01 + n00) * (n11 * n00 - n10 * n01)**2
                denominator = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
                try:
                    chi2 = numerator / denominator
                except ZeroDivisionError:
                    chi2 = 0

                results[corpus].append((term, chi2))
            
            # Sort terms by Chi-square score in descending order
            results[corpus] = sorted(results[corpus], key=lambda x: x[1], reverse=True)
        
        return results
    
    def run_LDA(self, k):
        """
        Run LDA on the entire set of corpora with k topics.
        Compute the average topic distribution for each corpus.
        Returns a dictionary mapping corpus names to average topic distributions.
        """

        # Combine all documents and track their corpus labels
        all_documents = []
        for documents in self.corpora.values():
            for doc in documents:
                all_documents.append(doc)

        # Create a dictionary and corpus for LDA
        dictionary = corpora.Dictionary(all_documents)
        corpus_bow = [dictionary.doc2bow(doc) for doc in all_documents]

        # Run LDA
        lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=k, random_state=15)
        
        # Compute the average topic distribution for each corpus
        corpus_topic_avgs = {}
        for c in self.corpora.keys():
            c_docs = [dictionary.doc2bow(doc) for doc in self.corpora[c]]
            for doc in c_docs:
                topic_scores = lda_model.get_document_topics(doc, minimum_probability=0.0)
                for topic_num, prob in topic_scores:
                    corpus_topic_avgs[c] = corpus_topic_avgs.get(c, [0.0] * k)
                    corpus_topic_avgs[c][topic_num] += prob
            for i in range(k):
                corpus_topic_avgs[c][i] /= len(c_docs)

        # for each corpus, find the topic with the highest average score
        for corpus, avg_dist in corpus_topic_avgs.items():
            top_topic_index = avg_dist.index(max(avg_dist))
            print()
            print([round(a, 4) for a in avg_dist])
            print(f"{corpus} - Top Topic {top_topic_index}: {round(max(avg_dist), 4)}")
            # find the top 10 tokens for the top topic
            top_tokens = lda_model.show_topic(top_topic_index, topn=10)
            print("Top 10 tokens:")
            for token, prob in top_tokens:
                print(f"{token} {round(float(prob), 4)}")
        
        return corpus_topic_avgs, lda_model


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
    # part 1 IR
    ir_eval = IR_EVAL()
    ir_eval.load_data()
    eval_data = ir_eval.eval()
    ir_eval.find_best_system(eval_data)
    
    # part 2 Text Analysis
    ta = TextAnalyser()
    ta.load()
    # Print the top 10 terms by MI score for each corpus
    MI_scores = ta.calculate_MI()
    print("MI scores:")
    for corpus, scores in MI_scores.items():
        print(corpus)
        for term, score in scores[:10]:
            print(f"{term},{round(score, 3)}")
    print()
    # Print the top 10 terms by Chi2 score for each corpus
    Chi2_scores = ta.calculate_Chi2()
    print("Chi2 scores:")
    for corpus, scores in Chi2_scores.items():
        print(corpus)
        for term, score in scores[:10]:
            print(f"{term},{round(score, 3)}")
    print()
    k = 20  # Number of topics
    topic_avgs, lda_model = ta.run_LDA(k)

    # part 3 Classification
    c = Classification()
    # fine tuning takes ~2 hours
    # c.finetune()
    c.train_and_eval()

