import csv
from math import log2
from collections import defaultdict
from gensim import corpora, models
from utils.processing import tokenize_text, process_tokens

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
    
if __name__ == "__main__":
    # part 2
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
