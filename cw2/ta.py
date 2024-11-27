import csv
from math import log2
from collections import defaultdict
# from gensim import corpora, models
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
    
    def calculate_MI(self):
        """
        Calculate Mutual Information for each term in each corpus
        p()
        Returns a dictionary mapping corpus names to lists of (term, MI_score) tuples.
        """
        results = {}
        corpus_tokens = {}
        # total docs
        N = 0
        
        # Count term frequencies per corpus
        for corpus, documents in self.corpora.items():
            N += len(documents)
            corpus_tokens[corpus] = set()
            
            for doc in documents:
                for term in doc:
                    corpus_tokens[corpus].add(term)
        
        # Calculate MI for each term in each corpus
        for corpus in self.corpora.keys():
            results[corpus] = []
            for term in corpus_tokens[corpus]:
                MI = 0
                # docs that both contain the term AND are in the corpus
                n11 = sum([1 for doc in self.corpora[corpus] if term in doc])
                # docs that contain the term but are not in the corpus
                n10 = sum(
                    [
                        sum(
                            [1 for doc in corpus_docs if term in doc and _corpus != corpus]
                        ) for _corpus, corpus_docs in self.corpora.items()
                    ]
                )
                # docs that don't contain the term but are in the corpus
                n01 = sum([1 for doc in self.corpora[corpus] if term not in doc])
                # all docs that contain the term
                n1_ = sum(
                    [
                        sum(
                            [1 for doc in corpus_docs if term in doc]
                        ) for corpus_docs in self.corpora.values()
                    ]
                )
                # all docs that are in the corpus
                # n_1 = len(self.corpora[corpus])
                n_1 = sum([1 for doc in self.corpora[corpus]] )
                # docs that don't contain the term
                n0_ = sum(
                    [
                        sum(
                            [1 for doc in corpus_docs if term not in doc]
                        ) for corpus_docs in self.corpora.values()
                    ]
                )
                # docs that are not in the corpus
                n_0 = sum(
                    [
                        sum([1 for doc in corpus_docs if _corp != corpus] )
                        for _corp, corpus_docs in self.corpora.items()
                    ] 
                )
                # docs that don't contain the term and are not in the corpus
                n00 = sum(
                    [
                        sum(
                            [1 for doc in corpus_docs if term not in doc and _corpus != corpus]
                        ) for _corpus, corpus_docs in self.corpora.items()
                    ]
                )
                
                # Calculate MI score
                for a, b, c in ((n11, n1_, n_1), (n01, n0_, n_1), (n10, n1_, n_0), (n00, n0_, n_0)):
                    try:
                        if a > 0:
                            MI += (a / N) * log2((N * a) / (b * c))
                        else:
                            MI += 0
                    except ZeroDivisionError:
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
        term_counts = {}
        corpus_sizes = {}
        total_docs = 0
        
        # Count term frequencies per corpus
        for corpus, documents in self.corpora.items():
            corpus_sizes[corpus] = len(documents)
            total_docs += len(documents)
            term_counts[corpus] = {}
            
            for doc in documents:
                for term in doc:
                    term_counts[corpus][term] = term_counts[corpus].get(term, 0) + 1
        
        # Calculate Chi-square for each term in each corpus
        for corpus in self.corpora:
            results[corpus] = []
            
            for term in term_counts[corpus]:
                # Observed frequencies
                O11 = term_counts[corpus].get(term, 0)  # term in this corpus
                O12 = sum(counts.get(term, 0) for corpus_name, counts in term_counts.items() 
                        if corpus_name != corpus)  # term in other corpora
                O21 = corpus_sizes[corpus] - O11  # other terms in this corpus
                O22 = sum(size for name, size in corpus_sizes.items() 
                        if name != corpus) - O12  # other terms in other corpora
                
                N = total_docs
                
                # Expected frequencies
                row1 = O11 + O12  # total term occurrences
                row2 = O21 + O22  # total non-term occurrences
                col1 = O11 + O21  # total corpus size
                col2 = O12 + O22  # total other corpora size
                
                E11 = (row1 * col1) / N
                E12 = (row1 * col2) / N
                E21 = (row2 * col1) / N
                E22 = (row2 * col2) / N
                
                # Calculate Chi-square
                if E11 > 0 and E12 > 0 and E21 > 0 and E22 > 0:
                    chi2 = (((O11 - E11) ** 2) / E11 + 
                        ((O12 - E12) ** 2) / E12 +
                        ((O21 - E21) ** 2) / E21 + 
                        ((O22 - E22) ** 2) / E22)
                    results[corpus].append((term, chi2))
            
            # Sort terms by Chi-square score in descending order
            results[corpus] = sorted(results[corpus], key=lambda x: x[1], reverse=True)
        
        return results
    
    # def run_LDA(self, k):
    #     """
    #     Run LDA on the entire set of corpora with k topics.
    #     Compute the average topic distribution for each corpus.
    #     Returns a dictionary mapping corpus names to average topic distributions.
    #     """

    #     # Combine all documents and track their corpus labels
    #     all_documents = []
    #     corpus_labels = []
    #     for corpus, documents in self.corpora.items():
    #         for doc in documents:
    #             all_documents.append(doc)
    #             corpus_labels.append(corpus)

    #     # Create a dictionary and corpus for LDA
    #     dictionary = corpora.Dictionary(all_documents)
    #     corpus_bow = [dictionary.doc2bow(doc) for doc in all_documents]

    #     # Run LDA
    #     lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=k, passes=10)

    #     # Get topic distributions for each document
    #     doc_topics = lda_model.get_document_topics(corpus_bow)

    #     # Initialize per-corpus topic distributions
    #     corpus_topic_sums = defaultdict(lambda: [0.0] * k)
    #     corpus_doc_counts = defaultdict(int)

    #     for i, topics in enumerate(doc_topics):
    #         corpus = corpus_labels[i]
    #         corpus_doc_counts[corpus] += 1
    #         # Convert sparse topic distribution to dense vector
    #         topic_dist = [0.0] * k
    #         for topic_num, prob in topics:
    #             topic_dist[topic_num] = prob
    #         # Sum topic distributions
    #         corpus_topic_sums[corpus] = [sum(x) for x in zip(corpus_topic_sums[corpus], topic_dist)]

    #     # Compute average topic distributions
    #     corpus_topic_avgs = {}
    #     for corpus in self.corpora.keys():
    #         doc_count = corpus_doc_counts[corpus]
    #         if doc_count > 0:
    #             avg_topic_dist = [value / doc_count for value in corpus_topic_sums[corpus]]
    #             corpus_topic_avgs[corpus] = avg_topic_dist

    #     return corpus_topic_avgs, lda_model
    
if __name__ == "__main__":
    # part 2
    ta = TextAnalyser()
    ta.load()
    # Print the top 10 terms by MI score for each corpus
    MI_scores = ta.calculate_MI()
    # token: {col: doc}
    print("MI scores:")
    for corpus, scores in MI_scores.items():
        print(corpus)
        for term, score in scores[:10]:
            print(f"{term}: {round(score, 3)}")
    # Print the top 10 terms by Chi2 score for each corpus
    # Chi2_scores = ta.calculate_Chi2()
    # print("\nChi2 scores:")
    # for corpus, scores in Chi2_scores.items():
    #     print(corpus)
    #     for term, score in scores[:10]:
    #         print(f"{term}: {round(score, 3)}")
    k = 20  # Number of topics
    # topic_avgs, lda_model = ta.run_LDA(k)
    # for corpus, avg_dist in topic_avgs.items():
    #     # Identify the topic with the highest average score
    #     top_topic_index = avg_dist.index(max(avg_dist))
    #     print(f"\n{corpus} - Top Topic {top_topic_index}:")
    #     # Get the top 10 tokens for this topic
    #     top_tokens = lda_model.show_topic(top_topic_index, topn=10)
    #     print("Top 10 tokens:")
    #     for token, prob in top_tokens:
    #         print(f"{token}: {round(prob, 4)}")
