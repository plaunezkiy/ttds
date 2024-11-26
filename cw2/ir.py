import csv
import heapq
from math import log2


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
        with open("data/cw2/ir_eval.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["system_number", "query_number", "P@10", "R@50", "r-precision", "AP", "nDCG@10", "nDCG@20"])
            
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

                # calculate mean metrics
                means = []
                for metric in [ps, rs, r_ps, aps, nDCG_10s, nDCG_20s]:
                    mean = sum(metric) / len(metric)
                    means.append(round(mean, 3))
                # save the results
                writer.writerow([sys_n, "mean", *means])


    def load_data(self):
        """
        Load the system results and query relevant data
        """
        # load system results
        data = csv.reader(open("data/collections/system_results.csv"), delimiter=",")
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
        data = csv.reader(open("data/collections/qrels.csv"), delimiter=",")
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
        for k in range(1, len(docs)):
            relevance = self.qrel[query_n].get(docs[k].doc_id, 0) # this or 1?
            precision += self.precision(sys_n, query_n, k) * relevance
        return precision / len(self.qrel[query_n])

    def DCG(self, sys_n, query_n, N):
        """
        Calculate the Discounted Cumulative Gain at cutoff N
        DCG@N = rel(1) + sum((rel(i) / log2(i)) for i in 2 to N)
        """
        # get the relevance of the first document
        doc1_id = self.systems[sys_n][query_n][0].doc_id
        rel1 = self.qrel[query_n].get(doc1_id, 0)
        # calculate the DCG
        DCG = rel1
        for i in range(2, N):
            doc_id = self.systems[sys_n][query_n][i].doc_id
            rel = self.qrel[query_n].get(doc_id, 0)
            DCG += rel / log2(i)
        return DCG
    
    def iDCG(self, sys_n, query_n, N):
        """
        Calculate the Ideal Discounted Cumulative Gain at cutoff N
        Doc ranking first sorted in order of relevance
        """
        docs = self.systems[sys_n][query_n]
        # sort the documents by relevance
        docs = sorted(docs, key=lambda x: x.score, reverse=True)
        # get the relevance of the first document
        doc1_id = docs[0].doc_id
        rel1 = self.qrel[query_n].get(doc1_id, 0)
        # calculate the iDCG
        iDCG = rel1
        for i in range(2, N):
            doc_id = docs[i].doc_id
            rel = self.qrel[query_n].get(doc_id, 0)
            iDCG += rel / log2(i)
        return iDCG

    def nDCG(self, sys_n, query_n, N):
        """
        Calculate the Normalized Discounted Cumulative Gain at cutoff N
        nDCG@N = DCG@N / iDCG@N
        DCG@N = rel(1) + sum((rel(i) / log2(i)) for i in 2 to N) (actual ranking)
        iDCG@N - Ideal DCG at cutoff N (ALL relevant come first in order of relevance)
        """
        DCGN = self.DCG(sys_n, query_n, N)
        iDCGN = self.iDCG(sys_n, query_n, N)
        # avoid division by zero
        if iDCGN == 0:
            return 0
        return DCGN / iDCGN


if __name__ == "__main__":
    # part 1
    ir_eval = IR_EVAL()
    ir_eval.load_data()
    ir_eval.eval()
