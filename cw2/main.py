import csv


class IR_EVAL:
    def __init__(self):
        # systems dict - {  system.id: {    query.id: { doc.id: (rank, relevance) }   } }
        self.systems = {}
        # query relevant data dict - {  query.id: {doc.id: relevance}   }
        self.qrel = {}

    def eval(self):
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
                P_10 = self.precision(sys_n, query_n, 10)
                ps.append(P_10)
                # R@50
                R_50 = self.recall(sys_n, query_n, 50)
                rs.append(R_50)
                # r-Precision
                r_P = self.rPrecision(sys_n, query_n)
                r_ps.append(r_P)
                # AP
                AP = self.averagePrecision(sys_n, query_n)
                aps.append(AP)
                # nDCG@10
                nDCG_10 = self.nDCG(sys_n, query_n, 10)
                nDCG_10s.append(nDCG_10)
                # nDCG@20
                nDCG_20 = self.nDCG(sys_n, query_n, 20)
                nDCG_20s.append(nDCG_20)
                # save the results
            # calculate mean metrics
            # save the results
        # export to `ir_eval.csv`
        # system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20


    def load_data(self):
        # load system results
        data = csv.reader(open("data/cw2/system_results.csv"), delimiter=",")
        next(data, None) # skip the headers
        for row in data:
            sys_n, query_n, doc_n, rank, score = row
            self.systems[sys_n] = self.systems.get(sys_n, {})
            self.systems[sys_n] = self.systems[sys_n].get(query_n, {})
            self.systems[sys_n][doc_n] = (rank, score)
        # load query relevant data
        data = csv.reader(open("data/cw2/qrels.csv"), delimiter=",")
        next(data, None) # skip the headers
        for row in data:
            query_n, doc_n, relevance = row
            self.qrel[query_n] = self.qrel.get(query_n, {})
            self.qrel[query_n][doc_n] = relevance

    def precision(self, sys_n, query_n, N):
        """
        Calculate the Precision at cutoff N for a given system and query
        Proportion of the relevant documents in top N docs
        """
        pass

    def recall(self, sys_n, query_n, N):
        """
        Calculate the Recall at cutoff N for a given system and query
        Proportion of all relevant documents in top N docs
        """
        pass

    def rPrecision(self, sys_n, query_n):
        """
        Calculate the R-Precision for a given system and query
        For a query with R relevant documents, the R-Precision is P@R
        """
        pass

    def averagePrecision(self):
        """
        Calculate the Average Precision
        AP = 1/r * sum(P@k * rel(k))
        r - number of relevant documents
        P@k - Precision at cutoff k
        rel(k) - is the relevance of item at rank k (1/0)
        """
        pass

    def nDCG(self, sys_n, query_n, N):
        """
        Calculate the Normalized Discounted Cumulative Gain at cutoff N
        nDCG@N = DCG@N / iDCG@N
        DCG@N = rel(1) + sum((rel(i) / log2(i)) for i in 2 to N) (actual ranking)
        iDCG@N - Ideal DCG at cutoff N (ALL relevant come first in order of relevance)
        """
        pass


if __name__ == "__main__":
    # part 1
    ir_eval = IR_EVAL()
    ir_eval.load_data()
    ir_eval.eval()
