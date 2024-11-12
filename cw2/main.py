import csv


class IR_EVAL:
    def __init__(self):
        pass
    
    def eval(self):
        pass

    def load_data(self):
        pass

    def precision(self, N):
        """
        Calculate the Precision at cutoff N
        Proportion of the relevant documents in top N docs
        """
        pass

    def recall(self, N):
        """
        Calculate the Recall at cutoff N
        Proportion of all relevant documents top N docs
        """
        pass

    def rPrecision(self, R):
        """
        Calculate the R-Precision
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


if __name__ == "__main__":
    IR_EVAL().eval()
