from scipy.sparse import dok_matrix
from sklearn.svm import SVC
from utils.processing import tokenize_text, process_tokens


def convert_to_bow(data, word2id):
    matrix_size = (len(data), len(word2id)+1)
    oov_index = len(word2id)
    bow = dok_matrix(matrix_size)
    for doc_id, doc in enumerate(data):
        for word in doc:
            word_id = word2id.get(word, oov_index)
            bow[doc_id, word_id] += 1
    return bow


def preprocess_data(data):
    documents = []
    categories = []
    vocab = set()

    for line in data.split("\n"):
        tweet_id, category, tweet = line.split("\t")
        tokens = tokenize_text(tweet)
        processed_tokens = process_tokens(tokens)
        documents.append(processed_tokens)
        categories.append(category)
        vocab.update(processed_tokens)

    word2id = {word: i for i, word in enumerate(vocab)}
    cat2id = {cat: i for i, cat in enumerate(set(categories))}

    return documents, categories, vocab, word2id, cat2id


def train():
    # test_data = open('data/collections/test.txt', encoding="utf-8").read()
    train_data = open('data/collections/train.txt', encoding="utf-8").read()
    train_docs, train_cats, train_vocab, word2id, cat2id = preprocess_data(train_data)
    # baseline
    X_train = convert_to_bow(train_docs, word2id)
    Y_train = [cat2id[cat] for cat in train_cats]
    model = SVC(C=1000, kernel='linear')
    model.fit(X_train, Y_train)

    # ytrn_pred = model.predict(Strn)
    # ydev_pred = model.predict(Sdev)
    # ytst_pred = model.predict(Stst)

    # CalAcc(ytrn_pred, ytrn, 'baseline', 'train')
    # CalAcc(ydev_pred, ydev, 'baseline', 'dev')
    # CalAcc(ytst_pred, ytst, 'baseline', 'test')

    # ThreeIns(ydev_pred, ydev, Xdev)
    return

print(train_data)