import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer


ps = PorterStemmer()
files = [
    "bible.txt",
    "quran.txt",
    "abstracts.wiki.txt",
]

zipfs = [
    10000, 10000, 10000
]
heaps = [
    [7, 0.54], [10, 0.54], [70, 0.54]
]


fig = plt.figure(figsize=(12, 8))
subfigs = fig.subfigures(nrows=len(files), ncols=1)

# load stopwords
with open("data/stopwords.txt", "r") as stopwords_file:
    stopwords = stopwords_file.read().strip().split("\n")

# go over files, drop stopwords and normalize
for i, file in enumerate(files):
    token_counts_file = f"output/token_counts_{file}"
    with open(token_counts_file, "r") as counts_file:
        data = counts_file.read()
        counts = list(map(lambda r: r.strip().split(), data.strip().split("\n")))
    # load the counts into a dataframe
    df = pd.DataFrame(counts, columns=["token_count", "token"])
    # cast str counts to ints
    df.token_count = df.token_count.apply(lambda c: int(c))
    # drop stopwords
    no_stopwords_df = df[~df["token"].isin(stopwords)]

    # normalize (replace the word with its stemmed form)
    no_stopwords_df.token.apply(lambda token: ps.stem(token or " "))
    # 
    # stemmed_df = no_stopwords_df[no_stopwords_df.token.duplicated()].groupby("token").token_count.sum()
    stemmed_df = no_stopwords_df[no_stopwords_df.token.duplicated()].groupby("token").token_count.sum()
    
    print(no_stopwords_df.token.duplicated().any())

    df.sort_values(by="token_count", ascending=False, inplace=True)
    df = df.reset_index()
    
    subfigs[i].suptitle(file)
    axs = subfigs[i].subplots(nrows=1, ncols=4)
    
    # Zipf's curve
    k = zipfs[i]
    x = np.linspace(1, max(df.index))
    axs[0].set_title("Zipf's Law")
    axs[0].plot(df.index, df.token_count)
    axs[0].grid()
    axs[0].plot(x, k / x)
    axs[0].legend(["Observed", "$f={0}/r$".format(k)])
    axs[0].set_xlabel("log(rank)")
    axs[0].set_ylabel("log(frequency)")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    
    # Benford's curve (all digits)
    first_digits = df.token_count.apply(lambda n: int(str(n)[0]))

    axs[1].set_title("Benford's Law (all)")
    axs[1].hist(first_digits, bins=first_digits.unique().sort(), rwidth=0.5)
    axs[1].set_xlabel("digit")
    axs[1].tick_params(axis='x', labelrotation=45)
    axs[1].set_ylabel("log(frequency)")
    # axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    # Benford's curve (non-one digits)
    first_digits = df[df.token_count > 10].token_count.apply(lambda n: int(str(n)[0]))

    axs[2].set_title("Benford's Law (>10)")
    axs[2].hist(first_digits, bins=first_digits.unique().sort(), rwidth=0.5)
    axs[2].set_xlabel("digit")
    axs[2].set_ylabel("log(frequency)")
    axs[2].set_yscale("log")

    # Vocab growth
    vocab_df = pd.read_csv(f"output/vocab_{file}")
    # vocab_df.plot(x=vocab_df.n, y=vocab_df.vocab_size, ax=axs[3])
    k, b = heaps[i]
    x = np.linspace(0, max(vocab_df.n))
    axs[3].plot(vocab_df.n, vocab_df.vocab_size)
    axs[3].grid()
    axs[3].plot(x, k * np.power(x, b))
    axs[3].legend(["Observed", "$V={0}n^{{{1}}}$".format(k, b)])
    axs[3].set_title("Heaps' Law")
    axs[3].set_xlabel("Total words")
    axs[3].set_ylabel("Vocab size")


# show the figure
plt.show()
