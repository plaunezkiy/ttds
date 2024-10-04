import os
import pandas as pd
from nltk.stem import PorterStemmer

ps = PorterStemmer()

token_counts_file = "token_counts.txt"

with open(token_counts_file, "r") as counts_file:
    data = counts_file.read()

    counts = list(map(lambda r: r.strip().split(), data.strip().split("\n")))

with open("data/stopwords.txt", "r") as stopwords_file:
    stopwords = stopwords_file.read().strip().split("\n")

df = pd.DataFrame(counts, columns=["token_count", "token"])

no_stopwords_df = df[~df["token"].isin(stopwords)]

for i, row in no_stopwords_df.iterrows():
    # ["token"].values:
    row.token = ps.stem(row.token or " ")


stemmed_df = no_stopwords_df[no_stopwords_df.duplicated()].groupby("token").token_count.sum()

print(no_stopwords_df)
df = no_stopwords_df.merge(stemmed_df, left_on="token", right_index=True, how='outer')
df.token_count_y.fillna(0, inplace=True)
df.token_count_x = pd.to_numeric(df.token_count_x)
df.token_count_y = pd.to_numeric(df.token_count_y)

df["token_count"] = df.token_count_x + df.token_count_y
df = df[["token", "token_count"]]

df.sort_values(by="token_count", ascending=False, inplace=True)
df = df.reset_index()
print(df)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

df.plot.line(y="token_count", ax=ax, loglog=True)
plt.show()
