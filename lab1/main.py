import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.processing import tokenize_text, process_tokens, count_tokens, generate_vocab_growth_data


files = [
    "bible.txt",
    "quran.txt",
    "abstracts.wiki.txt",
]

heaps = [
    [11.65, 0.54], [3.22, 0.68], [2.89, 0.74]
]

# setup for figs
fig = plt.figure(figsize=(12, 8))
subfigs = fig.subfigures(nrows=len(files), ncols=1)

# go over files, drop stopwords and normalize
for i, filename in enumerate(files):
    file_path = f"data/documents/{filename}"
    print(f"processing: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        processed_tokens = []
        print("Processing")
        # go line by line for efficiency
        for line in file:
            line_tokens = tokenize_text(line)
            line_tokens = process_tokens(line_tokens)
            processed_tokens.extend(line_tokens)
    print("Counting")
    counts = count_tokens(processed_tokens)
    # load the counts into a dataframe
    df = pd.DataFrame.from_dict(counts, orient='index', columns=["token_count"])

    df.sort_values(by="token_count", ascending=False, inplace=True)
    df = df.reset_index()

    subfigs[i].suptitle(filename)
    axs = subfigs[i].subplots(nrows=1, ncols=4)
    
    # Zipf's curve
    x = np.linspace(1, max(df.index))
    axs[0].set_title("Zipf's Law")
    axs[0].plot(df.index, df.token_count)
    axs[0].grid()
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
    print("About to generate")
    # Vocab growth
    vocab_data = generate_vocab_growth_data(processed_tokens, 100)
    vocab_df = pd.DataFrame(vocab_data, columns=["n", "vocab_size"])
    print(vocab_df)
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
