files = [
    "bible.txt",
    "quran.txt",
    "abstracts.wiki.txt",
]

# every 100 words take a note of the vocab size
n = 100

for file in files:
    text_file = f"output/tokens_{file}"
    vocab = set()
    data = [["n", "vocab_size"]]
    with open(text_file, "r") as counts_file:
        i = 0
        for token in counts_file:
            token = token.strip()
            if not token:
                continue
            if i % n == 0:
                data.append([i, len(vocab)])
            vocab.add(token)
            i += 1
        with open(f"output/vocab_{file}", "w") as vocab_file:
            vocab_file.write(
                "\n".join(
                    [",".join(map(str, item)) for item in data]
                )
            )

