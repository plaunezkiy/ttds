FNAME="abstracts.wiki.txt"
cat data/$FNAME | tr -c "[:alnum:]" " " | tr "A-Z" "a-z" | tr " " "\n" | sort | uniq -c | sort -n > output/token_counts_${FNAME}
