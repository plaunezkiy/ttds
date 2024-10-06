FNAME="abstracts.wiki.txt"
cat data/$FNAME | tr -c "[:alnum:]" " " | tr "A-Z" "a-z" | tr " " "\n" > output/tokens_${FNAME}
