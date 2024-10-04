cat data/bible.txt | tr -c "[:alnum:]" " " | tr "A-Z" "a-z" | tr " " "\n" | sort | uniq -c | sort -n > token_counts.txt
