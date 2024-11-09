python gpt-4o-docvqa.py  --shard "1/8" &
python gpt-4o-docvqa.py  --shard "2/8" &
python gpt-4o-docvqa.py  --shard "3/8" &
python gpt-4o-docvqa.py  --shard "4/8" &
python gpt-4o-docvqa.py  --shard "5/8" &
python gpt-4o-docvqa.py  --shard "6/8" &
python gpt-4o-docvqa.py  --shard "7/8" &
python gpt-4o-docvqa.py  --shard "8/8" &

python gather_evalute_docvqa.py