

CUDA_VISIBLE_DEVICES=0 python batch_evaluate_docvqa.py --shard "1/32" &
CUDA_VISIBLE_DEVICES=1 python batch_evaluate_docvqa.py --shard "2/32" &
CUDA_VISIBLE_DEVICES=2 python batch_evaluate_docvqa.py --shard "3/32" &
CUDA_VISIBLE_DEVICES=3 python batch_evaluate_docvqa.py --shard "4/32" &
CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docvqa.py --shard "5/32" &
CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docvqa.py --shard "6/32" &
CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docvqa.py --shard "7/32" &
CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docvqa.py --shard "8/32" &
CUDA_VISIBLE_DEVICES=0 python batch_evaluate_docvqa.py --shard "9/32" &
CUDA_VISIBLE_DEVICES=1 python batch_evaluate_docvqa.py --shard "10/32" &
CUDA_VISIBLE_DEVICES=2 python batch_evaluate_docvqa.py --shard "11/32" &
CUDA_VISIBLE_DEVICES=3 python batch_evaluate_docvqa.py --shard "12/32" &
CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docvqa.py --shard "13/32" &
CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docvqa.py --shard "14/32" &
CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docvqa.py --shard "15/32" &
CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docvqa.py --shard "16/32" &
CUDA_VISIBLE_DEVICES=0 python batch_evaluate_docvqa.py --shard "17/32" &
CUDA_VISIBLE_DEVICES=1 python batch_evaluate_docvqa.py --shard "18/32" &
CUDA_VISIBLE_DEVICES=2 python batch_evaluate_docvqa.py --shard "19/32" &
CUDA_VISIBLE_DEVICES=3 python batch_evaluate_docvqa.py --shard "20/32" &
CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docvqa.py --shard "21/32" &
CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docvqa.py --shard "22/32" &
CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docvqa.py --shard "23/32" &
CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docvqa.py --shard "24/32" &
CUDA_VISIBLE_DEVICES=0 python batch_evaluate_docvqa.py --shard "25/32" &
CUDA_VISIBLE_DEVICES=1 python batch_evaluate_docvqa.py --shard "26/32" &
CUDA_VISIBLE_DEVICES=2 python batch_evaluate_docvqa.py --shard "27/32" &
CUDA_VISIBLE_DEVICES=3 python batch_evaluate_docvqa.py --shard "28/32" &
CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docvqa.py --shard "29/32" &
CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docvqa.py --shard "30/32" &
CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docvqa.py --shard "31/32" &
CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docvqa.py --shard "32/32" &

#python gather_evalute_docvqa.py