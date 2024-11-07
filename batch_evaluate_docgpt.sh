

#CUDA_VISIBLE_DEVICES=0 python batch_evaluate_docgpt.py --shard "1/8" &
#CUDA_VISIBLE_DEVICES=1 python batch_evaluate_docgpt.py --shard "2/8" &
#CUDA_VISIBLE_DEVICES=2 python batch_evaluate_docgpt.py --shard "3/8" &
#CUDA_VISIBLE_DEVICES=3 python batch_evaluate_docgpt.py --shard "4/8" &
#CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docgpt.py --shard "5/8" &
#CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docgpt.py --shard "6/8" &
#CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docgpt.py --shard "7/8" &
#CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docgpt.py --shard "8/8" &

#ckpt="/mnt/msranlp/yuzhongzhao/ckpts/vqa529/checkpoint_1_210500.pt"
#ckpt="/mnt/msranlp/yuzhongzhao/ckpts/vqa601/checkpoint_1_210500.pt"
#ckpt="/mnt/msranlp/yuzhongzhao/ckpts/vqa602/checkpoint_1_300.pt"
ckpt="/home/yuzhongzhao/zyz/ckpts/layoutlm/checkpoint_1_600.pt"
nsample=50000

CUDA_VISIBLE_DEVICES=4 python batch_evaluate_docgpt.py --shard "1/4" --ckpt $ckpt --num-sample $nsample &
CUDA_VISIBLE_DEVICES=5 python batch_evaluate_docgpt.py --shard "2/4" --ckpt $ckpt --num-sample $nsample &
CUDA_VISIBLE_DEVICES=6 python batch_evaluate_docgpt.py --shard "3/4" --ckpt $ckpt --num-sample $nsample &
CUDA_VISIBLE_DEVICES=7 python batch_evaluate_docgpt.py --shard "4/4" --ckpt $ckpt --num-sample $nsample

sleep 10

python gather_evaluate_docvqa.py --ckpt $ckpt --num-sample $nsample --shard 4
