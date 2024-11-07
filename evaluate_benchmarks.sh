ckpt="/mnt/msranlp/yuzhongzhao/ckpts/downstream716/checkpoint_1_3900.pt"
#ckpt="/mnt/msranlp/yuzhongzhao/ckpts/downstream623/checkpoint_1_3300.pt"
#ckpt="/home/yuzhongzhao/zyz/ckpts/checkpoint_1_200.pt" # "/mnt/msranlp/yuzhongzhao/ckpts/downstream623/checkpoint_1_3300.pt"
save="debug/3900iter_716"

mkdir $save

datasets="DeepForm ChartQA InfographicsVQA KleisterCharity WikiTableQuestions DocVQA KleisterCharity TabFact VisualMRC TextVQA TextCaps"

for dataset in $datasets
do
  echo "start to evaluate "${dataset}
  python evaluate_benchmarks.py \
    --model_path $ckpt \
    --dataset $dataset \
    --downstream_dir /home/yuzhongzhao/zyz/data/docowl_data/DocDownstream-1.0/ \
    --save_dir ${save}/${dataset}
done
