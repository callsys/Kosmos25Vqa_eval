
ckpt="/mnt/msranlp/yuzhongzhao/ckpts/downstream803_1257/checkpoint_1_600.pt"
save="debug/600iter_804"

mkdir $save

datasets="DeepForm ChartQA InfographicsVQA WikiTableQuestions DocVQA KleisterCharity VisualMRC TextVQA TextCaps TabFact"

for dataset in $datasets
do
  echo "start to evaluate "${dataset}
  python evaluate_benchmarks2.py \
    --model_path $ckpt \
    --dataset $dataset \
    --downstream_dir /home/yuzhongzhao/zyz/data/docowl_data/DocDownstream-1.0/ \
    --save_dir ${save}/${dataset}
done
