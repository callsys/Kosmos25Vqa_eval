# Pretraining of LayoutXLM-Local1d

## Requirements
- Pytorch1.10
- Python 3.8
```shell script
pip install -r requirements.txt
pip install --editable ./
```

## Dataset
The dataset directory structure looks like the following, each JSON file is a page of a document. We put 100 pages in the "dataset" directory for reference.  
```
│── data
│   ├── index.txt
│   └── data
│      ├── en
|         └──15e6d67c-18eb-11ec-9178-000d3a9b3395
|            └──0.json
|         ...
|      ├── de
|         └──013f93f8-141d-11ec-8bce-000d3a9b3395
|            └──6.json
```
The code needs an index file to find these pages. Each line in the index contains a path and the page number (split by "+"). 
```
en/170efda0-0ddc-11ec-a04a-0022481e496b+1
en/17406232-1098-11ec-b8c4-0022481e496b+5
de/61cbdca2-140b-11ec-bbd4-0022481c859f+0
...
el/27eb64c2-109a-11ec-b755-000d3a9b3395+0
``` 



## Useage
```
python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=4 --node_rank=$RANK --master_addr=$MASTER_ADDR \
    --master_port=$$MASTER_PORT \
    --examples/run_pt.py \
    --data_dir ./dataset \
    --model_name_or_path bert-base-multilingual-uncased \   # xlm-roberta-base for cased version
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.01 \
    --logging_steps 100 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --save_steps 10000 \
    --max_steps 1000000 \
    --dataloader_num_workers 16 \
    --output_dir path/to/output \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --layout_embedding_v2_coordinate_size 128 \ # 171 for large
    --layout_embedding_v2_shape_size 128 \      # 170 for large
    --seed 43 \
    --fp16 \
```
