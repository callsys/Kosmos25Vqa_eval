# LayoutXLM_local1d

## Results
| Model | Data | #Params | Speedup(ONNX,GPU) | Speedup(ONNX,CPU) | EN(FUNSD) | ZH | JA | ES | FR | IT | DE | PT | Avg |
|-----------|:------------|:------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| LayoutXLM (24x1024, ACL 2022) w/ image | 22M | 625M | 1.0x | 1.0x | 0.8068 | 0.9155 | 0.8216 | 0.8055 | 0.8384 | 0.8372 | 0.853 | 0.865 | 0.8429 |
| LayoutXLM (24x1024, Teacher) w/o image | 400M | 559M | 1.0x | 1.0x | 0.9175 | 0.9453 | 0.8416 | 0.8892 | 0.9131 | 0.9115 | 0.9188 | 0.9159 | 0.9066 |
| LayoutXLM (6x384, PT) | 400M | 107M | 4.6x | 5.6x | 0.8753 | 0.9038 | 0.8043 | 0.8344 | 0.8757 | 0.8893 | 0.8793 | 0.8623 | 0.8656 |
| LayoutXLM (4x384, DT) | - | 103M | 5.7x | 6.6x | 0.8812 | 0.9039 | 0.8006 | 0.84 | 0.8681 | 0.875 | 0.8794 | 0.8561 | 0.8630 |
| LayoutXLM (6x384, DT) | - | 107M | 4.6x | 5.6x | 0.8832 | 0.9126 | 0.8137 | 0.8605 | 0.889 | 0.8936 | 0.8939 | 0.8709 | 0.8772 |
| LayoutXLM (12x384, DT) | - | 117M | 2.9x | 3.9x | 0.9127 | 0.9283 | 0.828 | 0.8777 | 0.9051 | 0.9049 | 0.9024 | 0.8866 | 0.8932 |
| LayoutXLM (12x768, DT) | - | 277M | 2.4x | 2.5x | 0.9177 | 0.9336 | 0.8262 | 0.8756 | 0.9159 | 0.9071 | 0.9156 | 0.9096 | 0.9002 |
| LayoutXLM-uncased (12x768, PT) | 400M | 249M | - | - | 0.9167 | 0.9271 | 0.8259 | 0.8930 | 0.9110 | 0.9050 | 0.9038 | 0.9036 | 0.8983 |
| LayoutXLM-uncased (24x1024, PT) | 400M | 521M | - | - | 0.9224 | 0.9342 | 0.8218 | 0.8920 | 0.9269 | 0.9210 | 0.9231 | 0.9205 | 0.9077 |

## Pre-trained Model (Cased)
### layoutxlm-local1d (all languages)
<pre>
24 x 1024 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_large/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_large/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_large/sentencepiece.bpe.model">sentencepiece.bpe.model</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_large/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_large/tokenizer_config.json">tokenizer_config.json</a>
</pre>

<pre>
12 x 768 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12768/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12768/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12768/sentencepiece.bpe.model">sentencepiece.bpe.model</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12768/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12768/tokenizer_config.json">tokenizer_config.json</a>
</pre>

<pre>
12 x 384 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12384/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12384/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12384/sentencepiece.bpe.model">sentencepiece.bpe.model</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12384/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_12384/tokenizer_config.json">tokenizer_config.json</a>
</pre>

<pre>
6 x 384 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_6384/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_6384/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_6384/sentencepiece.bpe.model">sentencepiece.bpe.model</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_6384/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_6384/tokenizer_config.json">tokenizer_config.json</a>
</pre>

<pre>
4 x 384 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_4384/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_4384/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_4384/sentencepiece.bpe.model">sentencepiece.bpe.model</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_4384/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_local1d_4384/tokenizer_config.json">tokenizer_config.json</a>
</pre>

## Pre-trained Model (Uncased)
### layoutxlm-local1d (all languages)
<pre>
24 x 1024 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_large/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_large/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_large/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_large/tokenizer_config.json">tokenizer_config.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_large/vocab.txt">vocab.txt</a>
</pre>

<pre>
12 x 768 model/
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_12768/config.json">config.json</a>
├── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_12768/pytorch_model.bin">pytorch_model.bin</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_12768/special_tokens_map.json">special_tokens_map.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_12768/tokenizer_config.json">tokenizer_config.json</a>
└── <a href="https://conversationhub.blob.core.windows.net/tengchaolv/share/devOps/models/400m-Layoutxlm_uncased_local1d_12768/vocab.txt">vocab.txt</a>
</pre>


## NOTE
**Due to the anonymous access for all the containers has been disabled, please add the below SAS as suffix to the url when you download models.**

**SAS:** ?sv=2020-04-08&si=share&sr=c&sig=dwYa3bHdPgg3CE9mvd5ZMziK%2BCBvcCGx6t2y73kOzZY%3D

## Usage
## Dataset
FUNSD: [download](https://conversationhub.blob.core.windows.net/tengchaolv/datasets/ner_Funsd.tar.gz)

RVL-CDIP: [download](https://conversationhub.blob.core.windows.net/tengchaolv/datasets/rvlcdip.tar.gz)

NOTE: The FUNSD dataset has been processed and the word bbox is replaced by its corresponding line bbox in the dataset. So if you want to use this model on other datasets, please check that the model use line bbox, not word bbox. 

## Fine-tuning on FUNSD
```
python -m torch.distributed.launch --nproc_per_node=8 run_funsd.py
        --config_json config/config_funsd.json
```

## Fine-tuning on RVL-CDIP
**NOTE**: This training only supports the cased model. If you need to use it on an uncased model, refer to the code for modifications.
```
python -m torch.distributed.launch --nproc_per_node=8 run_rvlcdip.py
        --config_json config/config_rvlcdip.json
```

