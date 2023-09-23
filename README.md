## pytonh version
```
default=3.8
```
 
## Environment
```shell
pip install -r requirements.txt
```

## Download model and tokenizer
```shell
bash download.sh
```
## preprocess+train
```shell
python train.py --train_file "${1}"  --eval_file "${2}" --context_file "${3}"
```
## predict
```shell
python3.8 predict.py --context_file "${1}"  --test_file "${2}" --pred_file "${3}"
```