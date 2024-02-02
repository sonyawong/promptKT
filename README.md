# promptKT

## Installation

```
cd promptKT
pip install -r requirements.txt
```

## Datasets Preprocess

### Preprocess
```
cd examples
python data_preprocess.py --dataset_name=algebra2005
python data_preprocess.py --dataset_name=bridge2algebra2006
python data_preprocess.py --dataset_name=nips_task34
python data_preprocess.py --dataset_name=peiyou
python data_preprocess.py --dataset_name=assist2009
python data_preprocess.py --dataset_name=ednet5w
python data_preprocess.py --dataset_name=ednet
```

## Training

### pretrain stage
```
cd examples

GPUS_PER_NODE=8 # Select the number of Gpus you have


ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

echo $ROOT_DIR
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    wandb_gpt4kt_train.py

```
### prompt-tuning stage
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    wandb_gpt4kt_train.py \
    --emb_type qid_frozen2 \
    --pretrain_path {model_path} \
    --pretrain_epoch {best_model_epoch} \
    --train_mode ft \
    --dataset_name {assist2009,...,ednet5w}

## Evaluation
```
cd examples

python 
    wandb_predict.py \
    --dataset_name {assist2009,...,ednet5w} \
    --save_dir {model_path}
``` 