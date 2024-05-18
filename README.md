# LeMA (Leveraging Misleading Answer)
---
This is the PyTorch code of the works Focusing valid search space in Open-World Compositional Zero-Shot Learning by leveraging misleading answers.
The code provides the implementation of our proposed training method, LeMA, with other baselines (e.g. SymNet, AoP, TMN, LabelEmbed+,RedWine).



## Setup 

1. Clone the repo 

2. We recommend using Anaconda for environment setup. To create the environment and activate it, please run:
```
conda env create --file anno_environment.yml
conda activate czsl
```

4. Go to the cloned repo and open a terminal. Download the datasets and embeddings, specifying the desired path (e.g. `DATA_ROOT` in the example):
```
bash ./utils/download_data.sh DATA_ROOT
mkdir logs
```
The word2vec s3 url has stopped working. You can download the data from [Kaggle](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300) or use [this Google Drive link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) (be careful downloading files from Google Drive).
```
cd DATA_ROOT/w2v
mv download_location/GoogleNews-vectors-negative300.bin.gz .
gzip -d GoogleNews-vectors-negative300.bin.gz
rm GoogleNews-vectors-negative300.bin.gz

```


## Train

### If you want to apply LeMA during training, enter the "--lema" argument. We provide the LeMA training option for three models (e.g., Co-CGE, CompCos, KG-SP).

### train CGQA
```
python train.py --config configs/anno/cgqa.yml
```

### train MIT
```
python train.py --config configs/anno/mit.yml
```

### train UT-Zappos50k
```
python train.py --config configs/anno/utzappos.yml
```

