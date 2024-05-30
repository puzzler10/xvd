This is the code for the paper [XVD: Cross-Vocabulary Differentiable Training for Generative Adversarial Attacks](https://aclanthology.org/2024.lrec-main.1544/), with the abstract

> An adversarial attack to a text classifier consists of an input that induces the classifier into an incorrect class prediction, while retaining all the linguistic properties of correctly-classified examples. A popular class of adversarial attacks exploits the gradients of the victim classifier to train a dedicated generative model to produce effective adversarial examples. However, this training signal alone is not sufficient to ensure other desirable properties of the adversarial attacks, such as similarity to non-adversarial examples, linguistic fluency, grammaticality, and so forth. For this reason, in this paper we propose a novel training objective which leverages a set of pretrained language models to promote such properties in the adversarial generation. A core component of our approach is a set of vocabulary-mapping matrices which allow cascading the generative model to any victim or component model of choice, while retaining differentiability end-to-end. The proposed approach has been tested in an ample set of experiments covering six text classification datasets, two victim models, and four baselines. The results show that it has been able to produce effective adversarial attacks, outperforming the compared generative approaches in a majority of cases and proving highly competitive against established token-replacement approaches.

## Installation 

To run yourself, create a virtual environment (using whatever tool you prefer) and install the packages at `environment.yml` file, which contains a complete list of packages used in the project. You probably don't need all of these, so for a smaller environment, try starting from a basic python install with the following versions of some key packages:
* python=3.10.9
* pandas==1.5.2
* torch==1.13.1+cu116
* transformers==4.27.4
* tokenizers==0.13.2
* wandb==0.15.0
* sentence-transformers==2.2.2
* textattack==0.3.8  (for running the baselines)
* datasets==2.9.0
* evaluate==0.4.0
* pytorch-lightning==1.9.0


## Running 
There are three scripts in this repo. The `training.py` file runs the adversarial attack on a victim dataset. The `victim_finetune.py` file will finetune a victim model on a given dataset, so that you can then run the attack on it.  The `run_baselines.py` file will compute the baselines, using the OpenAttack package in this repo (which is a slightly modified fork of the original OpenAttack text adversarial attack package).


### Adversary training 
Activate your virtual environment and then run with `python training.py` with the command line flags as arguments. You'll have to pass a trained victim model. See `src/parsing_fns.py` for the parameters and what they do. For example

```
python training.py --dataset_name trec --vm_name model.ckpt --hparam_profile overall_2 --seed 1007 --max_length_orig 32 --batch_size 10 --batch_size_eval 10 --wandb_mode disabled --run_mode prod 
```

The code uses `wandb` for logging. Set `wandb_mode` to `disabled` to disable wandb and just run the code. To use wandb yourself, you can pass your entity name and project name via command line parameters (see the code for details). 

There are three running "modes": dev, test, and prod. These just refer to different combinations of hyperparameters. Dev is used for quick CPU runs on a few datapoints to see if the code is working and to fix where it isn't. Test does a full run on a small dataset, and prod is for the full runs on the complete datasets. You can use the `run_mode` parameter to set this. 



### Finetuning classifiers / creating a victim model
To finetune a classifier on a dataset to then attack, you can run, for example

```
 python victim_finetune.py --model_name_or_path google/electra-small-discriminator --dataset_name rotten_tomatoes --run_mode test 
 ```
See the code for the parameters. This script allows for `run_mode` of `dev` and `test`.


### Baselines
To run baselines, specify a victim model, a dataset, and an attack algorithm. For example

```
python run_baselines.py --vm_name model.ckpt --dataset_name trec --run_mode test --attack_name textfooler --wandb_mode disabled --accelerator gpu 
 ```

See the code for the parameters. This script allows for `run_mode` of `dev` and `test`. This can also be logged to wandb if you choose - follow the instructions above for the adversary finetuning. 
