
import warnings
import pytorch_lightning as pl
from argparse import ArgumentParser
from distutils.util import strtobool
import multiprocessing as mp
from src.adversary import WhiteboxAdversary
from src.victim_model import VictimModel
from src.utils import get_least_occupied_gpu

def add_common_args(parser): 
    common_args = parser.add_argument_group('Common arguments for diff projects')
    ## Data
    common_args.add_argument("--dataset_name", type=str, required=True, 
        choices=['simple', 'rotten_tomatoes', 'financial_phrasebank', 'trec', 'subj', 'emotion', 'hate_speech'], 
        help="The name of the dataset to use.")
    common_args.add_argument('--max_length_orig', type=int, 
        help="Set to a value to remove any examples with more tokens that that from the dataset.")
    common_args.add_argument('--min_length_orig', type=int,
        help="Minimum number of tokens in original.")
    common_args.add_argument('--shuffle_train', type=lambda x: bool(strtobool(x)),
        help="Shuffle the training set during training. Cannot be used with bucket_by_length=True during adversary training.")
    common_args.add_argument('--disable_hf_caching', type=lambda x: bool(strtobool(x)), 
        help="If True, will not use the cache to reload the dataset.")
    common_args.add_argument('--n_shards', type=int,
        help="If above 0, will shard the dataset into that many shards. Used to get a small dataset for quick testing.")
    ## Optimisation and training
    common_args.add_argument('--seed',          type=int,  help='Random seed')
    common_args.add_argument('--batch_size',      type=int, help='Batch size for training')
    common_args.add_argument('--batch_size_eval', type=int, help='Batch size for evaluation')
    common_args.add_argument('--learning_rate', type=float, help='Learning rate.')
    common_args.add_argument('--weight_decay',  type=float, help='Weight decay for AdamW')
    common_args.add_argument('--num_workers', type=int, help='Number of parallel worker threads for data loaders')
    common_args.add_argument('--early_stopping', type=lambda x: bool(strtobool(x)), 
                            help='If to do early stopping or not.')
    common_args.add_argument('--patience', type=int, help='Patience for early stopping.')
    common_args.add_argument("--optimizer_type", type=str, choices=['AdaFactor', 'AdamW'], help="Which optimiser to use.")
    ## Misc
    common_args.add_argument("--run_mode", type=str, required=True, choices=['dev', 'test', 'prod'],
        help="Run mode. Dev is for development, test is for test runs on wandb, prod is for the actual experiments.")
    common_args.add_argument('--wandb_mode', type=str, choices=['online', 'disabled'], 
         help='Set to "disabled" to suppress wandb logging.')
    common_args.add_argument('--wandb_project', type=str,help='wandb project name if using wandb')    
    common_args.add_argument('--wandb_entity',  type=str,help='wandb entity name if using wandb')    
    common_args.add_argument('--log_to_stdout', type=lambda x: bool(strtobool(x)),
         help='Set to True to log to stdout as well as the log file.')
    return parser

def get_args_adversary():
    parser = ArgumentParser(description="Fine-tune a model on a text classification dataset.")    
    parser = pl.Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--long_example_behaviour', type=str,  
        choices=['remove', 'truncate'],
        help="Set to 'remove' to remove examples longer than `max_length_orig`, or 'truncate' to truncate them.")
    data_args.add_argument("--bucket_by_length", type=lambda x: bool(strtobool(x)),
        help="Set to True to load the data from longest-to-smallest (good for memory efficiency)")
    data_args.add_argument("--shuffle_buckets", type=lambda x: bool(strtobool(x)),
        help="Set to True to shuffle the buckets when bucket_by_length=True")
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--pp_name',  type=str,                help="Name or path of paraphrase model to use. Should be an identifier from huggingface.co/models")
    model_args.add_argument('--vm_name',  type=str, required=True, help="Name or path of victim model to use. Can be locally saved. ")
    model_args.add_argument('--sts_name', type=str,                help="Name or path of STS model to use. ")
    model_args.add_argument('--nli_name', type=str,                help="Name or path of NLI model to use. ")
    model_args.add_argument('--download_models', type=lambda x: bool(strtobool(x)), 
        help="If True, downloads the model from the HuggingFace model hub. Else, uses the cache.")
    model_args.add_argument('--freeze_vm_model', type=lambda x: bool(strtobool(x)), 
        help="If True, freezes the victim model so it doesn't update during training. Else, will also finetune parameters of the vm model.")
    misc_args = parser.add_argument_group('Logging and other misc arguments')
    misc_args.add_argument('--run_untrained', type=lambda x: bool(strtobool(x)),
         help='Set to True to evaluate val + test sets before we train to measure baseline untrained performance.')
    misc_args.add_argument('--delete_final_model', type=lambda x: bool(strtobool(x)),
         help='Set to True to delete the model after training (useful when running a sweep)')
    misc_args.add_argument('--less_logging', type=lambda x: bool(strtobool(x)),
         help='Set to True to log less (and train faster))')
    misc_args.add_argument("--hparam_profile", type=str, help="Preset hparam choices")

    parser = WhiteboxAdversary.add_model_specific_args(parser) 
    args = parser.parse_args()

    # Set parameters as required 
    args = parse_vm_model_name(args)
    args = setup_defaults_adversary(args)
    if   args.run_mode == 'dev':  args = setup_dev_mode_adversary(args)
    elif args.run_mode == 'test': args = setup_test_mode_adversary(args)
    elif args.run_mode == 'prod': args = setup_prod_mode_adversary(args)
    else: raise Exception("shouldn't get here")
    args = setup_hparam_profiles_adversary(args)

    ##### Checks and postprocessing arguments
    # Verify that the vm_model and the dataset match
    if args.dataset_name == 'simple': 
        if 'rotten' not in args.vm_name: raise Exception("For now, when using simple ds, use a VM model finetuned on rotten_tomatoes dataset.")
    else: 
        for ds_name in ['rotten', 'financial']: 
            if (ds_name in     args.vm_name and ds_name not in args.dataset_name) or \
               (ds_name not in args.vm_name and ds_name in     args.dataset_name):
                raise Exception("Check that your VM model and dataset correspond. For now, both have to contain the name of the dataset in them.")
    # If needed, change check_val_every_n_epoch if its higher than max_epochs
    if args.max_epochs not in ["-1", -1] and args.max_epochs < args.check_val_every_n_epoch:
        warnings.warn(f"Max epochs of {args.max_epochs} is less than check_val_every_n_epoch value of {args.check_val_every_n_epoch} so will change check_val_every_n_epoch to {args.max_epochs}")
        args.check_val_every_n_epoch = args.max_epochs
    args = set_eval_gen_settings_from_eval_condition(args, args.eval_condition)
    return args

def get_args_victim_finetune(): 
    parser = ArgumentParser(description="Fine-tune a model on a text classification dataset.")    
    parser = pl.Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--model_name_or_path', type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser = VictimModel.add_model_specific_args(parser) 
    args = parser.parse_args()
    args = setup_defaults_victim_finetuning(args)
    if   args.run_mode == 'dev':  args = setup_dev_mode_victim_finetuning(args)
    elif args.run_mode == 'test': args = setup_test_mode_victim_finetuning(args)
    else: raise Exception("shouldn't get here")
    return args

def get_args_run_baselines(): 
    parser = ArgumentParser(description="Run baselines on datasets. ")   
    parser = pl.Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)
    run_args = parser.add_argument_group('Run related arguments')
    run_args.add_argument('--vm_name',      type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    run_args.add_argument('--attack_name',  type=str, required=True, help="What baseline attack to run.")
    args = parser.parse_args()
    args.freeze_vm_model = False 
    args.long_example_behaviour = "remove"
    args = parse_vm_model_name(args)
    if  args.run_mode == 'dev':  
        args.n_examples = 10
        args.max_length_orig = 10
    elif args.run_mode == 'test':
        args.n_examples = -1
        args.max_length_orig = 32
    return args

def parse_vm_model_name(args): 
    """This is just a helper function. Sometimes it gets a little annoying putting in long paths.
        Feel free to adjust it to your needs. """
    if "|" in args.vm_name: 
        vm_model_type = args.vm_name.split('|')[1] 
        args.vm_name = f"/home/tproth/Data/model_checkpoints/whitebox_victim_finetuning/final/{args.dataset_name}_{vm_model_type}_vm/model.ckpt"
    return args

def set_eval_gen_settings_from_eval_condition(args, eval_condition): 
    assert eval_condition in ['standard', 'dev', 'ablation']
    gen_settings = {
        "val": {  # used in validation
            "num_return_sequences": 8,
            "num_beams": 16,
            "num_beam_groups": 8,
        }, 
        "bs_1": {
            "num_return_sequences": 1,
            "num_beams":16
        }, 
        "dbs_2": {
            "num_return_sequences": 2,
            "num_beams": 16,
            "num_beam_groups": 2,
        }, 
        "dbs_4": {
            "num_return_sequences": 4,
            "num_beams": 16,
            "num_beam_groups": 2,
        },
        "dbs_8": {
            "num_return_sequences": 8,
            "num_beams": 16,
            "num_beam_groups": 4,
        }, 
        "dbs_16": {
            "num_return_sequences": 16,
            "num_beams": 16,
            "num_beam_groups": 8,
        },
        "dbs_32": {
            "num_return_sequences": 32,
            "num_beams": 32,
            "num_beam_groups": 16,
        },
        "dbs_48": {
            "num_return_sequences": 48,
            "num_beams": 48,
            "num_beam_groups": 24,
        },
        "dbs_64": {
            "num_return_sequences": 64,
            "num_beams": 64,
            "num_beam_groups": 32,
        },
    }

    for k in gen_settings.keys():
        gen_settings[k]["temperature"] = 1.
        gen_settings[k]["top_p"] = 0.98
        gen_settings[k]["do_sample"] = False
        gen_settings[k]["return_dict_in_generate"] = True
        gen_settings[k]["output_scores"] = True
        if k != 'bs_1':  gen_settings[k]["diversity_penalty"] = 1.
    if   eval_condition == "standard": args.gen_settings = {'val': gen_settings['val'], 'dbs_32': gen_settings['dbs_32']}
    elif eval_condition == "dev":      args.gen_settings = {'val': gen_settings['val'], 'dbs_4':  gen_settings['dbs_4']}
    elif eval_condition == "ablation": args.gen_settings = gen_settings
    return args

def setup_hparam_profiles_adversary(args): 
    hparam_profiles = {
        "overall_1": {
            "gumbel_tau": 1.05, 
            "coef_diversity": 0.05, 
            "coef_kl": 0.20, 
            "coef_sts": 9.70, 
            "coef_vm": 28.0
        },
        "overall_2": {
            "gumbel_tau": 1.05, 
            "coef_diversity": 3.65, 
            "coef_kl": 0.15, 
            "coef_sts": 3.45, 
            "coef_vm": 20.0
        },
        "alternate_1": {
            "gumbel_tau": 1.25, 
            "coef_diversity": 10, 
            "coef_kl": 0.10, 
            "coef_sts": 3.50, 
            "coef_vm": 20.0
        },
        "alternate_2": {
            "gumbel_tau": 0.90, 
            "coef_diversity": 10, 
            "coef_kl": 0.15, 
            "coef_sts": 3.45, 
            "coef_vm": 20.0
        },
        "final_3": {
            "gumbel_tau": 1.25, 
            "coef_diversity": 10, 
            "coef_kl": 0.10, 
            "coef_sts": 3.50, 
            "coef_vm": 20.0
        },
        "final_3": {
            "gumbel_tau": 1.25, 
            "coef_diversity": 10, 
            "coef_kl": 0.10, 
            "coef_sts": 3.50, 
            "coef_vm": 20.0
        },
        "high_ASR_1" :{
            "gumbel_tau": 0.91, 
            "coef_diversity": 2.25, 
            "coef_kl": 0.11, 
            "coef_sts": 2.80, 
            "coef_vm": 25.8
        }, 
        "high_ASR_2" :{
            "gumbel_tau": 0.92, 
            "coef_diversity": 1.86, 
            "coef_kl": 0.22, 
            "coef_sts": 4.85, 
            "coef_vm": 29.6
        }
    }
    if args.hparam_profile is not None: 
        args.gumbel_tau     = hparam_profiles[args.hparam_profile]["gumbel_tau"]
        args.coef_diversity = hparam_profiles[args.hparam_profile]["coef_diversity"]
        args.coef_kl        = hparam_profiles[args.hparam_profile]["coef_kl"]
        args.coef_sts       = hparam_profiles[args.hparam_profile]["coef_sts"]
        args.coef_vm        = hparam_profiles[args.hparam_profile]["coef_vm"]
    return args

def setup_defaults_adversary(args): 
    # Models
    args.pp_name = "prithivida/parrot_paraphraser_on_T5"
    args.sts_name = "sentence-transformers/paraphrase-MiniLM-L12-v2"
    args.nli_name = "howey/electra-small-mnli"
    args.download_models = False
    args.freeze_vm_model = True
    # Hardware
    args.accelerator = "gpu"
    args.devices = [get_least_occupied_gpu()]
    args.num_workers= min(16, mp.cpu_count()-1)
    # Data
    args.min_length_orig = 0
    args.long_example_behaviour = "remove"
    args.bucket_by_length = True
    args.shuffle_train = False
    args.disable_hf_caching = False
    args.delete_final_model = True 
    args.shuffle_buckets = True

    # Training
    args.optimizer_type = "AdaFactor"
    args.learning_rate = 1e-5
    args.weight_decay = None
    args.less_logging = True

    # good defaults 
    args.coef_nli = 0.5
    args.eval_nli_threshold = 0.4
    args.eval_sts_threshold = 0.3
    args.eval_kl_threshold = 4.5
    args.num_gumbel_samples = 5 

    # Misc
    args.default_root_dir = ""
    args.num_sanity_val_steps = 0
    args.fast_dev_run = False
    args.log_to_stdout = False
    args.profiler = None

    return args

def setup_dev_mode_adversary(args): 
    args.run_untrained = False

    # Datasets and models
    args.n_shards = 2

    # Paraphrase and orig parameters
    args.max_length_orig = 16
    
    args.eval_condition = "standard"

    # Batches and epochs
    args.batch_size = 2
    args.batch_size_eval = 2
    args.num_gumbel_samples = 3
    args.overfit_batches = 2
    args.max_epochs = 2
    args.early_stopping = True
    args.patience = 2 # any value will do 

    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 1

    # # For debugging
    args.accelerator = "cpu"
    args.devices = 1 
    args.num_workers = 0 # min(8, mp.cpu_count()-1)
    return args

def setup_test_mode_adversary(args): 
    args.run_untrained = True
    args.n_shards = -1
    args.overfit_batches = 8

    # Paraphrase and orig parameters
    args.eval_condition = "standard" 

    # Batches and epochs
    args.max_epochs = 15
    args.early_stopping = True
    args.patience = 8
    args.log_every_n_steps = 20
    args.check_val_every_n_epoch = 1
    return args    

def setup_prod_mode_adversary(args): 
    args.run_untrained = False
    args.n_shards = -1

    # Paraphrase and orig parameters
    args.eval_condition = "ablation" 
    args.val_check_interval = 24

    # Batches and epochs
    args.learning_rate = 4e-5
    args.max_epochs = 30
    args.early_stopping = True
    args.patience = 35
    args.limit_val_batches= 100
    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 1
    args.max_time={"days": 0, "hours": 12    }
    return args    

def setup_defaults_victim_finetuning(args): 
    # Models
    args.download_models = False

    # Hardware
    args.accelerator = "gpu"
    args.devices = 1 
    args.num_workers= min(8, mp.cpu_count()-1)
    # Data
    args.min_length_orig = 0
    args.shuffle_train = True
    args.disable_hf_caching = True
    args.bucket_by_length = False   # need the parameter set here for now to patch over some lazy code


    # Training
    args.seed = 9994
    args.optimizer_type = "AdamW"
    args.learning_rate = 0.0001
    args.weight_decay = 0.01

    # Misc
    args.default_root_dir = ""  # update this to wherever you like
    args.num_sanity_val_steps = 0
    args.fast_dev_run = False
    return args

def setup_dev_mode_victim_finetuning(args): 
    args.wandb_mode = 'disabled'

    # Datasets and models
    args.n_shards = 10 

    # Paraphrase and orig parameters
    args.max_length_orig = 16

    # Batches and epochs
    args.batch_size = 8
    args.batch_size_eval = 16
    args.overfit_batches = 2
    args.max_epochs = 3
    args.early_stopping = False
    args.patience = None

    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 1

    # Misc
    args.log_to_stdout = True

    # # For debugging
    args.accelerator = "cpu"
    args.devices = 1 
    args.num_workers = 0 # min(8, mp.cpu_count()-1)
    return args

def setup_test_mode_victim_finetuning(args): 
    args.wandb_mode = 'online'
    # Datasets and models
    args.n_shards = -1

    # Paraphrase and orig parameters
    args.max_length_orig = 128

    # Batches and epochs
    args.batch_size = 64
    args.batch_size_eval = 256

    # args.overfit_batches = 12
    args.max_epochs = -1
    args.num_sanity_val_steps = 0
    args.early_stopping = True
    args.patience = 5

    args.log_every_n_steps = 50
    args.check_val_every_n_epoch = 1

    # Misc
    args.log_to_stdout = False
    return args
