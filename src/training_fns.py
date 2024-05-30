import pickle
import logging
import os
import warnings
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset_prep import DS_INFO, AdversaryDataset, VictimFineTuningDataset
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from src.utils import file_is_unchanged
logger = logging.getLogger(__name__)

def setup_environment(args): 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    warnings.filterwarnings("ignore", message="Passing `max_length` to BeamSearchScorer is deprecated")  # we ignore the warning because it works anyway for diverse beam search 
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    pl.seed_everything(seed=args.seed)


def setup_loggers(args, project): 
    wandb_logger = WandbLogger(project=project, entity=args.wandb_entity, save_dir=args.default_root_dir, mode=args.wandb_mode)
    path_run = f"{args.default_root_dir}/{wandb_logger.experiment.name}"
    if not os.path.exists(path_run): os.makedirs(path_run, exist_ok=True)
    log_filename = path_run + "/log.txt"
    handlers = [logging.FileHandler(log_filename)]
    if args.log_to_stdout: handlers += [logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, level=logging.INFO)
    print(f"Files and models are logged in folder {path_run}")
    print(f"Log file: {log_filename}")
    return wandb_logger, path_run

def load_dataset(args, project, **kwargs): 
    if args.run_mode == "dev" and file_is_unchanged("src/dataset_prep.py") and file_is_unchanged("src/parsing_fns.py"): 
        with open('cache/dataset_cached.pickle', 'rb') as handle: dataset = pickle.load(handle)
    else: 
        if   project == "whitebox_finetune_for_classification": DatasetClass = VictimFineTuningDataset
        elif project == "whitebox_adversary":                   DatasetClass = AdversaryDataset
        dataset = DatasetClass(args, **kwargs)
        if args.run_mode == "dev": 
            with open('cache/dataset_cached.pickle', 'wb') as handle: pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def get_callbacks(args, metric, mode): 
    callbacks = [
        ModelCheckpoint(monitor=metric, save_top_k=1, mode=mode, every_n_epochs=args.check_val_every_n_epoch, verbose=True)
    ]
    if args.early_stopping: callbacks.append(EarlyStopping(monitor=metric, mode=mode, patience=args.patience))
    return callbacks

def get_profiler(args): 
    if   args.profiler == "simple": 
        profiler = SimpleProfiler(dirpath=".", filename="perf_logs_simple")
    elif args.profiler == "advanced": 
        profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_adv")
    elif args.profiler == "pytorch": 
        profiler = PyTorchProfiler(dirpath=".", filename="perf_logs_pt", export_to_chrome=False)
    elif args.profiler is None: 
        profiler = None
    return profiler


def log_best_epoch_to_wandb(trainer, wandb_logger): 
    best_epoch = trainer.checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0]
    trainer.best_epoch = best_epoch
    wandb_logger.experiment.summary[f"best_epoch"] = best_epoch
    