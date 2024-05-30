from pprint import pprint
import os
import pytorch_lightning as pl
from src.parsing_fns import get_args_adversary
from src.adversary import WhiteboxAdversary
from src.utils import *
from pytorch_lightning.utilities import rank_zero_only

def main(args): 
    from src.training_fns import setup_loggers,setup_environment,load_dataset,get_callbacks, get_profiler,log_best_epoch_to_wandb 
    setup_environment(args)
    project = "whitebox_adversary"
    project = args.wandb_project  if args.wandb_project is not None else "whitebox_adversary"
    wandb_logger, path_run = setup_loggers(args, project=project)
    adversary = WhiteboxAdversary(args) 
    adversary.path_run = path_run
    dataset = load_dataset(args, project, pp_tokenizer=adversary.pp_tokenizer, vm_tokenizer=adversary.vm_tokenizer, nli_tokenizer=adversary.nli_tokenizer,
                        vm_model=adversary.vm_model, sts_model=adversary.sts_model)
    try: 
        del dataset.vm_model
        del dataset.sts_model
    except AttributeError: 
        pass

    # Handle multi-gpu training
    if rank_zero_only.rank == 0:
        wandb_logger.experiment.config.update(dataset.dataset_info)
        wandb_logger.experiment.config.update(adversary.adversary_info)
    
    # Save our processed datasets to csv to join with the training data later 
    for k, ds in dataset.dsd.items(): 
        ds.remove_columns(['orig_ids_pp_tknzr', 'attention_mask_pp_tknzr', 'orig_ids_nli_tknzr', 'attention_mask_nli_tknzr', 'token_type_ids_nli_tknzr']).to_csv(f"{path_run}/orig_{k}.csv",index=False)
    profiler = get_profiler(args)
    callbacks = get_callbacks(args, metric="any_adv_example_proportion_validation", mode="max")

    trainer = pl.Trainer.from_argparse_args(args,
        logger=wandb_logger, 
        default_root_dir=path_run, 
        profiler=profiler,
        callbacks=callbacks
    )
    train_dataloaders,val_dataloaders,test_dataloaders=dataset.dld['train'],dataset.dld['validation'],dataset.dld['test']
    assert len(train_dataloaders) > 0 and len(val_dataloaders) > 0 and len(test_dataloaders) > 0, "A dataloader is empty"
    if args.run_untrained:
        adversary.untrained_run = True
        test_results = trainer.test(model=adversary,  dataloaders=test_dataloaders,  verbose=True)
        if rank_zero_only.rank == 0:
            if len(test_results) > 1:
                wandb_logger.experiment.config.update({f"untrained_{k}":v for k,v in test_results[0].items()})
            else: 
                print("Test results didn't have values")

    adversary.untrained_run = False
    if not adversary.matched_vocab_sizes: adversary.df_token_mapping['vm'].to_csv(f'{path_run}/df_token_mapping_vm.csv')
    trainer.fit( model=adversary, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders) 
    adversary.best_epoch  = trainer.checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0]
    wandb_logger.experiment.summary[f"best_epoch"] = adversary.best_epoch
    trainer.test(model=adversary,  dataloaders=test_dataloaders, ckpt_path="best", verbose=True)
    if args.delete_final_model: os.remove(trainer.checkpoint_callback.best_model_path)

        
if __name__ == "__main__":
    args = get_args_adversary()
    pprint(vars(args))
    main(args)