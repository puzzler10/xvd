from pprint import pprint
import OpenAttack
import ssl
from src.models import get_vm_tokenizer_and_model 
from src.dataset_prep import BaselineDataset
import src.parsing_fns 
import wandb
import os
from src.utils import is_t5

def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    vm_is_t5 = is_t5(args.vm_name)
    device = 'cuda' if args.accelerator == "gpu" else "cpu"
    # Init wandb
    if args.wandb_mode is None: args.wandb_mode = "online"
    wandb.init(project="whitebox-baselines", entity="uts_nlp", mode=args.wandb_mode)

    # add args to wandb config 
    wandb.config.update(args)

    # Needed because the certificate of thunlp is expired
    # https://github.com/thunlp/OpenAttack/issues/266
    ssl._create_default_https_context = ssl._create_unverified_context
    vm_tokenizer, vm_model   = get_vm_tokenizer_and_model(args)


    victim = OpenAttack.classifiers.TransformersClassifier(vm_model, vm_tokenizer, embedding_layer=vm_model.get_input_embeddings(), device=device)
    dataset = BaselineDataset(args, vm_tokenizer)
    wandb.config.update(dataset.dataset_info)

    mlm = "bert-base-uncased"
    if args.attack_name == 'scpn':
        attacker = OpenAttack.attackers.SCPNAttacker()
    elif args.attack_name == 'gan':
        attacker = OpenAttack.attackers.GANAttacker()
    elif args.attack_name == 'bertattack':
        attacker = OpenAttack.attackers.BERTAttacker(mlm_path=mlm, device=device)
    elif args.attack_name == 'bae':
        attacker = OpenAttack.attackers.BAEAttacker(mlm_path=mlm, k=60, batch_size=4, device=device)
    elif args.attack_name == 'textfooler':
        token_unk = vm_tokenizer.unk_token
        attacker = OpenAttack.attackers.TextFoolerAttacker(token_unk=token_unk)
    else: 
        raise Exception("invalid attack name")
    

    attack_eval = OpenAttack.AttackEval(attacker, victim, metrics = [
        OpenAttack.metric.BARTScore(),
        OpenAttack.metric.BERTScore(),
        OpenAttack.metric.EntailmentProbability("howey/electra-small-mnli"),
    ])

    result,results_df = attack_eval.eval(dataset.dsd, visualize=True, progress_bar=True)
    result['attack_name'] = args.attack_name
    result['dataset_name'] = args.dataset_name
    result['vm_name'] = vm_model.__class__.__name__
    wandb.log(result)

    import datetime
    timenow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("./results/"): os.makedirs("./results/")
    results_df.to_csv(f"results/{args.dataset_name}_{args.attack_name}_{result['vm_name']}_{timenow}.csv", index=False)


if __name__ == "__main__":
    args = src.parsing_fns.get_args_run_baselines()
    pprint(vars(args))
    main(args)