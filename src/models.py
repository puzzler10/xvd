
import copy
import torch
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer)
from sentence_transformers import SentenceTransformer
from types import MethodType
from undecorated import undecorated
import logging
from src.victim_model import VictimModel
from src.utils import is_t5
logger = logging.getLogger(__name__)
T5_PREFIX = {
    "paraphrase": "paraphrase: ", 
}
T5_POSTFIX = " </s>"

ALG_TOKENIZER_MAPPING = {
    'SentencePiece': ["T5", "Albert"], 
    'WordPiece': ["Bert", "DistilBert", 'Electra'], 
    "BPE": ['Roberta']
}
for k,v in ALG_TOKENIZER_MAPPING.items(): ALG_TOKENIZER_MAPPING[k] = [o + "TokenizerFast" for o in v]
TOKENIZER_ALG_MAPPING = dict()
for alg,tokenizer_l in ALG_TOKENIZER_MAPPING.items(): 
    for t in tokenizer_l: 
        TOKENIZER_ALG_MAPPING[t] = alg


##### LOAD MODELS #####
def get_pp_tokenizer_and_model(args, ref_model=False):
    """As well as preparing the pp model and tokenizer this function also adds a new method `generate_with_grad` to
    the pp model so that we can backprop when generating."""
    pp_config    = AutoConfig.from_pretrained(   args.pp_name)
    pp_tokenizer = AutoTokenizer.from_pretrained(args.pp_name, model_max_length=args.max_length_orig)
    if  "t5" in args.pp_name.lower():
        pp_model = AutoModelForSeq2SeqLM.from_pretrained(args.pp_name, local_files_only= not args.download_models, config=pp_config)
        # For t5 there is a problem with different sizes of embedding vs vocab size. 
        # See https://github.com/huggingface/transformers/issues/4875
        pp_model.resize_token_embeddings(len(pp_tokenizer))
    else:               
        pp_model = AutoModelForSeq2SeqLM.from_pretrained(args.pp_name, local_files_only= not args.download_models, config=pp_config, 
            max_position_embeddings = args.max_length_orig + 10
        )
    generate_with_grad = undecorated(pp_model.generate)      # removes the @no_grad decorator from generate so we can backprop
    pp_model.generate_with_grad = MethodType(generate_with_grad, pp_model)
    if ref_model: 
        for i, (name, param) in enumerate(pp_model.named_parameters()): param.requires_grad = False   # freeze 
        pp_model.eval()
    else:         
        pp_model.train()
    return pp_tokenizer, pp_model

def get_vm_tokenizer_and_model(args):
    if args.vm_name[-5:] == ".ckpt":
        vm_module = VictimModel.load_from_checkpoint(args.vm_name)
        vm_config    = vm_module.config
        vm_tokenizer = vm_module.tokenizer
        vm_model     = vm_module.model
    else: 
        vm_config    = AutoConfig.from_pretrained(   args.vm_name)
        vm_tokenizer = AutoTokenizer.from_pretrained(args.vm_name)
        vm_model = AutoModelForSequenceClassification.from_pretrained(args.vm_name, local_files_only=not args.download_models, config=vm_config)
    vm_model.eval()
    if 'T5' in vm_tokenizer.__class__.__name__: 
        old_lm_head = copy.deepcopy(vm_model.get_output_embeddings())
        vm_model.resize_token_embeddings(len(vm_tokenizer))  # this overwrites lm_head (which is bad because we have a classification layer)
        vm_model.set_output_embeddings(old_lm_head)

    if args.freeze_vm_model: 
         for i, (name, param) in enumerate(vm_model.named_parameters()): param.requires_grad = False
    return vm_tokenizer, vm_model

def get_nli_tokenizer_and_model(args):
    nli_config    = AutoConfig.from_pretrained(   args.nli_name)
    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_name)
    nli_model     = AutoModelForSequenceClassification.from_pretrained(args.nli_name, local_files_only=not args.download_models,  config=nli_config)
    nli_model.eval()
    for i, (name, param) in enumerate(nli_model.named_parameters()): param.requires_grad = False
    return nli_tokenizer, nli_model

def get_cola_tokenizer_and_model(args):
    cola_config    = AutoConfig.from_pretrained(   args.cola_name)
    cola_tokenizer = AutoTokenizer.from_pretrained(args.cola_name)
    cola_model = AutoModelForSequenceClassification.from_pretrained(args.cola_name, local_files_only=not args.download_models, config=cola_config)
    cola_model.eval()
    for i, (name, param) in enumerate(cola_model.named_parameters()): param.requires_grad = False
    return cola_tokenizer, cola_model

def get_sts_model(args):
    sts_model = SentenceTransformer(args.sts_name)
    for i, (name, param) in enumerate(sts_model.named_parameters()): param.requires_grad = False
    return sts_model

def get_all_models(args):
    """Load tokenizers and models for vm, pp, sts.
    Pad the first embedding layer if specified in the config.
    Update config with some model-specific variables.
    """
    pp_tokenizer, pp_model =  get_pp_tokenizer_and_model(args)
    vm_tokenizer, vm_model =  get_vm_tokenizer_and_model(args)
    nli_tokenizer, nli_model   = get_nli_tokenizer_and_model(args)
    cola_tokenizer, cola_model = get_cola_tokenizer_and_model(args)
    sts_model = get_sts_model(args)
    return  pp_tokenizer, pp_model, vm_tokenizer, vm_model, nli_tokenizer, nli_model, cola_tokenizer, cola_model, sts_model

### INFERENCE 
def get_vm_probs_from_text(text, vm_tokenizer, vm_model, return_logits=False):
    """Get victim model predictions for a batch of text."""
    if vm_model.training: vm_model.eval()
    with torch.no_grad():
        tkns = vm_tokenizer(text, padding=True, pad_to_multiple_of=8, return_tensors="pt").to(vm_model.device)
        input_ids,attention_mask = tkns['input_ids'],tkns['attention_mask']
        start_id = vm_tokenizer.pad_token_id if vm_tokenizer.bos_token_id is None else vm_tokenizer.bos_token_id
        if is_t5(vm_model.__class__.__name__):
            decoder_input_ids = torch.full(size=(input_ids.shape[0], 1), fill_value=start_id, device=vm_model.device)
            outputs = vm_model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        else: 
            outputs = vm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        probs = torch.softmax(logits,1)
        preds = torch.argmax(logits, 1) 
    if return_logits: return logits 
    else:             return probs, preds

def get_vm_scores_from_vm_logits(labels, orig_truelabel_probs, vm_logits): 
    """vm_logits -> vm_scores"""
    vm_probs = vm_logits.softmax(axis=1)
    vm_predclass = torch.argmax(vm_probs, axis=1)
    vm_truelabel_probs   = torch.gather(vm_probs, 1, labels[:,None]).squeeze()
    vm_scores = orig_truelabel_probs - vm_truelabel_probs
    return dict(vm_predclass=vm_predclass, vm_truelabel_probs=vm_truelabel_probs, vm_scores=vm_scores)    

def get_vm_scores_from_vm_logits_gumbel_sampled(labels, orig_truelabel_probs, vm_logits): 
    """vm_logits -> vm_scores for the case where vm_logits has gumbel samples"""
    gumbel_samples = vm_logits.shape[0]
    batch_size = vm_logits.shape[1]
    assert len(labels) == len(orig_truelabel_probs) == batch_size 
    vm_probs = vm_logits.softmax(axis=-1)
    vm_predclass = torch.argmax(vm_probs, axis=-1)
    assert vm_predclass.shape == (gumbel_samples, batch_size)
    labels_repeated = labels.repeat((gumbel_samples,1))
    assert labels_repeated.shape == (vm_probs.shape[0], vm_probs.shape[1]) == (gumbel_samples, batch_size)
    vm_truelabel_probs   = torch.gather(vm_probs, 2, labels_repeated[:,:,None]).squeeze()
    assert vm_truelabel_probs.shape == labels_repeated.shape
    orig_truelabel_probs_repeated = orig_truelabel_probs.repeat((gumbel_samples,1))
    assert orig_truelabel_probs_repeated.shape == vm_truelabel_probs.shape
    vm_scores = orig_truelabel_probs_repeated - vm_truelabel_probs
    return dict(vm_predclass=vm_predclass, vm_truelabel_probs=vm_truelabel_probs, vm_scores=vm_scores)    

def get_nli_probs(orig_l, pp_l, nli_tokenizer, nli_model):
    inputs = nli_tokenizer(orig_l, pp_l, return_tensors="pt", padding=True, truncation=True).to(nli_model.device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = logits.softmax(1)
    return probs

def get_cola_probs(text, cola_tokenizer, cola_model):
    inputs = cola_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = cola_model(**inputs).logits
        probs = logits.softmax(1)
    return probs

def get_logp(orig_ids, pp_ids, tokenizer, model):
    """model: one of pp_model, ref_model (same with tokenizer)"""
    start_id = tokenizer.pad_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    decoder_start_token_ids = torch.tensor([start_id], device=model.device).repeat(len(orig_ids), 1)
    pp_ids = torch.cat([decoder_start_token_ids, pp_ids], 1)
    logprobs = []
    for i in range(pp_ids.shape[1] - 1):
        decoder_input_ids = pp_ids[:, 0:(i+1)]
        outputs = model(input_ids=orig_ids, decoder_input_ids=decoder_input_ids)
        token_logprobs = outputs.logits[:,i,:].log_softmax(1)
        pp_next_token_ids = pp_ids[:,i+1].unsqueeze(-1)
        pp_next_token_logprobs = torch.gather(token_logprobs, 1, pp_next_token_ids).detach().squeeze(-1)
        logprobs.append(pp_next_token_logprobs)
    logprobs = torch.stack(logprobs, 1)
    logprobs = torch.nan_to_num(logprobs, nan=None, posinf=None, neginf=-20) 
    logprobs = logprobs.clip(min=-20)
    attention_mask = model._prepare_attention_mask_for_generation(pp_ids[:,1:], tokenizer.pad_token_id, tokenizer.eos_token_id)
    logprobs = logprobs * attention_mask
    logprobs_sum = logprobs.sum(1)
    logprobs_normalised = logprobs_sum / attention_mask.sum(1)  # normalise for length of generated sequence
    return logprobs_normalised