from datasets import load_dataset
from torch.utils.data import Dataset
import torch, random
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from datasets import disable_caching
import logging
import gc 
import src.models 
from src.utils import *
import copy
logger = logging.getLogger(__name__)

DS_INFO = {
  'rotten_tomatoes': {
    'task': 'sentiment',
    'LABEL2ID': {'negative': 0, 'positive': 1},
    'ID2LABEL': {0: 'negative', 1: 'positive'},
    'text_field': 'text',
    'label_field': 'label',
    'num_labels': 2
  },
  'financial_phrasebank': {
    'task': 'sentiment',
    'LABEL2ID': {'negative': 0, 'neutral': 1, 'positive': 2},
    'ID2LABEL': {0: 'negative', 1: 'neutral', 2: 'positive'},
    'text_field': 'sentence',
    'label_field': 'label',
    'num_labels': 3
  },
  'trec': {
    'task': 'question-type_classification',
    'LABEL2ID': {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5},
    'ID2LABEL': {0: 'ABBR', 1: 'ENTY', 2: 'DESC', 3: 'HUM', 4: 'LOC', 5: 'NUM'},
    'text_field': 'text',
    'label_field': 'coarse_label',
    'num_labels': 6
  }, 
  'subj': {
    'task': 'subjectivity_status',
    'LABEL2ID': {'objective': 0, 'subjective': 1},
    'ID2LABEL': {0: 'objective', 1: 'subjective'},
    'text_field': 'text',
    'label_field': 'label',
    'num_labels': 2
  },
  'emotion': {
    'task': 'emotion_recognition',
    'LABEL2ID': {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5},
    'ID2LABEL': {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'},
    'text_field': 'text',
    'label_field': 'label',
    'num_labels': 6
  },
  'hate_speech': {
    'task': 'hate_speech',
    'LABEL2ID': {'hate speech': 0, 'offensive language': 1, 'neither': 2},
    'ID2LABEL': {0: 'hate speech', 1: 'offensive language', 2: 'neither'},
    'text_field': 'text',
    'label_field': 'label',
    'num_labels': 3
  }, 
  'simple': {
    'task': 'sentiment',
    'LABEL2ID': {'negative': 0, 'positive': 1},
    'ID2LABEL': {0: 'negative', 1: 'positive'},
    'text_field': 'text',
    'label_field': 'label',
    'num_labels': 2
  }
}

class BaseDataset(Dataset):  
    """Common functions for both AdversaryDataset and VictimDataset"""
    def load_data(self): 
        if self.ds_name == "rotten_tomatoes": 
            dsd = load_dataset("rotten_tomatoes")
        elif self.ds_name == "financial_phrasebank":
            dsd = load_dataset("financial_phrasebank", "sentences_50agree")
            dsd = self.get_train_valid_test_split(dsd)
        elif self.ds_name == "trec":
            dsd = load_dataset("trec")
            dsd = self.get_train_valid_split(dsd)
            dsd = dsd.rename_column('coarse_label', 'label') 
            dsd = dsd.remove_columns('fine_label')
        elif self.ds_name == "subj":
            dsd = load_dataset("SetFit/subj")
            dsd = self.get_train_valid_split(dsd)
            dsd = dsd.remove_columns('label_text')
        elif self.ds_name == "emotion":
            dsd = load_dataset("dair-ai/emotion")
        elif self.ds_name == "hate_speech":
            dsd = load_dataset("SetFit/hate_speech_offensive")
            dsd = self.get_train_valid_test_split(dsd)
            dsd = dsd.remove_columns('label_text')
        elif self.ds_name == "simple":
            dsd = DatasetDict()
            for s in ['train', 'validation', 'test']:
                dsd[s] = load_dataset('csv', data_files=f"./data/simple_dataset_{s}.csv", keep_in_memory=False)['train']
        return dsd
    
    def add_idx(self, batch, idx):
        """Add row numbers"""
        batch['idx'] = idx
        return batch
    
    def add_n_tokens(self, batch, field):
        """Add the number of tokens present in the tokenised text """
        batch['n_tokens'] = [len(o) for o in batch[field]]
        return batch

    def get_dataloaders_dict(self, dsd, collate_fn):
        """Prepare a dict of dataloaders for train, valid and test"""
        if self.args.bucket_by_length and self.args.shuffle_train:  raise Exception("Can only do one of bucket by length or shuffle")
        persistent_workers = True if self.args.num_workers > 0 else False
        d = dict()
        for split, ds in dsd.items():
            batch_size = self.args.batch_size if split == "train" else self.args.batch_size_eval
            drop_last = True if len(ds) % batch_size == 1 else False
            if self.args.shuffle_train:
                if split == "train":
                    d[split] =  DataLoader(ds, batch_size=batch_size,
                                           shuffle=True,
                                            collate_fn=collate_fn, drop_last=drop_last,
                                           num_workers=self.args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
                else:
                    d[split] =  DataLoader(ds, batch_size=batch_size,
                                           shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                           num_workers=self.args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
            if self.args.bucket_by_length:
                if self.args.shuffle_buckets: 
                    # Sort the dataset by token count,  group sorted indices into batches, shuffle, flatten, make new ds
                    sorted_indices = sorted(range(len(ds)), key=lambda i: ds[i]['n_tokens'])
                    batches = [sorted_indices[i:i+batch_size] for i in range(0, len(sorted_indices), batch_size)]
                    random.Random(x=self.args.seed).shuffle(batches)  # x = seed for random
                    shuffled_indices = [i for batch in batches for i in batch]
                    shuffled_ds = torch.utils.data.Subset(ds, shuffled_indices)
                    d[split] = DataLoader(shuffled_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                          num_workers=self.args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
                else: 
                    d[split] =  DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                       num_workers=self.args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
        return d

    def get_train_valid_test_split(self, dsd, train_size=0.8):
        dsd1 = dsd['train'].train_test_split(train_size=train_size, shuffle=True, seed=0)
        dsd2 = dsd1['test'].train_test_split(train_size=0.5, shuffle=True, seed=0)
        return DatasetDict({
            'train': dsd1['train'],
            'validation': dsd2['train'],
            'test': dsd2['test']
        })
    
    def get_train_valid_split(self, dsd, train_size=0.8): 
        dsd1 = dsd['train'].train_test_split(train_size=train_size, shuffle=True, seed=0)
        return DatasetDict({
            'train': dsd1['train'],
            'validation': dsd1['test'],
            'test': dsd['test']
        })
    
    def update_dataset_info(self, dsd, dld=None): 
        """Make a dict with dataset stats. Used later for wandb logging. Useful for debugging """
        d = dict()
        for ds_name in ['train', 'validation', 'test']:
            d[f"ds_{ds_name}_len"] = len(dsd[ds_name])
            for i in range(self.num_labels):
                d[f"ds_{ds_name}_class_{self.ID2LABEL[i]}"] =  len(dsd[ds_name].filter(lambda x: x['label'] == i))
                if dld is not None: d[f"ds_{ds_name}_num_batches"] = len(dld[ds_name])
        d['ds_num_labels']    = self.num_labels
        d['ds_text_field']    = self.text_field
        d['ds_label_field']   = self.label_field
        self.dataset_info = d



class AdversaryDataset(BaseDataset):
    """Class with methods for the whitebox adversary""" 
    def __init__(self, args, pp_tokenizer, vm_tokenizer, nli_tokenizer, vm_model, sts_model): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.pp_tokenizer = pp_tokenizer
        self.vm_tokenizer = vm_tokenizer
        self.nli_tokenizer = nli_tokenizer
        self.vm_model = vm_model
        self.sts_model = sts_model
        if     args.accelerator == "gpu": self.device = 'cuda' 
        elif   args.accelerator == "cpu": self.device = 'cpu'
        else: raise Exception('only "cpu" and "gpu" supported for --accelerator argument.') 
        self.vm_model   = self.vm_model.to(self.device) 
        self.sts_model  = self.sts_model.to(self.device) 
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        self.dsd,self.dld = self.prepare_data(dsd)
        del self.vm_model
        del self.sts_model
        torch.cuda.empty_cache()
        gc.collect()

    def add_n_letters(self, batch):
        """Add number of letters in the original text"""
        batch['n_letters'] = [len(o) for o in batch[self.text_field]]
        return batch

    def prep_input_for_t5_paraphraser(self, batch, task):
        """To paraphrase the t5 model needs a "paraphrase: " prefix. 
        See the appendix of the T5 paper for the prefixes. (https://arxiv.org/abs/1910.10683) """  
        if  task == 'paraphrase':  
            batch[f'{self.text_field}_with_prefix'] = [src.models.T5_PREFIX[task]  + sen for sen in batch[self.text_field]]
        else:  
            raise Exception("shouldn't get here")                                      
        return batch

    def add_sts_embeddings(self, batch):
        """Calculate and save sts embeddings of the original text"""
        batch['orig_sts_embeddings'] = self.sts_model.encode(batch[self.text_field], batch_size=64, convert_to_tensor=False)
        return batch

    def tokenize_fn(self, batch, tokenizer, use_prefix):
        """Tokenize a batch of orig text using a tokenizer."""
        text_field = f'{self.text_field}_with_prefix' if use_prefix else self.text_field
        if self.args.long_example_behaviour == 'remove':    
            return tokenizer(batch[text_field])  # we drop the long examples later
        elif self.args.long_example_behaviour == 'truncate':         
            return tokenizer(batch[text_field], truncation=True, max_length=self.args.max_length_orig)

    def collate_fn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  
            d[k] = [o[k] for o in x]
        ## Tokenize with the pp_tokenizer and nli_tokenizer seperately 
        # pp tokenizer
        d_pp = copy.deepcopy(d)
        for k in ['orig_ids', 'attention_mask']: 
            d_pp[k] = d_pp.pop(f'{k}_pp_tknzr'); 
            d_pp.pop(f'{k}_nli_tknzr')
        d_pp.pop('token_type_ids_nli_tknzr')
        d_pp['input_ids'] = d_pp['orig_ids']; d_pp.pop('orig_ids')
        # print("d_pp_modified", d_pp)
        batch_pp = self.pp_tokenizer.pad(d_pp, pad_to_multiple_of=1, return_tensors="pt")
        # print("batch_pp", batch_pp)
        for k in ['input_ids', 'attention_mask']: batch_pp[f'{k}_pp_tknzr'] = batch_pp.pop(k)
        batch_pp['orig_ids_pp_tknzr'] = batch_pp.pop('input_ids_pp_tknzr')

        batch_nli = {'orig_ids_nli_tknzr':d['orig_ids_nli_tknzr'], 'attention_mask_nli_tknzr': d['attention_mask_nli_tknzr'], 
                     'token_type_ids_nli_tknzr': d['token_type_ids_nli_tknzr']}
        # combine and return
        return_d = {**batch_pp, **batch_nli}
        return return_d
        
    def add_vm_orig_score(self, batch):
        """Add the vm score of the orig text"""
        labels = torch.tensor(batch['label'], device=self.vm_model.device)
        orig_probs,orig_preds = src.models.get_vm_probs_from_text(text=batch[self.text_field], vm_tokenizer=self.vm_tokenizer, vm_model=self.vm_model)
        batch['orig_truelabel_probs'] = torch.gather(orig_probs,1, labels[:,None]).squeeze().cpu().tolist()
        batch['orig_vm_predclass'] = orig_preds.cpu().tolist()
        return batch
    
    def prepare_data(self, dsd):
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True)
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        if self.ds_name != "simple":  # not enough rows to shard with simple dataset.
            if self.args.n_shards > 0: 
                for k,v in dsd.items():  dsd[k] = v.shard(self.args.n_shards, 0, contiguous=True)  # contiguous to stop possible randomness of sharding
        dsd = dsd.map(self.add_vm_orig_score, batched=True,  batch_size=256)
        # Remove misclassified examples
        dsd = dsd.filter(lambda x: x['orig_vm_predclass']== x['label'])
        dsd = dsd.map(self.add_sts_embeddings,  batched=True,  batch_size=256)  # add STS score
        dsd = dsd.map(self.prep_input_for_t5_paraphraser,  batched=True,  fn_kwargs={'task': 'paraphrase'})  # preprocess raw text so pp model can read
        dsd = dsd.map(self.tokenize_fn,        batched=True, fn_kwargs={'tokenizer': self.pp_tokenizer, 'use_prefix' : True})  # tokenize with pp_tokenizer
        dsd = dsd.rename_column("input_ids", "orig_ids_pp_tknzr")
        dsd = dsd.rename_column("attention_mask", "attention_mask_pp_tknzr")
        dsd = dsd.map(self.tokenize_fn,        batched=True, fn_kwargs={'tokenizer': self.nli_tokenizer, 'use_prefix' : False})  # tokenize with nli_tokenizer 
        dsd = dsd.rename_column("input_ids", "orig_ids_nli_tknzr")
        dsd = dsd.rename_column("attention_mask", "attention_mask_nli_tknzr")
        dsd = dsd.rename_column("token_type_ids", "token_type_ids_nli_tknzr")
        # add n_tokens & filter out examples that have more tokens than a threshold
        dsd = dsd.map(self.add_n_tokens,       batched=True, fn_kwargs={'field': "orig_ids_pp_tknzr"})  # add n_tokens
        if self.args.long_example_behaviour == 'remove':    dsd = dsd.filter(lambda x: x['n_tokens'] <= self.args.max_length_orig)
        #dsd = dsd.map(self._add_n_letters,            batched=True)  # add n_letters
        if self.args.bucket_by_length: dsd = dsd.sort("n_tokens", reverse=True)  # sort by n_tokens (high to low), useful for cuda memory caching and reducing number of padding tokens
        assert dsd.column_names['train'] == dsd.column_names['validation'] == dsd.column_names['test']
        dsd_numeric = dsd.remove_columns([self.text_field, f'{self.text_field}_with_prefix'])
        dld = self.get_dataloaders_dict(dsd_numeric, collate_fn=self.collate_fn)  # dict of data loaders that serve tokenized text
        self.update_dataset_info(dsd, dld)
        return dsd,dld

class VictimFineTuningDataset(BaseDataset): 
    """This class just contains methods to finetune a victim model on a given dataset"""
    def __init__(self, args, vm_tokenizer): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.vm_tokenizer = vm_tokenizer
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        self.dsd,self.dld = self.prepare_data(dsd)

    def tokenize_fn(self, batch, tokenizer):
        """Tokenize a batch of orig text using a tokenizer."""
        return tokenizer(batch[self.text_field], max_length=self.args.max_length_orig)

    def create_dataloaders(self, dsd): 
        dsd_numeric = dsd.remove_columns([self.text_field])
        dld = self.get_dataloaders_dict(dsd_numeric, collate_fn=self.collate_fn)  # dict of data loaders that serve tokenized text
        return dld

    def collate_fn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  
            d[k] = [o[k] for o in x]
        batch_pp = self.vm_tokenizer.pad(d, return_tensors="pt", padding=True)
        return batch_pp 
    
    def prepare_data(self, dsd): 
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True) 
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        if self.args.n_shards > 0: 
            for k,v in dsd.items():  dsd[k] = v.shard(self.args.n_shards, 0, contiguous=True)  # contiguous to stop possible randomness of sharding
        dsd = dsd.map(self.tokenize_fn,        batched=True, fn_kwargs={'tokenizer': self.vm_tokenizer})  # tokenize with pp_tokenizer
        assert dsd.column_names['train'] == dsd.column_names['validation'] == dsd.column_names['test']
        dld = self.create_dataloaders(dsd)  # dict of data loaders that serve tokenized text
        self.update_dataset_info(dsd, dld)
        return dsd,dld
    
class BaselineDataset(BaseDataset): 
    def __init__(self, args, vm_tokenizer): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.vm_tokenizer = vm_tokenizer
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        dsd = dsd['test']
        self.dsd = self.prepare_data(dsd)

    def tokenize_fn(self, batch, tokenizer):
        """Tokenize a batch of orig text using a tokenizer."""
        return tokenizer(batch[self.text_field], max_length=self.args.max_length_orig)

    def prepare_data(self, dsd): 
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True) 
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        # add n_tokens & filter out examples that have more tokens than a threshold
        dsd = dsd.map(self.tokenize_fn,   batched=True, fn_kwargs={'tokenizer': self.vm_tokenizer})  # tokenize with pp_tokenizer
        dsd = dsd.map(self.add_n_tokens,  batched=True, fn_kwargs={'field': "input_ids"})  # add n_tokens
        if self.args.long_example_behaviour == 'remove':    dsd = dsd.filter(lambda x: x['n_tokens'] <= self.args.max_length_orig)    
        # If doing dev, select only a subset 
        if self.args.n_examples > -1:  dsd = dsd.select(range(self.args.n_examples))
        # For OpenAttack, create x and y columns
        dsd = dsd.map(function=lambda x: {"x": x[self.text_field], "y":  x["label"]})
        self.update_dataset_info(dsd)
        return dsd
    
    def update_dataset_info(self, ds): 
        d = dict()
        d[f"ds_test_len"] = len(ds)
        for i in range(self.num_labels):
            d[f"ds_test_class_{self.ID2LABEL[i]}"] =  len(ds.filter(lambda x: x['label'] == i))
        d['ds_num_labels']    = self.num_labels
        d['ds_text_field']    = self.text_field
        d['ds_label_field']   = self.label_field
        self.dataset_info = d