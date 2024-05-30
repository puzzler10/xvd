import gc
import itertools
from distutils.util import strtobool
import logging
from typing import List
import pytorch_lightning as pl
import torch
import numpy as np
import OpenAttack
from torch.nn.functional import gumbel_softmax
from transformers.utils.versions import require_version
from transformers import GenerationConfig 
from src.models import  get_nli_probs, get_nli_tokenizer_and_model, get_pp_tokenizer_and_model, get_sts_model, get_vm_scores_from_vm_logits_gumbel_sampled, get_vm_tokenizer_and_model
from src.models import  TOKENIZER_ALG_MAPPING, get_vm_probs_from_text, get_vm_scores_from_vm_logits
from src.models import get_logp
from sentence_transformers.util import pytorch_cos_sim
from src.dataset_prep import DS_INFO 
from src.utils import * 
from src.tests import * 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from transformers.optimization import Adafactor, AdamW

class WhiteboxAdversary(pl.LightningModule): 
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # Save arguments to hparams attribute.
        self.args = args
        self.adversary_info = dict()
        self.orig_cols_to_ignore = ['attention_mask_pp_tknzr', 'orig_ids_pp_tknzr', 'attention_mask_nli_tknzr', 'orig_ids_nli_tknzr', 'token_type_ids_nli_tknzr', 'orig_sts_embeddings']
        self.dataset_num_labels = DS_INFO[args.dataset_name]['num_labels']

        # Get models 
        self.pp_tokenizer,   self.pp_model   = get_pp_tokenizer_and_model(args)
        _,                   self.ref_model  = get_pp_tokenizer_and_model(args, ref_model=True)
        self.vm_tokenizer,   self.vm_model   = get_vm_tokenizer_and_model(args)
        self.sts_model    = get_sts_model(args)
        self.sts_base_model    = self.sts_model[0].auto_model
        self.sts_pooling_layer = self.sts_model[1]
        self.sts_tokenizer = self.sts_model.tokenizer
        self.nli_tokenizer,  self.nli_model  = get_nli_tokenizer_and_model(args)
        if   args.nli_name  == "microsoft/deberta-base-mnli"     :   self.entail_label = 2
        elif args.nli_name  == "howey/electra-small-mnli"        :   self.entail_label = 0

        # Store conveniently 
        self.tokenizers = {
            "pp":  self.pp_tokenizer, 
            "ref": self.pp_tokenizer,  # same as pp
            "vm": self.vm_tokenizer, 
            "sts": self.sts_tokenizer, 
            "nli": self.nli_tokenizer
        }
        self.tokenizer_names = {k:v.__class__.__name__ for k,v in self.tokenizers.items()}
        self.tokenizer_algs = {k:TOKENIZER_ALG_MAPPING[v] for k,v in self.tokenizer_names.items()}
        self.models = { 
            "pp": self.pp_model, 
            'ref': self.ref_model,
            "vm": self.vm_model, 
            "sts": self.sts_base_model, 
            "nli": self.nli_model
        }
        for k, model in self.models.items(): 
            if k != "pp" and model.training: model.eval()

        if not 'T5Tokenizer' in self.tokenizer_names['pp'] : raise Exception("For now, paraphaser must be a T5 model.")
        self.vm_is_t5 = is_t5(self.tokenizer_names['vm'])

        ## Embeddings and sizes 
        self.emb,self.vocab_size,self.emb_size = dict(),dict(),dict()
        for model_name in self.models.keys(): 
            self.emb[model_name] = self.models[model_name].get_input_embeddings().weight
            self.vocab_size[model_name],  self.emb_size[model_name]  = self.emb[model_name].size()

        self._init_check_models_tokenizers()

        # Get token mapping
        token_mapping,self.df_token_mapping,self.token_mapping_stats = dict(),dict(),dict()
        for model_name in [o for o in self.models.keys() if o not in ['pp']]: 
            if not self.matched_vocab_sizes[model_name]: 
                token_mapping[model_name],self.df_token_mapping[model_name] = self._get_token_mapping_sparse_matrix(self.tokenizers[model_name], model_name)
                # register these so they are moved to device, saved, cast to right type, etc 
                # register buffer used for things that are model state but not trainable parameter
                self.register_buffer(f"token_mapping['{model_name}']",  token_mapping[model_name])
            self.adversary_info.update(self.token_mapping_stats) 

        self.adversary_info.update({  # for logging
                "vocab_size": self.vocab_size, 
                "emb_size":   self.emb_size,
                "matched_vocab_sizes": self.matched_vocab_sizes
        })

        # Generation parameters
        self.gen_params_train = dict(num_return_sequences=1, num_beams=1, do_sample=False, 
                                     return_dict_in_generate=True, output_scores=True)
        self.gen_config_train = GenerationConfig(**self.gen_params_train)
        self.gen_config_eval_d = dict()
        for k,v in self.args.gen_settings.items(): 
            self.gen_config_eval_d[k] = GenerationConfig(**v)
        ## Data stores 
        # training is for training, train is for eval 
        self.df_l = dict(training=[],train=[],validation=[],test=[])
        # testing 
        self.test_results_d = dict()
        for eval_setting in self.gen_config_eval_d.keys():
            if eval_setting != 'val': self.test_results_d[eval_setting] = []
        self.asr_d = dict()

    def _get_token_mapping_sparse_matrix(self, tokenizer, model_name): 
        """Get token mapping matrix between pp_model and tokenizer specified."""
        ### Identify the VM tokenizer type and the subword string
        pp_space_str = "▁"  # denotes space for T5. not actually a uscore - see: ▁_▁_▁_
        tokenizer_alg = self.tokenizer_algs[model_name]
        if tokenizer_alg == "WordPiece":   tkn_continue_str = "##"  # "##" token denotes word continuation 

        ## MATCHING
        # Get tokens from vocab and put into dataframe
        V_pp,V_tk = self.pp_tokenizer.vocab,tokenizer.vocab
        V_pp_df = pd.DataFrame({'pp_tkn':V_pp.keys(), 'pp_idx':V_pp.values()})
        V_tk_df = pd.DataFrame({'tk_tkn':V_tk.keys(), 'tk_idx':V_tk.values()})
        V_pp_df['pp_idx'] = V_pp_df['pp_idx'].astype('str')
        V_tk_df['tk_idx'] = V_tk_df['tk_idx'].astype('str')
        # Create flags for if there is any special tokens. Then strip them out
        V_pp_df['has_pp_space_str']    = V_pp_df['pp_tkn'].map(lambda x: pp_space_str in x)
        V_tk_df['has_tk_continue_str'] = V_tk_df['tk_tkn'].map(lambda x: tkn_continue_str in x)
        V_pp_df['pp_tkn'] = V_pp_df['pp_tkn'].map(lambda x: x.replace(pp_space_str, ''))
        V_tk_df['tk_tkn'] = V_tk_df['tk_tkn'].map(lambda x: x.replace(tkn_continue_str, ''))
        
        # First: special token matches
        special_mapping = dict()
        for k in self.pp_tokenizer.additional_special_tokens_ids: special_mapping[k] = tokenizer.unk_token_id  # tokens like <extra_id_2>
        if tokenizer_alg == "WordPiece":
            special_mapping.update({
                self.pp_tokenizer.pad_token_id: tokenizer.pad_token_id,
                self.pp_tokenizer.eos_token_id: tokenizer.sep_token_id,
                self.pp_tokenizer.unk_token_id: tokenizer.unk_token_id,
                3: 1517 # 3 is space str ▁, 1517 is em dash. there is no space character in wordpiece tokenizer so picking a randomish punctuation char
            })
        else:
             raise Exception("only WordPiece token algorithm supported so far for modules. need to configure the mapping including the special token map.")
        pp_special_ids = [str(o) for o in special_mapping.keys()]
        df_special_match = V_pp_df.query('pp_idx in @pp_special_ids').copy(deep=True)
        df_special_match['tk_idx'] = df_special_match['pp_idx'].map(lambda x: str(special_mapping[int(x)]))
        df_special_match = df_special_match.merge(right=V_tk_df[['tk_idx','tk_tkn']], on='tk_idx', how='left')
        df_special_match['has_pp_space_str'] = False
        df_special_match['has_tk_continue_str'] = False
        df_special_match['match_type'] = "special_tkn" 

        # CASE 1 matches:  _ with pp vocab (ie start of word), and no ## for tk (ie not middle of word)
        # e.g.: pp "_shift" to tk "shift"
        df_case1_match = pd.merge(V_pp_df.query('has_pp_space_str==True'), V_tk_df.query('has_tk_continue_str==False'), left_on='pp_tkn', right_on='tk_tkn', how='inner')
        df_case1_match['match_type'] = "case1" 

        # CASE 2 matches: no _ with pp vocab (ie not start of word), and ## for tk (ie middle of word)
        # e.g.: pp "shift" to tk "##shift"
        df_case2_match = pd.merge(V_pp_df.query('has_pp_space_str==False'), V_tk_df.query('has_tk_continue_str==True'), left_on='pp_tkn', right_on='tk_tkn', how='inner')
        df_case2_match['match_type'] = "case2" 
        
        # Tokeniser matches
        # Identify unmatched tokens, put them through tk tokenizer and get mapping
        matched_pp_idx = list(df_special_match['pp_idx']) + list(df_case1_match['pp_idx']) + list(df_case2_match['pp_idx']) 
        unmatched_pp_tokens_idxs = V_pp_df.query('pp_idx not in @matched_pp_idx')[['pp_tkn','pp_idx']].values
        l = []
        for pp_tkn,pp_idx in unmatched_pp_tokens_idxs:   
            d = dict(pp_tkn=pp_tkn, pp_idx=pp_idx, tk_idx=tokenizer(pp_tkn, add_special_tokens=False)['input_ids'])  # get tokenizer mapping of each token
            l.append(d)
        # Convert to dataframe, merge, and add weights
        df_tokeniser_match = unpack_nested_lists_in_df(pd.DataFrame(l), scalar_cols=['pp_tkn','pp_idx'])
        df_tokeniser_match['match_type']= 'tokeniser'
        df_tokeniser_match['pp_idx'] = df_tokeniser_match['pp_idx'].astype('str')
        df_tokeniser_match['tk_idx'] = df_tokeniser_match['tk_idx'].astype('str')
        df_tokeniser_match = df_tokeniser_match.merge(V_tk_df, on='tk_idx').merge(V_pp_df[['pp_idx', 'has_pp_space_str']], on='pp_idx')

        ## Concat together all match types to make final df of matches
        df_all = pd.concat([df_special_match, df_case1_match, df_case2_match, df_tokeniser_match])
        df_all['tk_idx'] = df_all['tk_idx'].astype('str')
        # Check no duplicates
        sizes =  df_all.groupby(['pp_idx'])['match_type'].nunique().to_frame('n_match_types').reset_index()
        n_duplicate_matches = len(sizes.query('n_match_types>1'))

        # Handle unmatched tokens. These are edge cases like \xad that didn't work for some reason 
        df_unmatched = V_pp_df.merge(df_all.pp_idx.to_frame('pp_idx_matched').drop_duplicates(), left_on='pp_idx',right_on='pp_idx_matched',  how='left')
        df_unmatched['unmatched'] = df_unmatched['pp_idx_matched'].isna()
        df_unmatched = df_unmatched.query('unmatched==True').copy(deep=True)
        # General workaround is to map to UNK token for these unmatched tokens
        df_unmatched['tk_tkn'] =  tokenizer.unk_token
        df_unmatched['tk_idx'] = tokenizer.unk_token_id
        df_unmatched['has_tk_continue_str'] = False
        df_unmatched['match_type'] = "unmatched" 
        df_all = pd.concat([df_all, df_unmatched])

        # Get weights
        df_all = df_all.join(1/df_all.groupby('pp_idx')['tk_idx'].size(), on='pp_idx', rsuffix='_r').rename(columns={'tk_idx_r': 'weight'})
        # Check that weights sum to 1
        weight_sums = df_all[['pp_idx','weight']].groupby('pp_idx')['weight'].sum().value_counts()
        assert weight_sums.index[0] == 1 and len(weight_sums) == 1
        # Check no duplicates in pp_idx and match_type
        assert max(df_all[['pp_idx', 'match_type']].drop_duplicates().groupby('pp_idx').size().value_counts().index) == 1
        
        # Create sparse mapping matrix 
        token_mapping = torch.sparse_coo_tensor(
            indices=np.array([df_all['pp_idx'].values.astype('int32'), df_all['tk_idx'].values.astype('int32')]), 
            values=df_all['weight'].values,
            size=(len(V_pp), len(V_tk)), device=self.device, requires_grad=False
        )
        # coalesce sums up the weights for all [pp_idx, tk_idx] duplicates. needed because sometimes you have duplicate tokens in a mapping 
        # for example, "...hello" might map to ".",".",".","hello", and weights in df_all will be 0.25 for each "." but what we really 
        # want is 0.75 weight for one "."
        # coalesce does the sum for us. if you don't do coalesce, then you get a vector of [0.25, 0.25, 0.25] at the corresponding entry 
        #  for ["...hello","."], instead of the correct scalar of 0.75. 
        token_mapping = token_mapping.coalesce()
        
        # Save stats and dataframe for logging
        df_all = df_all.drop(columns=['unmatched', 'pp_idx_matched'])
        self.token_mapping_stats[model_name] = {
            f"token_matches_unmatched_tokens" : len(df_unmatched),
            f"token_matches_duplicate" :n_duplicate_matches,
            f"token_matches_tokeniser" :len(df_tokeniser_match.pp_idx.unique()),
            f"token_matches | pp: _x -> tk: x" : len(df_case1_match),
            f"token_matches | pp: x -> tk: ##x" : len(df_case2_match),
            f"token_matches_special" : len(df_special_match),
            f"token_mapping_shape": token_mapping.shape
        }
        return token_mapping, df_all

    def _init_check_models_tokenizers(self): 
        """Checks that the models and tokenizers are set up properly."""
        ## Verify the different methods of calculating vocab and emb size give the same answer
        # NOTE: these asserts don't hold. 
        # The embedding matrix is size 32128 and the tokenizer vocab size is 32100. 
        # See https://github.com/huggingface/transformers/issues/4875 for a discussion. 
        # assert self.vocab_size_pp == self.pp_tokenizer.vocab_size
        # assert self.vocab_size_vm == self.vm_tokenizer.vocab_size
        assert self.emb_size['pp']   == self.pp_model.get_input_embeddings().embedding_dim
        assert self.emb_size['vm']   == self.vm_model.get_input_embeddings().embedding_dim
        assert self.vocab_size['pp'] == self.vocab_size['ref']  # for now - can add later but will have to token map

        # Check if the vocab's of pp and vm are matched or not. 
        self.matched_vocab_sizes = {k:v.vocab==self.pp_tokenizer.vocab for k,v in self.tokenizers.items() }

        # Check victim model and dataset have same number of classes
        if hasattr(self.vm_model, "lm_head"): assert self.dataset_num_labels == self.vm_model.lm_head.out_features  # for t5conditionalgeneration models 
        else:                                 assert self.dataset_num_labels == self.vm_model.num_labels

    def forward(self, batch):
        """Prediction/inference only"""
        # Generate, check 
        pp_logits,pp_ids = self._generate_pp(batch, gen_config=self.gen_config_train) 
        if torch.any(torch.isnan(pp_logits)): raise Exception('we are generating nans')
        self._define_check_log_orig_pp_sizes(batch, pp_ids) 
        assert pp_logits.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])
        pp_text = self.pp_tokenizer.batch_decode(pp_ids, skip_special_tokens=True)
        
        gumbel_probs = self._get_gumbel_samples(pp_logits)
        weighted_emb,model_inputs = dict(),dict()
        for model_name in ['vm','sts', 'nli', 'ref']:     
            weighted_emb[model_name] = self._construct_weighted_emb(gumbel_probs, model_name)
        for model_name in ['vm', 'sts', 'ref']:  # NLI is done seperately
            model_inputs[model_name] = self._prepare_model_inputs(pp_ids, weighted_emb[model_name], model_name)

        vm_logits = self._get_vm_logits_from_inputs_embeds(model_inputs['vm'])
        vm_scores_d = get_vm_scores_from_vm_logits_gumbel_sampled(labels=batch['label'], orig_truelabel_probs=batch['orig_truelabel_probs'], vm_logits=vm_logits)
        sts_scores, diversity_score = self._get_sts_scores_and_diversity_score_from_inputs_embeds(batch['orig_sts_embeddings'], model_inputs['sts'])
        nli_inputs = self._prepare_nli_inputs(orig_ids_nli_tknzr=batch['orig_ids_nli_tknzr'], weighted_emb_nli=weighted_emb['nli'])
        nli_scores = self._get_nli_scores_from_inputs_embeds(nli_inputs)
        kl_divs = self._get_kl_div_from_inputs_embeds(orig_ids=batch['orig_ids_pp_tknzr'], orig_attention_mask=batch['attention_mask_pp_tknzr'],
                                                      gumbel_probs=gumbel_probs, inputs=model_inputs['ref'])
        
        del weighted_emb,model_inputs
        gc.collect()
        return {'pp_text': pp_text, **vm_scores_d, 'sts_scores': sts_scores, 'nli_scores': nli_scores, 
                'kl_divs': kl_divs, 'diversity_score': diversity_score}

    def forward_eval(self, batch, eval_setting):  
        """assume no labels with this function."""
        with torch.no_grad():
            gen_config = self.gen_config_eval_d[eval_setting]
            self.batch_size_eval = batch['orig_ids_pp_tknzr'].shape[0]
            # Generate paraphrases
            _,pp_ids = self._generate_pp(batch, gen_config=gen_config) 
            pp_text = self.pp_tokenizer.batch_decode(pp_ids, skip_special_tokens=True)       
            n_pp = gen_config.num_return_sequences    

            # VM scores    
            vm_logits = get_vm_probs_from_text(text=pp_text, vm_tokenizer=self.vm_tokenizer, vm_model=self.vm_model, return_logits=True)
            assert vm_logits.shape == (self.batch_size_eval * n_pp , self.dataset_num_labels)
            def nest_tensor(x): 
                # the docs promise that we get exactly num_return_sequences*batch_size_eval paraphrases back, so we can nest them like this. 
                return x.reshape(int(x.shape[0]/n_pp), n_pp,  x.shape[1])
            vm_logits_nested = nest_tensor(vm_logits) 
            assert vm_logits_nested.shape == (self.batch_size_eval, n_pp, self.dataset_num_labels)

            # STS scores
            pp_sts_embeddings = self.sts_model.encode(pp_text, convert_to_tensor=True, device=self.device, show_progress_bar=False)
            assert pp_sts_embeddings.shape == (self.batch_size_eval * n_pp, self.emb_size['sts'])
            pp_sts_embeddings_nested = pp_sts_embeddings.reshape(int(pp_sts_embeddings.shape[0]/n_pp), n_pp,  pp_sts_embeddings.shape[1])
            assert pp_sts_embeddings_nested.shape == (self.batch_size_eval, n_pp, self.emb_size['sts'])
            sts_scores = torch.stack(list(map(lambda orig, pp: pytorch_cos_sim(orig, pp), batch['orig_sts_embeddings'], pp_sts_embeddings_nested))).squeeze()
            if n_pp == 1: assert sts_scores.shape == (self.batch_size_eval,)
            else:         assert sts_scores.shape == (self.batch_size_eval, n_pp)
            sts_scores = sts_scores.cpu().tolist()            
            # NLI scores 
            def nest(x): return x.reshape(int(x.shape[0]/n_pp), n_pp)
            orig_text = self.nli_tokenizer.batch_decode(batch['orig_ids_nli_tknzr'], skip_special_tokens=True)
            orig_text_repeated = list(itertools.chain(*[[o]*n_pp for o in orig_text]))
            assert len(orig_text_repeated) == len(batch['orig_ids_nli_tknzr']) * n_pp
            assert orig_text_repeated[0]     == orig_text_repeated[n_pp-1]
            assert orig_text_repeated[-n_pp] == orig_text_repeated[-1]
            nli_scores = get_nli_probs(orig_l=orig_text_repeated, pp_l=pp_text, nli_tokenizer=self.nli_tokenizer, nli_model=self.nli_model)[:,self.entail_label]
            assert nli_scores.shape == torch.Size([self.batch_size_eval * n_pp])
            nli_scores_nested = nest(nli_scores)
            assert nli_scores_nested.shape == torch.Size([self.batch_size_eval, n_pp])
            nli_scores_nested = nli_scores_nested.cpu().tolist()   

            # KL_DIV 
            orig_ids_repeated = batch['orig_ids_pp_tknzr'].repeat_interleave(repeats=n_pp,dim=0)
            assert orig_ids_repeated.shape == (self.batch_size_eval * n_pp, batch['orig_ids_pp_tknzr'].shape[1])
            with torch.no_grad(): 
                pp_logp  = get_logp(orig_ids_repeated, pp_ids, self.tokenizers['pp'],  self.models['pp'])
                ref_logp = get_logp(orig_ids_repeated, pp_ids, self.tokenizers['ref'], self.models['ref'])
                kl_div = pp_logp - ref_logp
                assert kl_div.shape == torch.Size([self.batch_size_eval * n_pp])
            pp_logp_nested,ref_logp_nested,kl_div_nested = nest(pp_logp), nest(ref_logp), nest(kl_div)
            assert pp_logp_nested.shape == ref_logp_nested.shape == kl_div_nested.shape == torch.Size([self.batch_size_eval, n_pp])
            pp_logp_nested,ref_logp_nested,kl_div_nested = pp_logp_nested.cpu().tolist(), ref_logp_nested.cpu().tolist(), kl_div_nested.cpu().tolist()

            # pp_text
            pp_text_nested = [pp_text[i:i+n_pp] for i in range(0, len(pp_text), n_pp)]  # put paraphrases in nested lists
            assert len(pp_text_nested) == self.batch_size_eval
            assert all([len(l)==n_pp for l in pp_text_nested])
        return {'pp_text_nested': pp_text_nested, 'vm_logits_nested': vm_logits_nested, 'sts_scores': sts_scores, 'nli_scores': nli_scores_nested,
                'pp_logp': pp_logp_nested, 'ref_logp': ref_logp_nested, 'kl_divs': kl_div_nested}

    def _generate_pp(self, batch, gen_config): 
        """Generate token-transition logits and paraphrase ids, given a batch and the generation parameters"""
        inputs = {'input_ids': batch['orig_ids_pp_tknzr'], 'attention_mask': batch['attention_mask_pp_tknzr']}
        input_len = batch['orig_ids_pp_tknzr'].shape[1]
        gen_config.min_new_tokens = int(max(0, input_len - 2 - np.floor(input_len/4)))
        gen_config.max_new_tokens = input_len + 2
        pp_output = self.pp_model.generate_with_grad(**inputs, generation_config=gen_config)
        pp_logits,pp_ids = torch.stack(pp_output.scores, dim=1),pp_output.sequences
        return pp_logits,pp_ids

    def _define_check_log_orig_pp_sizes(self, batch, pp_ids): 
        """Check that the sizes of the original and paraphrase are expected during training. Also log them. """
        self.batch_size_orig_ids_pp_tknzr,  self.batch_len_orig_ids_pp_tknzr  = batch['orig_ids_pp_tknzr'].shape 
        #  batch['orig_ids_nli_tknzr'] is a list of int
        self.batch_size_orig_ids_nli_tknzr = len(batch['orig_ids_nli_tknzr'])
        self.batch_size_pp_ids_pp_tknzr,    self.batch_len_pp_ids_pp_tknzr   =  pp_ids.shape 
        # Check batch size doesn't change after generation 
        assert self.batch_size_orig_ids_pp_tknzr == self.batch_size_orig_ids_nli_tknzr == self.batch_size_pp_ids_pp_tknzr 
        self.batch_size = self.batch_size_orig_ids_pp_tknzr
        self.log_dict({
            "batch_size":                          float(self.batch_size), 
            "batch_len_orig_ids_pp_tknzr":         float(self.batch_len_orig_ids_pp_tknzr), 
            "batch_len_pp_ids_pp_tknzr":           float(self.batch_len_pp_ids_pp_tknzr), 
        }, on_step=True)

    def _get_gumbel_samples(self, pp_logits): 
        ## Take B samples from the gumbel_softmax distribution to approximate the softmax over log_coeffs
        # This uses a default Tau temperature of 1, and uses soft probabilities rather than the 
        #   harder one-hot (hard set to False)
        self.num_gumbel_samples = self.args.num_gumbel_samples
        # gumbel softmax returns probs not logits
        # tau -> 0 makes it much harder (closer to one-hot)
        # tau -> inf makes it much softer (closer to uniform)
        # change dtype of gumbel_probs to float32
        gumbel_probs = gumbel_softmax(pp_logits.repeat(self.num_gumbel_samples, 1,1,1), tau=self.args.gumbel_tau, hard=False)
        # Add a small amount to handle non-zeros
        gumbel_probs = gumbel_probs + 1e-12
        # sums = torch.sum(gumbel_probs1, 3)
        assert gumbel_probs.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])
        return gumbel_probs

    def _construct_weighted_emb(self, gumbel_probs, model_name): 
        """Construct weighted embeddings (i.e. \"expected\" embeddings) from the per-token probabilities given in `pp_logits`. """
        assert gumbel_probs.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])
        weighted_emb_l = []
        for b in range(self.num_gumbel_samples): 
            weights = gumbel_probs[b, :, :, :]
            weights_reshaped = weights.view(-1, self.vocab_size['pp'])
            assert weights_reshaped.shape == (self.batch_size * (self.batch_len_pp_ids_pp_tknzr - 1), self.vocab_size['pp'])
            assert self.emb[model_name].shape      == (self.vocab_size[model_name], self.emb_size[model_name])
            if self.matched_vocab_sizes[model_name]:  weighted_emb = weights_reshaped.mm(self.emb[model_name])
            else:                
                # first argument to torch.sparse.mm has to be sparse, second can be sparse or dense. output: dense  
                # because of this we have to take transposes of everything and then take transpose of the final result 
                # same :  wemb  = weights @ token_map  @ emb           
                #         wembT = embT    @ token_mapT @ weightsT   
                tkn_map = self.__getattr__(f"token_mapping['{model_name}']")
                assert tkn_map.shape == (self.vocab_size['pp'], self.vocab_size[model_name])
                weighted_emb = (self.emb[model_name].double().t().mm(torch.sparse.mm(tkn_map.t(), weights_reshaped.t().double()))).t()
            assert weighted_emb.shape    == (self.batch_size * (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name] )
            weighted_emb = weighted_emb.view(-1, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])
            assert weighted_emb.shape    == (self.batch_size, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])
            weighted_emb_l.append(weighted_emb)
        weighted_emb_all = torch.stack(weighted_emb_l, dim=0)
        assert weighted_emb_all.shape    == (self.num_gumbel_samples, self.batch_size, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])
        return weighted_emb_all

    def _prepare_nli_inputs(self, orig_ids_nli_tknzr: list[int], weighted_emb_nli): 
        """The NLI base model is a BERT derivative. The specs state: 'input for a pair of sequences: [CLS] A [SEP] B [SEP]'"""
        # Input checks
        assert len(orig_ids_nli_tknzr) == self.batch_size
        # Check orig_ids_nli_tknzr starts with cls_token_id and ends with sep_token_id
        for l in orig_ids_nli_tknzr: 
            assert l[0]  == self.tokenizers['nli'].cls_token_id
            assert l[-1] == self.tokenizers['nli'].sep_token_id
        assert weighted_emb_nli.shape == (self.num_gumbel_samples, self.batch_size, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size['nli'])
        # Orig ids have format [CLS] A [SEP] which becomes orig_emb 
        # weighted_emb_nli is made up of weighted token-transition- probability sums of embedding 
        # pp_emb is transition probs, first one it p(T1|cls) , up to (SEP|T1,T2...TZ)
        # So we concat the two to get [CLS] A  [SEP] B [SEP] - that's all!
        l1 = []
        for b in range(self.num_gumbel_samples):
            l_inputs_embeds,l_token_type_ids,l_attention_mask, = list(),list(),list()
            for i in range(self.batch_size): 
                orig_emb = self.emb['nli'][orig_ids_nli_tknzr[i], :] 
                pp_emb = weighted_emb_nli[b, i,:,:] 
                # sep_emb = prep_emb(self.tokenizers['nli'].sep_token_id)
                assert orig_emb.shape == (len(orig_ids_nli_tknzr[i]),               self.emb_size['nli'])
                assert pp_emb.shape   == (self.batch_len_pp_ids_pp_tknzr - 1,       self.emb_size['nli'])
                # assert sep_emb.shape   == (1,                                       self.emb_size['nli'])
                # l_inputs_embeds.append(torch.cat([orig_emb, pp_emb, sep_emb], axis=0))
                l_inputs_embeds.append(torch.cat([orig_emb, pp_emb], axis=0))
                # 0 for first CLS, all orig input ids, and first SEP; 1 for all pp input ids and second SEP
                l_token_type_ids.append(torch.tensor([0 for i in range(orig_emb.shape[0])] + [1 for i in (range(pp_emb.shape[0]))]))
                l_attention_mask.append(torch.tensor([1 for o in range(len(l_token_type_ids[i]))]))
            inputs_embeds = torch.nn.utils.rnn.pad_sequence( l_inputs_embeds,  batch_first=True).to(self.device) # pads with 0's 
            # attention mask and token_type_ids should be the same for each gumbel sample, so we can just use the first one
            if b == 0: 
                token_type_ids = torch.nn.utils.rnn.pad_sequence(l_token_type_ids, batch_first=True).to(self.device)  
                attention_mask = torch.nn.utils.rnn.pad_sequence(l_attention_mask, batch_first=True).to(self.device)  
            l1.append(inputs_embeds)
        inputs_embeds  = torch.stack(l1, dim=0)
        # Check shapes are correct        
        max_size_orig = max([len(l) for l in orig_ids_nli_tknzr])
        self.inputs_embeds_nli_expected_size = max_size_orig  + (self.batch_len_pp_ids_pp_tknzr - 1 ) # -1 because we don't duplicate the first CLS token
        assert inputs_embeds.shape == (self.num_gumbel_samples, self.batch_size, self.inputs_embeds_nli_expected_size, self.emb_size['nli'])
        assert attention_mask.shape == token_type_ids.shape == (inputs_embeds.shape[1], inputs_embeds.shape[2]) == (self.batch_size, self.inputs_embeds_nli_expected_size)
        return {'inputs_embeds': inputs_embeds.to(torch.float32), 'token_type_ids': token_type_ids, 'attention_mask':attention_mask}

    def _prepare_model_inputs(self, pp_ids,  weighted_emb, model_name): 
        """Prepare the model inputs for component models. 
        For example, adding 'sst2 sentence:' for a t5 victim model, or the CLS token for a bert model / wordpiece model.  """
        assert pp_ids.shape       == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert weighted_emb.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.emb_size[model_name])
        weighted_emb_before_change_len = weighted_emb.shape[2]
        tokenizer_name = self.tokenizer_names[model_name]
        if model_name == "nli":  raise Exception("Shouldn't call _postprocess_weighted_emb with model_name='nli'") # we process it differently later 
        if  TOKENIZER_ALG_MAPPING[tokenizer_name] == "SentencePiece":
            # format for sentnece X:  [PAD] X [SEP]
            pad_emb = self.emb[model_name][self.tokenizers[model_name].pad_token_id,:].unsqueeze(0).repeat(self.num_gumbel_samples,self.batch_size,1,1).to(self.device)
            assert pad_emb.shape == (self.num_gumbel_samples, self.batch_size, 1, self.emb_size[model_name])
            # start_emb = self.emb[model_name][self.tokenizers[model_name].pad_token_id,:].unsqueeze(0).repeat(self.num_gumbel_samples,self.batch_size,1,1).to(self.device)
            inputs_embeds = torch.concat([pad_emb, weighted_emb], dim=2) 
            assert inputs_embeds.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size[model_name])
            pre_ids  = torch.tensor(self.tokenizers[model_name].pad_token_id).repeat(self.batch_size, 1).to(self.device) # Duplicate PAD across the batch size
        elif TOKENIZER_ALG_MAPPING[tokenizer_name] == "WordPiece":  
            # Format for sentence X : [CLS] X [SEP] 
            # The last probailities in pp_logits are to calculate p(sep|pp) and this works out as a vector pretty close to SEP anyway. So we don't need to add it on at the end. 
            # You should end up with weighted_emb being equal in shape to pp_ids. 
            start_cls_emb = self.emb[model_name][self.tokenizers[model_name].cls_token_id,:].unsqueeze(0).repeat(self.num_gumbel_samples,self.batch_size,1,1).to(self.device)
            assert start_cls_emb.shape == (self.num_gumbel_samples, self.batch_size, 1, self.emb_size[model_name])
            inputs_embeds = torch.concat([start_cls_emb, weighted_emb], dim=2) 
            assert inputs_embeds.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size[model_name])
            pre_ids  = torch.tensor(self.tokenizers[model_name].cls_token_id).repeat(self.batch_size, 1).to(self.device) # Duplicate CLS across the batch size
        else:  raise Exception(f"Unsupported tokenizer algorithm for tokenizer {tokenizer_name}")
        inputs_embeds = inputs_embeds.to(torch.float32)  # The forward method of models seems to need this. 
        attention_mask =  self.models[model_name]._prepare_attention_mask_for_generation(
                torch.concat([pre_ids, pp_ids[:,1:] ], dim=1),  # Remove the bos pad token from pp_ids + concat 
                self.tokenizers[model_name].pad_token_id, self.tokenizers[model_name].eos_token_id
        )        
        assert attention_mask.shape == (inputs_embeds.shape[1],inputs_embeds.shape[2])
        self.log_dict({
            f"weighted_emb_length_{model_name}_before_postprocessing":       float(weighted_emb_before_change_len),
            f"weighted_emb_length_{model_name}_after_postprocessing":        float(inputs_embeds.shape[1]),
        }, on_step=True)     
        return {'inputs_embeds':inputs_embeds, 'attention_mask': attention_mask}
    
    def _get_vm_logits_from_inputs_embeds(self, inputs):
        """Feed embeddings through the victim model and get logits"""
        inputs_embeds,attention_mask = inputs['inputs_embeds'], inputs['attention_mask']
        assert inputs_embeds.shape  == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size['vm'])
        assert attention_mask.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        if self.vm_is_t5: 
            start_id = self.vm_tokenizer.pad_token_id 
            decoder_input_ids = torch.full(size=(inputs_rep.shape[0], 1), fill_value=start_id, device=self.vm_model.device)
            vm_logits = self.vm_model(inputs_embeds=inputs_rep, attention_mask=attention_rep, decoder_input_ids=decoder_input_ids).logits.squeeze()
        else: 
            vm_logits = self.vm_model(inputs_embeds=inputs_rep, attention_mask=attention_rep).logits.squeeze()
        vm_logits = vm_logits.reshape((self.num_gumbel_samples, self.batch_size, self.dataset_num_labels))
        assert vm_logits.shape     == (self.num_gumbel_samples, self.batch_size, self.dataset_num_labels)
        return vm_logits

    def _get_sts_scores_and_diversity_score_from_inputs_embeds(self, orig_sts_embeddings, inputs): 
        # See the forward methods for 
        # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
        # and https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
        # for details on how this code is structured
        inputs_embeds,attention_mask = inputs['inputs_embeds'],inputs['attention_mask']
        assert orig_sts_embeddings.shape == (                         self.batch_size,                    self.emb_size['sts'])
        assert inputs_embeds.shape       == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size['sts'])  # at this point CLS should have been included at front of embedding 
        assert attention_mask.shape      == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)

        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        orig_rep = orig_sts_embeddings.repeat((self.num_gumbel_samples, 1))
        assert orig_rep.shape == (  self.num_gumbel_samples*  self.batch_size, self.emb_size['sts'])
        # forward 
        token_embeddings = self.sts_base_model.forward(inputs_embeds=inputs_rep, attention_mask=attention_rep)['last_hidden_state']
        features = dict(inputs_embeds=inputs_rep, attention_mask=attention_rep, token_embeddings=token_embeddings)
        pp_sts_embedding = self.sts_pooling_layer.forward(features)['sentence_embedding']
        ### STS SCORES
        assert pp_sts_embedding.shape == orig_rep.shape
        sts_scores = pytorch_cos_sim(orig_rep, pp_sts_embedding).diagonal()        # training case
        sts_scores = sts_scores.reshape((self.num_gumbel_samples, self.batch_size))
        assert sts_scores.shape == (self.num_gumbel_samples, self.batch_size)
        ### Intra-batch diversity penalty
        # Reshape the embeddings so that all Gumbel samples for the same example are grouped together, take mean
        pp_mean_embeddings_across_gumbel = pp_sts_embedding.reshape((self.num_gumbel_samples, self.batch_size, -1)).mean(dim=0)
        sim_matrix = pytorch_cos_sim(pp_mean_embeddings_across_gumbel, pp_mean_embeddings_across_gumbel)
        assert sim_matrix.shape == (self.batch_size, self.batch_size)
        # Compute the diversity penalty as the mean of the upper triangular part of the similarity matrix
        diversity_score = torch.mean(sim_matrix.triu(diagonal=1))
        assert diversity_score.shape == torch.Size([])
        return sts_scores,diversity_score

    def _get_nli_scores_from_inputs_embeds(self, inputs): 
        """Returns probability of entailment."""
        inputs_embeds,attention_mask,token_type_ids = inputs['inputs_embeds'],inputs['attention_mask'],inputs['token_type_ids']
        assert inputs_embeds.shape  == (self.num_gumbel_samples, self.batch_size, self.inputs_embeds_nli_expected_size, self.emb_size['nli'])
        assert attention_mask.shape == (                         self.batch_size, self.inputs_embeds_nli_expected_size)
        assert token_type_ids.shape == (                         self.batch_size, self.inputs_embeds_nli_expected_size)
        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        token_rep = token_type_ids.repeat((self.num_gumbel_samples, 1))
        assert token_rep.shape == (  self.num_gumbel_samples *  self.batch_size, token_type_ids.shape[1])
        probs = self.nli_model(inputs_embeds=inputs_rep, attention_mask=attention_rep, token_type_ids=token_rep).logits.softmax(1)[:, self.entail_label]
        probs = probs.reshape((self.num_gumbel_samples, self.batch_size))
        return probs

    def _get_kl_div_from_inputs_embeds(self, orig_ids, orig_attention_mask, gumbel_probs, inputs): 
        """orig_ids and orig_attention_mask are ids+attention mask from the orig ids from pp_tokenizer. 
        inputs are a dict of the processed inputs_embeds and the corresponding attention mask. 
        """
        inputs_embeds,attention_mask_inputs = inputs['inputs_embeds'], inputs['attention_mask']
        assert inputs_embeds.shape         == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr,   self.emb_size['ref'])
        assert attention_mask_inputs.shape == (                         self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert gumbel_probs.shape      ==     (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr-1, self.vocab_size['pp'])
        pp_probs_all = gumbel_probs
        pp_logprobs_all = torch.log(pp_probs_all)
        pp_logprobs_all = torch.nan_to_num(pp_logprobs_all, nan=None, posinf=None, neginf=-20)  # -inf screws things up
        pp_logprobs_all = pp_logprobs_all.clip(min=-20)
        assert pp_probs_all.shape == pp_logprobs_all.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr-1, self.vocab_size['pp'])
        l1 = []
        for b in range(self.num_gumbel_samples):
            kl_divs_l = []
            for i in range(self.batch_len_pp_ids_pp_tknzr - 1):
                decoder_inputs_embeds = inputs_embeds[b, :, 0:(i+1), :]
                assert decoder_inputs_embeds.shape == (self.batch_size, i+1, self.emb_size['ref'])
                input_d = {'input_ids':orig_ids, 'attention_mask':orig_attention_mask, 'decoder_inputs_embeds':decoder_inputs_embeds}
                ref_outputs = self.models['ref'](**input_d)
                assert ref_outputs.logits.shape == (self.batch_size, i+1, self.vocab_size['ref'])
                ref_logprobs = ref_outputs.logits[:, i, :].log_softmax(1)
                ref_logprobs = torch.nan_to_num(ref_logprobs, nan=None, posinf=None, neginf=-20)  # -inf screws things up
                ref_logprobs = ref_logprobs.clip(min=-20)
                assert ref_logprobs.shape == (self.batch_size, self.vocab_size['ref'])
                pp_probs    = pp_probs_all   [b, :, i, :]
                pp_logprobs = pp_logprobs_all[b, :, i, :]
                assert pp_probs.shape == pp_logprobs.shape == (self.batch_size, self.vocab_size['pp'])
                # KL div:  p_pp dot (log(p_pp) -  log(p_ref))
                kl_div = torch.mm(pp_probs, (pp_logprobs - ref_logprobs).t()).diagonal()
                assert kl_div.shape == torch.Size([self.batch_size])
                kl_divs_l.append(kl_div)
            kl_divs = torch.stack(kl_divs_l, 1)
            assert kl_divs.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr-1)
            kl_divs = kl_divs * attention_mask_inputs[:, 1:]  # account for padding 
            kl_divs_normalised = kl_divs.sum(1) / attention_mask_inputs[:, 1:].sum(1)  # normalise for length of generated sequence
            assert kl_divs_normalised.shape == torch.Size([self.batch_size])
            l1.append(kl_divs_normalised)
        kl_divs_stacked = torch.stack(l1, 0)
        assert kl_divs_stacked.shape == (self.num_gumbel_samples, self.batch_size)
        return kl_divs_stacked

    def _loss_fn_example(self, vm_scores, sts_scores, nli_scores, **kwargs): 
        if type(vm_scores)  == float: vm_scores  = torch.tensor(vm_scores)
        if type(sts_scores) == float: sts_scores = torch.tensor(sts_scores)
        if type(nli_scores) == float: nli_scores = torch.tensor(nli_scores)
        vm_component  = self.args.coef_vm  * vm_scores
        sts_component = self.args.coef_sts * sts_scores
        nli_component = self.args.coef_nli * nli_scores
        if 'training' in kwargs and kwargs['training'] == True: 
            self.log_dict({'vm_component_batch_mean': torch.mean(vm_component).item(), 
                       'sts_component_batch_mean':torch.mean(sts_component).item(), 
                       'nli_component_batch_mean':torch.mean(nli_component).item()}, on_step=True)
        return -(vm_component + sts_component + nli_component)

    def loss_fn(self, vm_scores, sts_scores, nli_scores, kl_divs, diversity_score, **kwargs): 
        """batch one"""
        # Loss clipping
        vm_scores_clipped  = torch.where(vm_scores  < (1 - (1.0/self.dataset_num_labels)),  vm_scores,  torch.zeros_like(vm_scores))        
        sts_scores_clipped = torch.where(sts_scores < self.args.eval_sts_threshold,         sts_scores, torch.zeros_like(sts_scores))
        nli_scores_clipped = torch.where(nli_scores < self.args.eval_nli_threshold,         nli_scores, torch.zeros_like(nli_scores))
        kl_divs_clipped    = torch.where(kl_divs    > self.args.eval_kl_threshold,          kl_divs,    torch.zeros_like(kl_divs))
        # Penalties. Lower is better. In range [0, inf) - unbounded positively. 
        kl_penalty =  self.args.coef_kl * torch.mean(kl_divs)   # lower (towards 0) better 
        diversity_penalty = self.args.coef_diversity * diversity_score   # lower (towards 0) better 
        # Final loss fn
        loss_examples = self._loss_fn_example(vm_scores=vm_scores_clipped, sts_scores=sts_scores_clipped, nli_scores=nli_scores_clipped, training=True) 
        loss_examples_mean = torch.mean(loss_examples)
        loss_batch = loss_examples_mean + kl_penalty + diversity_penalty  # want as negative as possible
        if not self.args.less_logging:   # Log histograms to W&B
            self.logger.experiment.log({"vm_scores_clipped":  wandb.Histogram(vm_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
            self.logger.experiment.log({"sts_scores_clipped": wandb.Histogram(sts_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
            self.logger.experiment.log({"nli_scores_clipped": wandb.Histogram(nli_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
            self.logger.experiment.log({"kl_divs_clipped":    wandb.Histogram(kl_divs_clipped.cpu().detach().numpy()), "step": self.global_step})
        self.log("kl_penalty_batch",         kl_penalty.item(),          on_step=True)
        self.log("diversity_penalty_batch",  diversity_penalty.item(),   on_step=True)
        self.log("loss_examples_batch_mean", loss_examples_mean.item(),  on_step=True)
        self.log("loss_batch",               loss_batch.item(),          on_step=True)
        return {'loss_batch':loss_batch}
        
    def training_step(self, batch, batch_idx):
        """complete training loop"""
        if not self.pp_model.training: self.pp_model.training() 
        for k,model in self.models.items(): 
            if k != "pp" and model.training: model.eval() # lightning seems to automatically set models out of eval mode sometimes
        forward_d = self(batch)
        loss_d = self.loss_fn(**forward_d)        
        # detach values to save memory because we are just logging these values to a csv
        for k,v in forward_d.items(): 
            if type(v) is torch.Tensor:
                if v.grad_fn is not None: 
                    forward_d[k] = v.detach()
                forward_d[k] = forward_d[k].cpu()
        return {'loss': loss_d['loss_batch'], **{k: v for k, v in batch.items() if k not in self.orig_cols_to_ignore}, **forward_d}  # must include the key 'loss' 
  
    def eval_step(self, batch, batch_idx, eval_setting): 
        if self.vm_model.training: self.vm_model.eval()
        if self.pp_model.training: self.pp_model.eval()
        forward_d = self.forward_eval(batch, eval_setting)
        return {**{k: v for k, v in batch.items() if k not in self.orig_cols_to_ignore},  **forward_d}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Complete validation loop"""
        results = self.eval_step(batch, batch_idx, eval_setting='val')
        return results
        
    def test_step(self, batch, batch_idx):
        """complete testing loop"""
        for eval_setting in self.gen_config_eval_d.keys(): 
            if eval_setting != 'val': 
                self.test_results_d[eval_setting].append(self.eval_step(batch, batch_idx, eval_setting))
        return self.test_results_d

    def predict_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx)
        return results

    def is_label_flip(self, labels, vm_predclass): return ((vm_predclass != labels) * 1)

    def is_valid_pp(self, sts_scores, nli_scores, kl_divs):  
        return (sts_scores > self.args.eval_sts_threshold and nli_scores > self.args.eval_nli_threshold
                 and kl_divs < self.args.eval_kl_threshold)

    def is_adv_example(self, label_flip, is_valid): return label_flip and is_valid

    def _convert_end_of_epoch_metrics_to_pandas_df(self, outputs, split, gen_setting=None): 
        """Takes the output metrics, converts them to a pandas dataframe, 
            and appends it to the appropiate entry in self.df_l"""
        # Outputs -> dataframe skeleton 
        df = pd.DataFrame(outputs).apply(pd.Series.explode).reset_index(drop=True)  # list[dict] -> dataframe. one row per orig
        df = df.applymap(lambda x: x.item() if is_0d_tensor(x) else x) #  one-element tensors -> float/int scalars
       
        for k,v in outputs[0].items(): print(k, type(v), v.shape if type(v) is torch.Tensor else None, len(v) if type(v) is list else None)
        # Eval-specific preprocessing.
        if split != "training":
            # Eval has multiple pp per orig: train has one
            df = unpack_nested_lists_in_df(df, scalar_cols=df.select_dtypes(np.number).columns.tolist())  # one row per pp. scalar_cols fn will not pick up tensors
            df = df.rename({'pp_text_nested': 'pp_text'}, axis=1)
             # Add vm scores (already included in training)
            vm_scores_d = get_vm_scores_from_vm_logits(
                labels=torch.tensor(df['label'].values, device=self.device),
                orig_truelabel_probs=torch.tensor(df['orig_truelabel_probs'].values, device=self.device),
                vm_logits=torch.stack(df['vm_logits_nested'].values.tolist())
            )
            for k, v in vm_scores_d.items(): vm_scores_d[k] = v.cpu()
            df = pd.concat([df, pd.DataFrame(vm_scores_d)], axis=1)
            df = df.drop(columns='vm_logits_nested')

        # Add metric columns  
        df['epoch'] =  self.current_epoch #  self.best_epoch if split=="test" else
        def get_mean(c): return np.mean(df[f'{c}_mean'])  # we drop incomplete batches so they should all be means over the same number of elements
        s = 'train' if split == "training" else split
        if split == 'training': 
            # Metric dict for wandb
            d = {
                f'vm_score_{s}_mean': get_mean('vm_scores'),
                f'sts_score_{s}_mean': get_mean('sts_scores'), 
                f'nli_score_{s}_mean': get_mean('nli_scores'), 
                f'kl_div_{s}_mean': get_mean('kl_divs'),
            }
        if split != 'training': 
            # Calc paraphrase index and evaluation metric
            df['loss_example'] =  df.apply(lambda x: self._loss_fn_example(**x).item(), axis=1)
            df['pp_idx'] = df.groupby(['idx']).cumcount()
            df['label_flip'] = df.apply(lambda x: self.is_label_flip(labels=x.label,  vm_predclass=x.vm_predclass), axis=1)
            df['is_valid_pp'] = df.apply(lambda x: self.is_valid_pp(sts_scores=x.sts_scores, nli_scores=x.nli_scores, kl_divs=x.kl_divs) * 1, axis=1) 
            df['is_adv_example'] = df.apply(lambda x: self.is_adv_example(label_flip=x.label_flip, is_valid=x.is_valid_pp) * 1, axis=1)
            any_label_flip = (df.groupby('idx')['label_flip'].mean() > 0)*1
            any_adv_example = (df.groupby('idx')['is_adv_example'].mean() > 0)*1
            g = f'_{gen_setting}' if gen_setting else ""
            d = dict()
            if split == 'validation': # don't need these for test
                d = {
                    f'loss_{s}{g}_mean': df.loss_example.mean(), 
                    f'vm_score_{s}{g}_mean': df.vm_scores.mean(), 
                    f'sts_score_{s}{g}_mean': df.sts_scores.mean(), 
                    f'nli_score_{s}{g}_mean': df.nli_scores.mean(), 
                    f'kl_div_{s}{g}_mean': df.kl_divs.mean()
                }
                d[f'any_adv_example_proportion_{s}{g}'] = any_adv_example.mean()
                d[f'ref_logp_{s}{g}_mean'] = df['ref_logp'].mean()
                d[f'pp_logp_{s}{g}_mean']   = df['pp_logp'].mean()
                d[f'label_flip_{s}{g}_mean']    =  df['label_flip'].values.mean()
                d[f'is_valid_pp_{s}{g}_mean']   =  df['is_valid_pp'].values.mean()
                d[f'is_adv_example_{s}{g}_mean']=  df['is_adv_example'].values.mean()
            u = f"_untrained" if self.untrained_run else ""
            d[f'{s}_attack_success_rate{u}{g}'] = any_label_flip.mean()
            self.asr_d[f'{s}_attack_success_rate{u}{g}'] = d[f'{s}_attack_success_rate{u}{g}']
        # Log + return
        self.log_dict(d, on_epoch=True, sync_dist=True)
        self.df_l[split].append(df)
        return d

    def training_epoch_end(self, outputs: List[dict]):
        # We need to summarise metrics across all gumbel samples
        gumbel_keys = [k for k,v in outputs[0].items() if type(v) is not list and len(v.shape) == 2]
        results_l = list() 
        for output in outputs:
            results_d = dict()
            for k in gumbel_keys: 
                if k =='vm_predclass':  # We just look at the fraction of label flips
                    vm_predclass_same_labels = output['label'].repeat((self.num_gumbel_samples, 1)).cpu() != output['vm_predclass']
                    assert vm_predclass_same_labels.shape == (self.num_gumbel_samples, output['vm_predclass'].shape[1])
                    label_flip_frac = torch.sum(vm_predclass_same_labels, axis=0) / self.num_gumbel_samples
                    results_d['label_flip_frac'] = label_flip_frac
                else: 
                    # all other metrics - just get mean and (min, 25%, 50%, 75%, max)
                    results_d[f'{k}_mean'] = output[k].mean(axis=0)
                    summary_stats = output[k].quantile(torch.tensor([0, 0.25, 0.5, 0.75, 1]), dim=0, interpolation='linear')
                    results_d[f'{k}_quantiles']= summary_stats.t().tolist()
            results_l.append(results_d)                    
        assert(len(results_l) == len(outputs))
        new_outputs = [{**{k: v for k, v in d_orig.items() if k not in gumbel_keys}, **d_new} for d_orig,d_new in zip(outputs, results_l)]
        assert len(new_outputs) == len(outputs)

        self._convert_end_of_epoch_metrics_to_pandas_df(new_outputs, split='training')

    def validation_epoch_end(self, outputs):
        self._convert_end_of_epoch_metrics_to_pandas_df(outputs, split='validation')

    def test_epoch_end(self, results_d): 
        # results_d: dict with keys as generation settings, values as output from eval_step (like validation_epoch_end)
        # Set up evaluation metrics
        self._move_models_to_device('cpu')  # clear GPU space 
        self.BARTScore = OpenAttack.metric.BARTScore()
        self.BERTScore = OpenAttack.metric.BERTScore()
        self.ENTAIL = OpenAttack.metric.EntailmentProbability("howey/electra-small-mnli")
        for k, outputs in self.test_results_d.items(): 
            self.test_fname = f'{self.path_run}/test{"_untrained" if self.untrained_run else ""}_{k}.csv'
            self._convert_end_of_epoch_metrics_to_pandas_df(outputs, split='test', gen_setting=k)
            self.save_results_df(split='test', untrained_run=self.untrained_run, gen_setting=k)
            self.log_baseline_metrics_on_test_set(untrained_run=self.untrained_run, gen_setting=k)
        # clean up
        del self.BARTScore
        del self.BERTScore
        del self.ENTAIL
        torch.cuda.empty_cache()
        gc.collect()
        self._move_models_to_device(self.device)
        # reset testing datastores
        self.test_results_d = dict()
        for eval_setting in self.gen_config_eval_d.keys():
            if eval_setting != 'val': self.test_results_d[eval_setting] = []
 
    def _move_models_to_device(self, device): 
        for k,model in self.models.items():
            for param in model.parameters():
                param.data = param.data.to('cpu')
        for model_name, vocab_matched in self.matched_vocab_sizes.items(): 
            if not vocab_matched: 
                self.__setattr__(f"token_mapping['{model_name}']",self.__getattr__(f"token_mapping['{model_name}']").to('cpu') )
        torch.cuda.empty_cache()
        gc.collect()

    def save_results_df(self, split, untrained_run=False, gen_setting=None): 
        df_all = pd.concat(self.df_l[split])
        df_dataset = pd.read_csv(f'{self.path_run}/orig_{"train" if split == "training" else split}.csv')
        cols_to_remove = list(set(df_all.columns.to_list()).intersection(set(df_dataset.columns.to_list()))); cols_to_remove.remove('idx')
        cols_to_remove += self.orig_cols_to_ignore        
        df_dataset = df_dataset.drop(columns=cols_to_remove, errors='ignore')
        df_final = pd.merge(left=df_all, right=df_dataset, on='idx', how='left')
        fname =  self.test_fname if split =="test" else f'{self.path_run}/{split}{"_untrained" if untrained_run else ""}.csv'
        df_final.to_csv(fname, index=False)
        if untrained_run or split=='test': self.df_l[split] = []  # reset when doing the epoch 0/untrained ones
            
    def on_train_end(self): 
        # runs after the whole training and validation procedure
        # test is in self.test_epoch_end
        self.save_results_df(split='training')
        self.save_results_df(split='validation')

    def log_baseline_metrics_on_test_set(self, untrained_run=False, gen_setting=None):
        # Load dataset and select only label flips
        df_label_flips  = pd.read_csv(self.test_fname).query('label_flip==1')
        if len(df_label_flips) == 0: 
            print(f'No label flips detected in {self.test_fname}. Skipping metrics')
            return
        # group by idx and select the best example per idx, based on pseudo loss excluding VM score (more negative loss is better)
        def get_selection_score(sts_score, nli_score, kl_div): 
            return sts_score + nli_score - kl_div 
        df_label_flips['loss_pseudo'] = df_label_flips.apply(lambda x:  get_selection_score( 
            sts_score=x.sts_scores, nli_score=x.nli_scores, kl_div=x.kl_divs), axis=1) 
        
        def select_best_row(group):
            """Select the row with the highest pseudo_loss from rows with is_valid==1,
            or the row with the highest loss if no rows have is_valid==1"""
            valid_rows = group[group['is_valid_pp'] == 1]
            if len(valid_rows) > 0:  return valid_rows.loc[valid_rows['loss_pseudo'].idxmax()]
            else:                    return group.loc[group['loss_pseudo'].idxmax()]

        df_label_flips['loss_pseudo'] = df_label_flips.apply(lambda x: get_selection_score(
            sts_score=x.sts_scores, nli_score=x.nli_scores, kl_div=x.kl_divs), axis=1)
        df_label_flips = df_label_flips.groupby('idx').apply(select_best_row).reset_index(drop=True)
        df_label_flips = df_label_flips.sort_values('loss_pseudo', ascending=False).groupby('idx').head(1)
        if type(df_label_flips) == pd.Series: df_label_flips = df_label_flips.to_frame().T
        df_label_flips = df_label_flips.rename(columns={'pp':'pp_text', 'orig': 'text'})
        df_label_flips = df_label_flips.rename(columns={'sentence': 'text'})

        with torch.no_grad(): 
            df_label_flips['BARTScore']     = df_label_flips.apply(lambda x: self.BARTScore.calc_score(sentA=x.text, sentB = x.pp_text),  axis=1)
            df_label_flips['BERTScore']     = df_label_flips.apply(lambda x: self.BERTScore.calc_score(sentA=x.text, sentB = x.pp_text),  axis=1)
            df_label_flips['entailment']    = df_label_flips.apply(lambda x: self.ENTAIL.calc_score(   sentA=x.text, sentB = x.pp_text), axis=1)
        u = "_untrained"      if untrained_run else ""
        g = f"_{gen_setting}" if gen_setting   else ""
        d = {
            f"test_BERTScore{u}{g}_avg":        df_label_flips['BERTScore'].mean(),
            f"test_BERTScore{u}{g}_median":     df_label_flips['BERTScore'].median(),
            f"test_BARTScore{u}{g}_avg":        df_label_flips['BARTScore'].mean(),
            f"test_BARTScore{u}{g}_median":     df_label_flips['BARTScore'].median(),
            f"test_entailment{u}{g}_avg":       df_label_flips['entailment'].mean(),
            f"test_entailment{u}{g}_median":    df_label_flips['entailment'].median(),
        }
        def search_metric(asr, bartscore, ent): return min(asr, 0.8) + ent + 2*np.exp(bartscore)
        d[f'search_metric{u}{g}'] = search_metric(asr=self.asr_d[f'test_attack_success_rate{u}{g}'] , 
             bartscore=d[f"test_BARTScore{u}{g}_median"], ent=d[f"test_entailment{u}{g}_median"])
        self.log_dict(d)
        fname = self.test_fname[:-4] + "_evaluation.csv"
        df_label_flips.to_csv(fname, index=False)

    def configure_optimizers(self):
        """optimizers"""
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr = self.learning_rate)
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.args.optimizer_type == 'AdaFactor': 
            self.learning_rate = 3e-4 if self.args.learning_rate is None else self.args.learning_rate  # a good default for adafactor
            optimizer = Adafactor(parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=self.learning_rate)
        elif self.args.optimizer_type == 'AdamW': 
            # seems to be a good default for adamw too, by coincidence same as adafactor 
            self.learning_rate = 3e-4 if self.args.learning_rate is None else self.args.learning_rate  
            optimizer = AdamW(parameters, lr=self.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):     
        parser = parent_parser.add_argument_group("T5ForTextClassification")
        parser.add_argument("--detailed_logging", type=lambda x: bool(strtobool(x)), default=True, 
            help="set to true to log detailed logs for a few examples into the log file (used for debugging).")
        parser.add_argument("--num_gumbel_samples", type=int, help="number of gumbel samples to take for each input")
        parser.add_argument("--gumbel_tau", type=float, help="Tau parameter for gumbel-softmax sampling.")
        
        ## LOSS FN 
        parser.add_argument("--coef_vm",  type=float,  help="coefficient for the vm scores in the exampleloss (should be +ve)")
        parser.add_argument("--coef_sts", type=float,  help="coefficient for the sts scores in the example loss (should be +ve)")
        parser.add_argument("--coef_nli", type=float,  help="coefficient for the nli scores in the example loss (should be +ve)")
        parser.add_argument("--coef_kl",  type=float,  help="coefficient for the kl divergence in the batch loss (should be +ve)")
        parser.add_argument("--coef_diversity",  type=float, help="coefficient for the diversity component in the  batch loss (should be +ve)")

        ## EVAL THRESHOLDS
        parser.add_argument("--eval_sts_threshold",  type=float, help="minimum vm score for a valid adv example during eval")
        parser.add_argument("--eval_nli_threshold",  type=float, help="maximum nli score for a valid adv example during eval")
        parser.add_argument("--eval_kl_threshold",   type=float, help="maximum kl div for a valid adv example during eval")
        ## GEN PARAMETERS
        parser.add_argument("--eval_condition",            type=str,  choices=['standard', 'dev', 'ablation'],
            help="Evaluation preset conditions for generation. Will set all the eval generation paramters. See utils.set_eval_gen_settings_from_eval_condition")
        return parent_parser

