## Misc file to hold source code from tests

from __future__ import annotations 
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Only imports the below statements during type checking
    from src.adversary import WhiteboxAdversary
import torch
import src.adversary as adversary
from src.utils import round_t
from sentence_transformers.util import pytorch_cos_sim

ORIG_L = [
        "I like this movie.", 
        "I like this movie.", 
        "I did not like that movie.",
        "I did not like that movie.", 
        "none of this is half as moving as the filmmakers seem to think ."
    ]
PP_L = [
        "This movie was great. ",
        "This movie was terrible. ",
        "This movie was great. ",
        "This movie was terrible. ",
        "none of this is half as moving as the filmmakers seem to think............."
    ]


def run_tests(self: WhiteboxAdversary): 
    """Run all tests."""
    # self.test_model_input_is_correct_nli_model(self)
    self.test_get_nli_scores_from_inputs_embeds(self)
    self.test_get_sts_scores_from_inputs_embeds(self)
    self.test_get_vm_logits_from_inputs_embeds(self)

def test_get_nli_scores_from_inputs_embeds(self:WhiteboxAdversary):
    """Test that _get_nli_scores_from_inputs_embeds returns the same results as the model when given the same inputs."""
    tokenizer_inputs = _get_tokenizer_inputs(self, ORIG_L,PP_L, 'nli')
    # results from token_ids
    input_ids_results = self.models['nli'](**tokenizer_inputs).logits.softmax(1)[:, self.entail_label]  # probs
    # result using function
    emb = self.emb['nli'][tokenizer_inputs['input_ids']]
    input_d = {'inputs_embeds': emb, 'attention_mask': tokenizer_inputs['attention_mask'], 'token_type_ids': tokenizer_inputs['token_type_ids']}
    self.batch_size = tokenizer_inputs['input_ids'].shape[0]
    self.inputs_embeds_nli_expected_size = tokenizer_inputs['input_ids'].shape[1]
    fn_results = self._get_nli_scores_from_inputs_embeds(input_d)
    assert torch.allclose(fn_results, input_ids_results)

def test_get_sts_scores_from_inputs_embeds(self:WhiteboxAdversary):
    """Test fn gives correct outputs."""
    # results using encode from raw text
    encode_results = pytorch_cos_sim(self.sts_model.encode(ORIG_L), self.sts_model.encode(PP_L)).diagonal().to(self.device)
    # result using input_ids 
    tokenizer_inputs = _get_tokenizer_inputs(self, ORIG_L,PP_L, 'sts')
    orig_emb = self.sts_model.forward({**tokenizer_inputs['orig']})['sentence_embedding']
    pp_emb   = self.sts_model.forward({**tokenizer_inputs['pp']})['sentence_embedding']
    input_ids_results =  pytorch_cos_sim(orig_emb,pp_emb).diagonal()
    # result using function 
    orig_sts_embeddings = torch.tensor(self.sts_model.encode(ORIG_L), device=self.device)
    fn_inputs = {'inputs_embeds':self.emb['sts'][tokenizer_inputs['pp']['input_ids']].to(self.device), 
                    'attention_mask':tokenizer_inputs['pp']['attention_mask'].to(self.device)}
    self.batch_size = tokenizer_inputs['pp']['input_ids'].shape[0]
    self.batch_len_pp_ids_pp_tknzr = tokenizer_inputs['pp']['input_ids'].shape[1]
    fn_results, _ = self._get_sts_scores_and_diversity_score_from_inputs_embeds(orig_sts_embeddings, fn_inputs)
    assert torch.allclose(encode_results, fn_results)
    assert torch.allclose(encode_results, input_ids_results)
    assert torch.allclose(input_ids_results, fn_results)
    
def test_get_vm_logits_from_inputs_embeds(self:WhiteboxAdversary): 
    """Test that _get_vm_logits_from_inputs_embeds returns the same results as the model when given the same inputs."""
    tokenizer_inputs = _get_tokenizer_inputs(self, ORIG_L,PP_L, 'vm')
    # results from input_ids
    input_ids_results = self.models['vm'](**tokenizer_inputs).logits
    # result using function
    emb = self.emb['vm'][tokenizer_inputs['input_ids']]
    input_d = {'inputs_embeds': emb, 'attention_mask': tokenizer_inputs['attention_mask']}
    self.batch_size = tokenizer_inputs['input_ids'].shape[0]
    self.batch_len_pp_ids_pp_tknzr = tokenizer_inputs['input_ids'].shape[1]
    fn_results = self._get_vm_logits_from_inputs_embeds(input_d)
    assert torch.allclose(fn_results, input_ids_results)

def _get_tokenizer_inputs(self:WhiteboxAdversary, orig_l, pp_l, model_name): 
    """Set up the 'expected' input according to the tokenizers. Returns a dict"""
    if model_name == "vm": 
        inputs = self.tokenizers['vm'](pp_l, padding=True, truncation=True, return_tensors='pt').to(self.device)
    elif model_name =='nli': 
        inputs = self.tokenizers['nli'](orig_l, pp_l, return_tensors="pt", padding=True, truncation=True).to(self.device)
    elif model_name == "sts": 
        inputs = dict()
        inputs['orig'] = self.tokenizers['sts'](orig_l, return_tensors="pt", padding=True, truncation=True).to(self.device)
        inputs['pp']   = self.tokenizers['sts'](pp_l,   return_tensors="pt", padding=True, truncation=True).to(self.device)
    return inputs


####
def test_model_input_is_correct_nli_model(self:WhiteboxAdversary, orig_l, pp_l): 
    """Get preprocessing steps down for the NLI model. """
    # The model output should have three keys that take the following format
    # 1. input ids. CLS orig SEP pp SEP pad if needed. Note pad is at the end 
    # 2. token_type ids. 0 for first seq, 1 for 2nd seq, then 0 for pad at end. 
    # 3. attention mask. 1, 0 for pad. 
    orig_ids = self.tokenizers['nli'](orig_l, padding=False) # both orig and pp as list, no pad
    pp_ids   = self.tokenizers['nli'](pp_l,   padding=False)
    # append together
    l_input_ids, l_token_type_ids = list(),list()
    for s1,s2 in zip(orig_ids['input_ids'], pp_ids['input_ids']): 
        s2_without_cls = s2[1:]
        l_input_ids.append(s1 + s2_without_cls) # remove the CLS from the start of s2    
        l_token_type_ids.append([0 for i in s1] + [1 for i in s2_without_cls])
    tkns = self.tokenizers['nli'].pad({'input_ids':l_input_ids}, padding=True, return_tensors='pt').to(self.device)
    input_ids,attention_mask = tkns['input_ids'],tkns['attention_mask']
    token_type_ids = self.tokenizers['nli'].pad({'input_ids':l_token_type_ids}, padding=True, return_tensors='pt').to(self.device)['input_ids']
    inputs_done_manually =  {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask':attention_mask}
    # This is the reference
    inputs_from_tokenizer = self.tokenizers['nli'](orig_l, pp_l, return_tensors="pt", padding=True, truncation=True).to(self.nli_model.device)
    for k in inputs_done_manually.keys(): assert torch.equal(inputs_done_manually[k], inputs_from_tokenizer[k])

def get_model_results_from_input_ids(self:WhiteboxAdversary, orig_l, pp_l, model_name):
    """The usual way to get model output"""
    with torch.no_grad():
        if model_name in ['vm','nli']: 
            inputs = _get_tokenizer_inputs(self, orig_l, pp_l, model_name)
            return self.models[model_name](**inputs).logits.softmax(1)  # probs
        elif model_name == 'sts': 
            inputs = {'orig_sts_embeddings'  : self.sts_model.encode(orig_l), 
                        'pp_sts_embeddings'  : self.sts_model.encode(pp_l)}
            return pytorch_cos_sim(inputs['orig_sts_embeddings'], inputs['pp_sts_embeddings']).diagonal()


def test_input_ids_and_input_embeds_gives_same_answer(self, orig_l, pp_l): 


    def get_model_results_from_inputs_embeds(orig_l, pp_l, model_name): 
        """The model output from using embeddings."""
        inputs = _get_tokenizer_inputs(orig_l, pp_l, model_name)
        if model_name == 'nli':
            # These two give equivalent results 
            # 1. 
            # emb = self.models['nli'].electra.embeddings.word_embeddings(inputs['input_ids'])
            # 2. 
            emb = self.emb['nli'][inputs['input_ids']]
            input_d = {'inputs_embeds': emb, 'attention_mask': inputs['attention_mask'], 'token_type_ids': inputs['token_type_ids']}
        elif model_name == 'vm': 
            # These two as well 
            # 1. 
            # emb = self.models['vm'].distilbert.embeddings.word_embeddings(inputs['input_ids'])
            # 2. 
            emb = self.emb['vm'][inputs['input_ids']]
            input_d = {'inputs_embeds': emb, 'attention_mask': inputs['attention_mask']}
        elif model_name =='sts': 
            sts_emb = dict()
            for k in ['orig', 'pp']:
                emb = self.emb['sts'][inputs[k]['input_ids']]
                features = {'inputs_embeds': emb, 'attention_mask': inputs[k]['attention_mask']}
                sts_base_output = self.sts_base_model.forward(**features)
                features['token_embeddings'] = sts_base_output['last_hidden_state']
                sts_emb[k] = self.sts_pooling_layer.forward(features)['sentence_embedding']
            return pytorch_cos_sim(sts_emb['orig'], sts_emb['pp']).diagonal()
        with torch.no_grad():
            return self.models[model_name](**input_d).logits.softmax(1)

    # Go through each model and print out the results. Should be the same. 
    for model_name in ['vm', 'nli', 'sts']:
        print(f'\n##### {model_name} #####')
        text_probs  = get_model_results_from_input_ids(    orig_l, pp_l, model_name)
        embed_probs = get_model_results_from_inputs_embeds(orig_l, pp_l, model_name)
        print(round_t(text_probs))
        print(round_t(embed_probs))