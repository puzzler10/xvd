import pandas as pd 
from src.utils import * 
from ast import literal_eval
from pprint import pprint

ORIG_COLS = ['idx', 'orig', 'label', 'orig_truelabel_probs']
METRIC_COLS=dict()
METRIC_COLS['training']   = ['label_flip_frac', 'vm_truelabel_probs_mean', 'vm_truelabel_probs_quantiles',
       'vm_scores_mean', 'vm_scores_quantiles', 'sts_scores_mean', 'sts_scores_quantiles', 'nli_scores_mean', 'nli_scores_quantiles',
       'kl_divs_mean', 'kl_divs_quantiles']

METRIC_COLS['validation'] = METRIC_COLS['test'] = ['vm_truelabel_probs', 'vm_scores', 'sts_scores', 'nli_scores','kl_divs',
         'label_flip', 'is_valid_pp', 'is_adv_example', 'ref_logp', 'pp_logp' ]
PP_COLS = dict()
PP_COLS['training']   = ['epoch',          'pp']
PP_COLS['validation'] = ['epoch', 'pp_idx','pp', 'pp_predclass']
PP_COLS['test']       = ['epoch', 'pp_idx','pp', 'pp_predclass']

def get_dfs(path_run):
    """Return a dict of dataframes with all CSV files in path_run""" 
    df_d = dict()
    print("{:<42} {:<12}".format("Name", "Shape"))
    # Loop over all files in the directory
    for filename in os.listdir(path_run):
        # Check if file is a CSV
        if filename.endswith(".csv"):
            try:
                # Remove .csv extension to use as dictionary key
                key = filename[:-4]
                # Full file path
                fname = os.path.join(path_run, filename)
                df_d[key] = pd.read_csv(fname)
                pprint("{:<40} {:<12}".format(key, str(df_d[key].shape)))
            except FileNotFoundError:
                pass
    return df_d


def rename_df_cols(df): 
    df=df.rename(columns={
        'text': 'orig', 
        'text_with_prefix': 'orig_with_prefix', 
        'pp_text': 'pp', 
        'vm_predclass': 'pp_predclass'
    })
    df=df.rename(columns={
        'sentence': 'orig', 
        'sentence_with_prefix': 'orig_with_prefix', 
    })

    return df 

def preprocess_df(df, split): 
    evaluation_csv = True if "_evaluation" in split else False
    split = "test" if "test" in split else split
    cols = ORIG_COLS + PP_COLS[split]  + METRIC_COLS[split]
    if evaluation_csv: cols = cols + ['BARTScore', 'BERTScore', 'entailment']
    # handle ill-parsed lists
    if split == 'training': 
        cnames_quantiles = [o for o in df.columns if 'quantiles' in o]
        def parse_l(l): return [round(o, 3) for o in literal_eval(l)]
        for cname in cnames_quantiles: 
            df.loc[:,cname] = df[cname].apply(parse_l)
    return df[cols].round(3)

def get_interesting_idx(df, n):
    def get_idx_with_top_column_values(cname, n=5, ascending=False):
        return df[['idx',cname]].\
            drop_duplicates().\
            sort_values(cname, ascending=ascending)\
            ['idx'][0:n].values.tolist()
    
    def sample_idx_with_label_flips(n=5): 
        df1 = df[['idx','label_flip']].query("label_flip!=0")
        if len(df1) == 0 : print("No label flips detected"); return None
        else: return df1.drop_duplicates()['idx'].sample(n).values.tolist()
    
    idx_d = dict(
        random = df.idx.drop_duplicates().sample(n).tolist(),
    )
    return idx_d

def print_interesting_text_stats_training_step(df, n, split): 
    def print_stats(df, idx_d, key, i):
        print("\n###############\n")
        print(key, i+1, "\n")
        if idx_d[key] is None: return
        idx = idx_d[key][i]
        # Setup 
        df1 = df.query('idx==@idx')
        orig = pd.unique(df1['orig'])[0]
        print ("Idx:",idx)
        print("Original:", orig)
        print("Original label", pd.unique(df1['label'])[0] )
        print("Orig truelabel probs", pd.unique(df1['orig_truelabel_probs'])[0] )

        pp_all = list(df1['pp'])
        #print("All paraphrases", pp_all)
        pp_unique = list(pd.unique(df1['pp']))
        n_pp_unique = len(pp_unique)

        # showing a "timeline" of how the paraphrases change over the epochs
        g_fields =  ["pp"] + [o for o in METRIC_COLS[split] if 'quantiles' not in o]

        #g_fields = ["pp","vm_scores"]
        g = df1.groupby(g_fields).agg({'epoch' : lambda x: list(x)})
        g = g.sort_values(by='epoch', key = lambda col: col.map(lambda x: np.min(x)))
        print("Unique paraphrases:", n_pp_unique)
        print("How the paraphrases change:")
        display_all(g)

        best_pps = df1.sort_values('vm_scores_mean', ascending=False).iloc[0]
        print("Best Paraphrase")
        display_all(best_pps.to_frame().T)

    idx_d = get_interesting_idx(df, n)
    for key in idx_d.keys():
        for i in range(n): 
            print_stats(df, idx_d, key,i)


def show_random_examples_for_eval_set(df, split, n, epoch=0): 
    split = "test" if "test" in split else split
    assert epoch <= df.epoch.max()
    random_idx = df.idx.drop_duplicates().sample(n).tolist()
    cols = PP_COLS[split] + METRIC_COLS[split]
    agg_d = {k:np.mean for k in METRIC_COLS[split]}
    for i, idx in enumerate(random_idx): 
        print("\n###############\n")
        print("random", i+1, "\n")
        if split == 'validation': 
            df1 = df.query("idx==@idx").query("epoch==@epoch")
        else: 
            df1 = df.query("idx==@idx")
        print("Original:", pd.unique(df1['orig'])[0])
        print("Original label", pd.unique(df1['label'])[0] )
        print("Original label probs:", pd.unique(df1['orig_truelabel_probs'])[0] )
        df2 = df1[cols]
        print("Paraphrase-level view")
        df3 = df2.groupby(['pp', 'pp_predclass']).agg({'epoch': lambda x: list(x), **agg_d})
        df3.sort_values(by ='epoch', key=lambda col: col.map(lambda x: np.min(x)), inplace=True)
        display_all(df3)
        print("Epoch-level view")
        df4 = df2.groupby('epoch').agg(agg_d)
        df4.columns = [o + "_avg" for o in df4.columns]
        display_all(df4)
    return 