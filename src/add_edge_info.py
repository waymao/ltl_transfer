########### add_edge_info.py ###########
# Used to add traversed edge info to the policy status json file
# used to convert legacy run results
# not to be used in the main pipeline
############################################
import json
import gzip
import argparse
import os
from tqdm import tqdm

from ltl.dfa import DFA
from ts_utils.ts_policy_bank import list_to_tuple

path = "/home/wyc/data/shared/ltl-transfer-ts/results/miniworld_simp_no_vis_minecraft/mixed_p1.0/lpopl_dsac/map13/0/alpha=0.03"

ltl_id = int(os.environ["LTL_ID"])

with open(os.path.join(path, "ltl_list.json"), "r") as f:
    ltl_list = json.load(f)
ltl = ltl_list[ltl_id][0]
results = {}
results_new = {}
ltl = list_to_tuple(ltl)
with gzip.open(os.path.join(path, "classifier", f"policy{ltl_id}_status.json.gz"), "rt", encoding="UTF-8") as f:
    results = json.load(f)
    for loc, value in results.items():
        dfa = DFA(ltl)
        prev_state = dfa.state
        dfa.progress(value['true_proposition'])
        results_new[loc] = value
        results_new[loc]['edge'] = dfa.nodelist[prev_state][dfa.state]
with gzip.open(os.path.join(path, "classifier", f"policy{ltl_id}_status.json.gz"), "wt", encoding="UTF-8") as f:
    json.dump(results_new, f)

