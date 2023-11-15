
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation
import yaml
from utils import plot_verb_chart
default_EPIC_path = '../val_data'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--EPIC_path', type=str, help='Path to the EPIC folder containing the JPEGImages, Annotations, '
                                                    'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_EPIC_path)
parser.add_argument('--yaml_root', type=str, 
                    required=False, default=f'{default_EPIC_path}/EPIC100_state_positive_val.yaml')
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument('--results_path', type=str, help='Path to the folder containing the sequences folders',
                    default='/home/venom/projects/XMem_evaluation/XMem_output/Sep04_09.49.56_test_0904_not_freeze_epic_25000') 
parser.add_argument('--sequence_type', type=str, help='compute for all images or only for the second half images',
                    default='all', choices=['all', 'second_half']) 
args, _ = parser.parse_known_args()
csv_name_global = f'{args.sequence_type}_global_results-{args.set}.csv'
csv_name_per_sequence = f'{args.sequence_type}_per-sequence_results-{args.set}.csv'


csv_name_global_path = os.path.join(args.results_path, csv_name_global)

csv_name_per_sequence_path = os.path.join(args.results_path, csv_name_per_sequence)


if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
    print('Using precomputed results...')
    table_g = pd.read_csv(csv_name_global_path)
    table_seq = pd.read_csv(csv_name_per_sequence_path)
else:
    print(f'Evaluating sequences for the {args.task} task...')
    
    dataset_eval = DAVISEvaluation(davis_root=args.EPIC_path, yaml_root=args.yaml_root, task=args.task, gt_set=args.set, sequences=args.sequence_type)
    metrics_res = dataset_eval.evaluate(args.results_path)
    J, F, B = metrics_res['J'], metrics_res['F'], metrics_res['B']

    
    g_measures = ['J&F-Mean', 'Blob-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(B['M']), np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                        np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")
    print(f'Global results saved in {csv_name_global_path}')

    
    with open(args.yaml_root, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    seq_names = list(J['M_per_object'].keys())
    
    plot_verb_chart(verb_class_csv_path='./EPIC_100_verb_classes.csv', 
                yaml_path='../val_data/EPIC100_state_positive_val.yaml', output_path=args.results_path, J_dict=J)
    
    narrations = []
    for name in seq_names:
        name = '_'.join(name.split('_')[:-1])
        narration = yaml_data[name]['narration']
        narrations.append(narration)
    seq_measures = ['narration', 'Sequence', 'J-Mean', 'F-Mean', 'Blob-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    B_per_object = [B['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, narrations, J_per_object, F_per_object, B_per_object)), columns=seq_measures)

    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print(f'Per-sequence results saved in {csv_name_per_sequence_path}')


sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
