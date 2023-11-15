import pandas as pd
import yaml
import matplotlib.pyplot as plt
def plot_verb_chart(verb_class_csv_path, yaml_path, output_path=None, J_dict=None):
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    verb_class_df = pd.read_csv(verb_class_csv_path).drop(columns=['id', 'category'])
    verb_class = {}
    for index, row in verb_class_df.iterrows():        
        verb_class.update({row['key']: row['instances'].strip('"').strip('[').strip(']').replace("'",'').replace(' ','').split(',')})
    seq_names = list(J_dict['M_per_object'].keys())
    eff_verb_J = {}
    for k in seq_names:
        yaml_k = '_'.join(k.split('_')[:-1])
        verb_name = yaml_data[yaml_k]['verb']
        if verb_name in list(eff_verb_J.keys()):
            eff_verb_J[verb_name]['count'] += 1
            eff_verb_J[verb_name]['J'] += J_dict['M_per_object'][k]
        else:
            eff_verb_J.update({verb_name: {'count': 1, 'J': J_dict['M_per_object'][k]}})
    caption_list = list(eff_verb_J.keys())
    J_list = []
    cnt_list = []
    for k in caption_list:
        J = eff_verb_J[k]['J']/eff_verb_J[k]['count']
        J_list.append(J)
        cnt_list.append(eff_verb_J[k]['count'])
    
    fig, ax1 = plt.subplots(figsize=(30,15))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    line1 = ax1.bar(range(len(J_list)), J_list, color='green', width=0.4, tick_label=caption_list, label='J_metric')
    ax2 = ax1.twinx()
    line2 = ax2.bar([x+0.4 for x in list(range(len(J_list)))], cnt_list, color='tab:blue', width=0.4, tick_label=caption_list, label='count')
    ax2.tick_params(axis='y', labelsize=20)
    ax1.set_ylim((0, 1))
    
    
    
    
    
    
    plt.legend(handles=[line1, line2],loc='upper right',fontsize=20)
    
    plt.savefig(f'{output_path}/verb_chart.png')

if __name__ == "__main__":
    plot_verb_chart('./EPIC_100_verb_classes.csv',
                '../val_data/EPIC100_state_positive_val.yaml')