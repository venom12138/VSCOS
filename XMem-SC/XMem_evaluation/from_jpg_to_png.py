import os

DIR = '/home/venom/projects/XMem_evaluation/XMem_output/Sep02_10.45.20_test_0902_nframes_3_epic_25000'

def iteratively_rename_pic(dir_path):
    print(dir_path)
    for dir in os.listdir(dir_path):
        if os.path.isdir(f'{dir_path}/{dir}'):
            if (not dir.startswith('P')) and dir != 'anno_masks':
                continue
            else:
                iteratively_rename_pic(f'{dir_path}/{dir}')
        else:
            old_name = f'{dir_path}/{dir}'
            new_name = f'{dir_path}/{dir}'.replace('jpg', 'png')
            os.rename(old_name, new_name)

iteratively_rename_pic(DIR)