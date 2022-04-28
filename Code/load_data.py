#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd

"""
Loads shared test data
"""

import os
import pandas as pd

def check_dir_exists(path):
    """Checks if folder directory already exists, else makes directory.
    Args:
        path (str): folder path for saving.
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Creating {path} folder")
    else:
        print(f"Folder exists: {path}")


def upsampler(DIR, df, current_round):
    """Generates modelling data with upsampled train for current round.

    Args:
        DIR (str): directory with Github repo e.g. ./Hatemoji
        df (pandas.DataFrame): dataframe with concatenated round data
        current_round (int): current round of iterative process in [5,6,7]
    """
    # Subset to relevant portion of data
    current_df = df[df['round.base']<=current_round]

    # Set upsample schedule 
    upsample_schedule = {}
    upsample_schedule['variable'] = [1,5,10,100]
    if current_round == 5:
        upsample_schedule['fixed'] = {0:1, 1:5, 2:100, 3:1, 4:1}
    elif current_round == 6:
        upsample_schedule['fixed'] = {0:1, 1:5, 2:100, 3:1, 4:1, 5:100}
    elif current_round == 7:
        upsample_schedule['fixed'] = {0:1, 1:5, 2:100, 3:1, 4:1, 5:100, 6:1}
    elif current_round > 7:
        upsample_schedule['fixed'] = {0:1, 1:5, 2:100, 3:1, 4:1, 5:100, 6:1, 7:5}

    # Save upsampled data for model training
    for variable_multiplier in upsample_schedule['variable']:
        dev_frames = []
        test_frames = []
        train_frames = []
        # Add fixed upsamples from prior rounds
        for fixed_round, fixed_multiplier in upsample_schedule['fixed'].items():
            round_df = current_df[current_df['round.base']==fixed_round]
            # Upsample train
            base_train_frames = round_df[round_df['split']=='train']
            upsampled_train_frames = pd.concat([base_train_frames]*fixed_multiplier, axis = 0)
            assert (len(upsampled_train_frames) == len(base_train_frames)*fixed_multiplier)
            train_frames.append(upsampled_train_frames)
            # Append dev and test (not upsampled)
            dev_frames.append(round_df[round_df['split']=='dev'])
            test_frames.append(round_df[round_df['split']=='test'])
        # Add variable upsample for current round
        round_df = current_df[current_df['round.base']==current_round]
        # Upsample train
        base_train_frames = round_df[round_df['split']=='train']
        upsampled_train_frames = pd.concat([base_train_frames]*variable_multiplier, axis = 0)
        assert (len(upsampled_train_frames) == len(base_train_frames)*variable_multiplier)
        train_frames.append(upsampled_train_frames)
        # Append dev and test (not upsampled)
        dev_frames.append(round_df[round_df['split']=='dev'])
        test_frames.append(round_df[round_df['split']=='test'])

        # Create save directory
        print(f'\nCurrent round: {current_round}, Multiplier: {variable_multiplier}')
        data_dir = f'upsample{variable_multiplier}'
        full_path = f'{DIR}/Code/train_step/r{current_round}/{data_dir}'
        check_dir_exists(full_path)
        # Concatenate frames and save
        split_frames = [train_frames, dev_frames, test_frames]
        split_names = ['train', 'dev', 'test']
        for f, n in zip(split_frames, split_names):
            save_df = pd.concat(f, axis = 0, ignore_index = True)
            save_df = save_df[['id', 'text', 'label_gold']].rename(columns = {'text': 'sentence', 'label_gold':'label'})
            print(f'split: {n}, size: {len(save_df)}')
            # Shuffle train
            if n == 'train':
                save_df = save_df.sample(frac=1, random_state=123).reset_index(drop=True)
            save_df.sample.to_csv(f'{full_path}/{n}.csv', index = False)
        
def main():
    DIR, tail = os.path.split(os.getcwd())
    print(f'Current directory: {DIR}')

    ## Modelling Data ##
    # Load and clean datasets
    keep_cols = ['id', 'text', 'label_gold', 'type', 'target', 'round.base', 'round.set', 'set', 'split','matched_id','source']
    # Round 0 (The R0 data is not publicly avaliable - please email the authors for more information.)
    r0_path = "{DIR}/path_to_r0_data/r0.csv"
    r0 = pd.read_csv(r0_path)
    r0= r0.rename(columns = {'label':'label_gold', 'round':'round.set'})

    # Round 1-4
    r1_4_path = "https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv"
    r1_4 = pd.read_csv(r1_4_path, index_col = 0)
    r1_4['label'] = r1_4['label'].replace({'hate': 1, 'nothate': 0})
    r1_4= r1_4.rename(columns = {'acl.id': 'id', 'acl.id.matched':'matched_id', 'label':'label_gold', 'level':'set', 'round':'round.set'})
    r1_4['source'] = 'DynamicallyGeneratedHate'
    r1_4 = r1_4[keep_cols]

    # Round 1-5
    r5_7_dir = "{DIR}/HatemojiBuild"
    frames = []
    for split in ["train", "validation", "test"]:
        tmp = pd.read_csv(f'{r5_7_dir}/{split}.csv', index_col = 0)
        frames.append(tmp)
    r5_7 = pd.concat(frames, axis = 0, ignore_index = True)
    r5_7['id'] = r5_7.index.map(lambda x: f'hmoji_{x}')
    r5_7['matched_id'] = r5_7['matched_id'].map(lambda x: f'hmoji_{x}')
    r5_7['source'] = 'HatemojiBuild'
    r5_7 = r5_7[keep_cols]

    # Concatenate rounds
    df = pd.concat([r0, r1_4, r5_7], axis = 0, ignore_index = True)

    # Save modelling data
    for current_round in [5,6,7]:
        upsampler(DIR, df, current_round)
    
    ## Evaluation Data ##
    # Load benchmark datasets
    hatemojicheck_path = f"{DIR}/HatemojiCheck/test.csv"
    hatemojicheck = pd.read_csv(hatemoji_check_path)
    hatemojicheck['id'] = hatemojicheck['case_id'].map(lambda x: f'hatemojicheck_{x}')

    hatecheck_path = "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv"
    hatecheck = pd.read_csv(hatecheck_path, index_col = 0)
    hatecheck = hatecheck.rename(columns = {'test_case':'text'})
    hatecheck['label_gold'] = hatecheck['label_gold'].replace({'hateful': 1, 'non-hateful': 0})
    hatecheck['id'] = hatecheck['case_id'].map(lambda x: f'hatecheck_{x}')


    # Subset concatenated round datasets
    r1_7_test = df[df['split']=='test']
    r1_4_test = df[(df['split']=='test') & (df['round.base'].isin([1,2,3,4]))]
    r5_7_test = df[(df['split']=='test') & (df['round.base'].isin([5,6,7]))]

    # Define list of test sets
    test_sets = {'r1_7':r1_7_test, 'r1_4': r1_4_test, 'r5_7': r5_7_test, 'hatemojicheck':hatemojicheck, 'hatecheck': hatecheck}

    for  k,v in test_sets.items():
        # Create save directory
        print(f'\nTest set: {k}, size: {len(v)}')
        data_dir = k
        full_path = f'{DIR}/Code/eval_step/shared_test_sets/{data_dir}'
        check_dir_exists(full_path)
        save_df = v[['id', 'text', 'label_gold']].rename(columns = {'text': 'sentence', 'label_gold':'label'})
        save_df.to_csv(f'{full_path}/test.csv', index = False)
        
if __name__ == '__main__':
    main()


