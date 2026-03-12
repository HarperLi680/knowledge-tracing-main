import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import os
from collections import defaultdict
from scipy.optimize import minimize
import csv
import numpy as np

#Calculates b4_correct and b4_incorrect coloumnns for dataset
def process_csv_files(input_folder, output_folder):
  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            user_skill_counts = defaultdict(lambda: {'correct': 0, 'incorrect': 0})   
        
            with open(input_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                
                header = next(reader)
                writer.writerow(header + ['bf_correct', 'bf_incorrect'])
                
                for row in reader:
                    user, skill, item, correct = row
                    user_skill = (user, skill)
                    
                    bf_correct = user_skill_counts[user_skill]['correct']
                    bf_incorrect = user_skill_counts[user_skill]['incorrect']
                    
                    writer.writerow(row + [bf_correct, bf_incorrect])
                    
                    if correct == '1':
                        user_skill_counts[user_skill]['correct'] += 1
                    else:
                        user_skill_counts[user_skill]['incorrect'] += 1

    print(f"Processing complete. Updated files are in the '{output_folder}' folder.")

def pfa_train(train_data):
    df_all = pd.concat([pd.read_csv(file) for file in train_data], ignore_index=True)

    print("Combined dataset shape:", df_all.shape)

    unique_skills = df_all['skill'].unique()
    all_results = []

    gamma_min, gamma_max = 0, 10
    rho_min, rho_max = -10, 10
    beta_min, beta_max = -10, 10

    for skill in unique_skills:
        x_vars = np.zeros(3)  # gamma, rho, beta for the current skill

        bounds = [(gamma_min, gamma_max), (rho_min, rho_max), (beta_min, beta_max)]

        # filter the dataset for the current skill
        skill_df = df_all[df_all['skill'] == skill]
        skill_response_array = skill_df[['correct', 'b4_correct', 'b4_incorrect']].to_numpy()

        if skill_response_array.shape[0] == 0:
            continue
        
        #TODO Check this new implementation is appropriate changing optimization function
        #res = minimize(pfa, x_vars, args=(skill_response_array,), method='L-BFGS-B', bounds=bounds, options={'disp': True})
        res = minimize(pfa_nll, x_vars, args=(skill_response_array,), method='L-BFGS-B', bounds=bounds, options={'disp': True})

        all_results.append([skill, res.x[0], res.x[1], res.x[2]])

        print(f"Skill {skill} optimized parameters: {res.x}")

    results_df = pd.DataFrame(all_results, columns=['skill', 'gamma', 'rho', 'beta'])
    
    return results_df

def pfa(x_vars, response_array):
    gamma, rho, beta = x_vars
    correctness = response_array[:, 0]
    B4_correct = response_array[:, 1]
    B4_incorrect = response_array[:, 2]
    skill_binary = np.ones_like(B4_correct)

    m_param = gamma * B4_correct + rho * B4_incorrect + beta * skill_binary
    p_m = 1 / (1 + np.exp(-m_param))

    sr = (p_m - correctness) ** 2

    return np.sum(sr)

def pfa_nll(x_vars, response_array):
    gamma, rho, beta = x_vars
    correctness = response_array[:, 0]
    b4_correct = response_array[:, 1]
    b4_incorrect = response_array[:, 2]

    m_param = gamma * b4_correct + rho * b4_incorrect + beta
    p_m = 1 / (1 + np.exp(-m_param))

    eps = 1e-12
    p_m = np.clip(p_m, eps, 1 - eps)

    nll = -(correctness * np.log(p_m) + (1 - correctness) * np.log(1 - p_m))
    return np.sum(nll)

def pfa_predict(params_df, test_data_path):
    result_df = pd.read_csv(test_data_path)
    
    predictions = np.zeros(len(result_df))
    
    grouped = result_df.groupby('skill')
    
    for skill, group in grouped:
        skill_params = params_df[params_df['skill'] == skill]
        
        if skill_params.empty:
            print(f"Warning: No parameters found for skill {skill}. Skipping.")
            continue
        
        gamma = skill_params['gamma'].values[0]
        rho = skill_params['rho'].values[0]
        beta = skill_params['beta'].values[0]
        
        m_param = (gamma * group['b4_correct'] + 
                   rho * group['b4_incorrect'] + 
                   beta)

        p_m = 1 / (1 + np.exp(-m_param))
        
        predictions[group.index] = p_m
    
    true_values = result_df['correct'].values
    
    return predictions, true_values


def train_predict_PFA(train_data, test_data):
    params_df = pfa_train(train_data)

    predictions, true_values = pfa_predict(params_df, test_data)

    return predictions, true_values


if __name__ == "__main__":
    process_csv_files('/Users/abhinavshukla/Downloads/Knowledge-Tracing-Models-Abhinav-s-Code/tabular_data','/Users/abhinavshukla/Downloads/Knowledge-Tracing-Models-Abhinav-s-Code/pfa_tabular_data')


