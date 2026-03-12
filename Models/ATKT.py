import os
import os.path
import math
import gc
import argparse
import numpy as np
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from Utils.utils_ATKT import KT_backbone
from Utils.utils_ATKT import DATA
from sklearn.metrics import roc_auc_score
from Utils.utils_ATKT import KTLoss, _l2_normalize_adv
from Utils.utils_ATKT import EarlyStopping
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_ATKT(train_data, valid_path:str,lr:float, gamma:float, lr_decay:int, hidden_emb_dim:int, skill_emb_dim:int, answer_emb_dim:int, beta:float, epsilon:float, n_skill:int, seqlen:int, max_iter:int, batch_size:int):
    # model
    net = KT_backbone(skill_emb_dim, answer_emb_dim, hidden_emb_dim, n_skill)
    net=net.to(device)
        
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=gamma)
        
    # loss Function 
    kt_loss = KTLoss()

    dat = DATA(n_question=n_skill, seqlen=seqlen, separate_char=',')
    print("train_data loading")
    train_skill_data, train_answer_data = dat.load_data(train_data)
    print("val_data data loading")
    val_skill_data, val_answer_data = dat.load_data([valid_path])
    print(f"Train skill data shape: {train_skill_data.shape}")
    print(f"Train answer data shape: {train_answer_data.shape}")
    print(f"Validation skill data shape: {val_skill_data.shape}")
    print(f"Validation answer data shape: {val_answer_data.shape}")

    print(f"Number of unique skills in train data: {len(np.unique(train_skill_data))}")
    print(f"Max skill ID in train data: {np.max(train_skill_data)}")
    print(f"n_skill parameter: {n_skill}")
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(net.state_dict())
        # train and validation
    for epoch in range(max_iter):
        print(f"Epoch {epoch+1}/{max_iter}")
        
        shuffled_ind = np.arange(train_skill_data.shape[0])
        np.random.shuffle(shuffled_ind)
        train_skill_data = train_skill_data[shuffled_ind, :]
        train_answer_data = train_answer_data[shuffled_ind, :]
        
        net.train()
        
        y_pred_train_list = []
        y_true_train_list = []
        train_N = int(math.ceil(len(train_skill_data) / batch_size))
        print(f"Number of batches: {train_N}")

        for idx in range(train_N):
            print(f"Processing batch {idx+1}/{train_N}")
            optimizer.zero_grad()
            train_batch_skill = train_skill_data[idx*batch_size:(idx+1)*batch_size]
            train_batch_answer = train_answer_data[idx*batch_size:(idx+1)*batch_size]
            print(f"Batch skill shape: {train_batch_skill.shape}")
            print(f"Batch answer shape: {train_batch_answer.shape}")
            

            skill = torch.LongTensor(train_batch_skill)

            answer = torch.LongTensor(train_batch_answer)
            skill = torch.where(skill==-1, torch.tensor([n_skill]), skill)
            answer = torch.where(answer==-1, torch.tensor([2]), answer)
            skill, answer = skill.to(device), answer.to(device)

            # print(f"TORCH UNIQUE SKILLS: {torch.unique(skill)}")

            pred_res, features = net(skill, answer)
            loss, y_pred, y_true = kt_loss(pred_res, answer)
        
            features_grad = grad(loss, features, retain_graph=True)
            p_adv = torch.FloatTensor(epsilon * _l2_normalize_adv(features_grad[0].data))
            p_adv = Variable(p_adv).to(device)
            pred_res, features = net(skill, answer, p_adv)
            adv_loss, _ , _ = kt_loss(pred_res, answer)

            total_loss = loss + beta*adv_loss
            total_loss.backward()
            optimizer.step()

            y_pred_train_list.append(y_pred.cpu().detach().numpy())
            y_true_train_list.append(y_true.cpu().detach().numpy())
    
        scheduler.step()  
        
        if y_pred_train_list:
            all_y_pred_train = np.concatenate(y_pred_train_list, axis=0)
            all_y_true_train = np.concatenate(y_true_train_list, axis=0)
        else:
            print("Warning: No predictions were made in this epoch")
            continue  # Skip to the next epoch


        auc_train = roc_auc_score(all_y_true_train, all_y_pred_train)
        print('train epoch: ', (epoch+1), 'train auc: ', auc_train)

        val_total_loss = []
        y_true_val_list = []
        y_pred_val_list = []
        val_N = int(math.ceil(len(val_skill_data) / batch_size))
        net.eval()
        
        with torch.no_grad():
            for idx in range(val_N):
                val_batch_skill   = val_skill_data[idx*batch_size:(idx+1)*batch_size]
                val_batch_answer  = val_answer_data[idx*batch_size:(idx+1)*batch_size]
                    
                skill=torch.LongTensor(val_batch_skill)
                answer=torch.LongTensor(val_batch_answer)
                skill = torch.where(skill==-1, torch.tensor([n_skill]), skill)
                answer = torch.where(answer==-1, torch.tensor([2]), answer)
                skill,answer=skill.to(device),answer.to(device)
                    
                pred_res, features = net(skill, answer)
                loss, y_pred, y_true = kt_loss(pred_res, answer)
                    
                val_total_loss.append(loss.item())
                y_pred_val_list.append(y_pred.cpu().detach().numpy())
                y_true_val_list.append(y_true.cpu().detach().numpy())
                    
            all_y_pred_val = np.concatenate(y_pred_val_list, axis=0)
            all_y_true_val = np.concatenate(y_true_val_list, axis=0)
           
            auc_val = roc_auc_score(all_y_true_val, all_y_pred_val)
            avg_val_loss = np.mean(val_total_loss)
            print('val epoch: ', (epoch+1), 'val loss: ', avg_val_loss, 'val auc: ', auc_val)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = copy.deepcopy(net.state_dict())
            
            early_stopping(avg_val_loss, net)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            del auc_train
            del auc_val
            gc.collect()
            torch.cuda.empty_cache()
    # restore best weights
    net.load_state_dict(best_state_dict)
    return net


def train_predict_ATKT(train_data, valid_path:str, test_path:str,lr:float, gamma:float, lr_decay:int, hidden_emb_dim:int, skill_emb_dim:int, answer_emb_dim:int, beta:float, epsilon:float, n_skill:int, seqlen:int, max_iter:int, batch_size:int):
    dat = DATA(n_question=n_skill, seqlen=seqlen, separate_char=',')
    test_skill_data,test_answer_data = dat.load_data(path_list=[test_path])

    model = train_ATKT(train_data=train_data,valid_path=valid_path,lr=lr,gamma=gamma,lr_decay=lr_decay,hidden_emb_dim=hidden_emb_dim,skill_emb_dim=skill_emb_dim,answer_emb_dim=answer_emb_dim,beta=beta,epsilon=epsilon,n_skill=n_skill,seqlen=seqlen,max_iter=max_iter,batch_size=batch_size)
    
    model.eval()
    y_true_test_list = []
    y_pred_test_list = []
    test_N = int(math.ceil(len(test_skill_data) / batch_size))

    kt_loss = KTLoss()

    with torch.no_grad():
        for idx in range(test_N):
            test_batch_skill   = test_skill_data[idx*batch_size:(idx+1)*batch_size]
            test_batch_answer  = test_answer_data[idx*batch_size:(idx+1)*batch_size]
                
            skill=torch.LongTensor(test_batch_skill)
            answer=torch.LongTensor(test_batch_answer)
            skill = torch.where(skill==-1, torch.tensor([n_skill]), skill)
            answer = torch.where(answer==-1, torch.tensor([2]), answer)
            skill,answer=skill.to(device),answer.to(device)
                
            pred_res, features = model(skill, answer)
            loss, y_pred, y_true = kt_loss(pred_res, answer)
                
            y_true_test_list.append(y_true.cpu().detach().numpy())
            y_pred_test_list.append(y_pred.cpu().detach().numpy())
                
        all_y_true_test = np.concatenate(y_true_test_list, 0)
        all_y_pred_test = np.concatenate(y_pred_test_list, 0)

        return all_y_pred_test,all_y_true_test


                   
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Script to train KT')
    parser.add_argument('--max_iter', type=int, default=100, help='number of iterations')
    parser.add_argument('--seed', type=int, default=224, help='default seed')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
    parser.add_argument('--lr-decay', type=int, default=50, help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--hidden-emb-dim', type=int, default=80, help='Dimension of concept embedding.')
    parser.add_argument('--skill-emb-dim', type=int, default=256)
    parser.add_argument('--answer-emb-dim', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=110)
    parser.add_argument('--n_skill', type=int, default=110)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--train_path', type=str, default="dataset/assist2009_ATKT/assist2009_ATKT_train1.csv")
    parser.add_argument('--valid_path', type=str, default="dataset/assist2009_ATKT/assist2009_ATKT_valid1.csv")
    parser.add_argument('--test_path', type=str, default="dataset/assist2009_ATKT/assist2009_ATKT_test1.csv")
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--epsilon', type=float, default=10)
    params = parser.parse_args()



    train_predict_ATKT(lr=params.lr,gamma=params.gamma,lr_decay=params.lr_decay,hidden_emb_dim=params.hidden_emb_dim,skill_emb_dim=params.skill_emb_dim,answer_emb_dim=params.answer_emb_dim,beta=params.beta,epsilon=params.epsilon,n_skill=params.n_skill,seqlen=params.seq_len,max_iter=params.max_iter,batch_size=params.batch_size,train_path=params.train_path,valid_path=params.valid_path,test_path=params.test_path)


    
