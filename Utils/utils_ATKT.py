# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.n_question = n_question
        self.seqlen = seqlen

    def load_data(self, path_list):
        skill_data = []
        answer_data = []

        for path in path_list:
            print(f"Reading file: {path}")
            with open(path, 'r') as f_data:
                lineID = 0
                S = []
                A = []
                for line in f_data:
                    # print(lineID)
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    
                    if lineID % 4 == 1:
                        # print(line)
                        S = line.split(self.separate_char)
                        if len(S[-1]) == 0:
                            S = S[:-1]
                    elif lineID % 4 == 3:
                        A = line.split(self.separate_char)
                        if len(A[-1]) == 0:
                            A = A[:-1]
                        
                        mod = 0 if len(S) % self.seqlen == 0 else (self.seqlen - len(S) % self.seqlen)
                        
                        for i in range(len(S)):
                            skill_data.append(int(S[i]))
                            answer_data.append(int(A[i]))
                        for j in range(mod):
                            skill_data.append(-1)
                            answer_data.append(-1)
                    
                    lineID += 1

        print(f"Processed skill_data length: {len(skill_data)}")
        print(f"Processed answer_data length: {len(answer_data)}")

        skill_data_array = np.array(skill_data).astype(np.int_).reshape([-1, self.seqlen])
        answer_data_array = np.array(answer_data).astype(np.int_).reshape([-1, self.seqlen])

        print(f"Final skill_data shape: {skill_data_array.shape}")
        print(f"Final answer_data shape: {answer_data_array.shape}")

        return skill_data_array, answer_data_array


class KT_backbone(nn.Module):
    def __init__(self, skill_dim, answer_dim, hidden_dim, output_dim):
        super(KT_backbone, self).__init__()
        self.skill_dim=skill_dim
        self.answer_dim=answer_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0
        
        self.attention_dim = 80
        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)
        
    def _get_next_pred(self, res, skill):
        
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output


    def forward(self, skill, answer, perturbation=None):
        
        skill_embedding=self.skill_emb(skill)
        answer_embedding=self.answer_emb(answer)
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)

        skill_answer_embedding1=skill_answer_embedding
        
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        
        out,_ = self.rnn(skill_answer_embedding)
        out=self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)
        
        return pred_res, skill_answer_embedding1

class KTLoss(nn.Module):

    def __init__(self):
        super(KTLoss, self).__init__()

    def forward(self, pred_answers, real_answers):

        real_answers = real_answers[:, 1:]
        answer_mask = torch.ne(real_answers, 2)
        
        y_pred = pred_answers[answer_mask].float()
        y_true = real_answers[answer_mask].float()
        
        loss=nn.BCELoss()(y_pred, y_true)
        return loss, y_pred, y_true


def _l2_normalize_adv(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

def calculate_n_skill(data_files):
    all_skills = set()
    for file in data_files:
        with open(file, 'r') as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if line_id % 4 == 1:
                    skills = [int(x) for x in line.split(',') if x != ""]
                    all_skills.update(skills)
    return max(all_skills) + 1


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0