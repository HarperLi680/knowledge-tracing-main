import torch
import random
import math
import os

def getdata(window_size, path, model_type, drop=False):
    """
    @param model_type: 'sakt' or 'saint'
    path can be:
    - a single file path (str)
    - a list of file paths
    """
    if isinstance(path, (str, os.PathLike)):
        paths = [path]
    else:
        paths = list(path)

    N = 0
    E = -1
    units = []
    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    bis = 0

    for one_path in paths:
        count = 0
        with open(one_path, "r") as file:
            while True:
                line = file.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                if count % 4 == 0:
                    pass

                elif count % 4 == 1:
                    tlst = [x for x in line.split(',') if x != ""]

                    for item in tlst:
                        val = int(item)
                        if val > E:
                            E = val

                    tlst_1 = tlst[:-1]
                    tlst_2 = tlst[1:]

                    if drop and len(tlst_1) > window_size:
                        tlst_1 = tlst_1[:window_size]
                        tlst_2 = tlst_2[:window_size]

                    while len(tlst_1) > window_size:
                        input_1.append([int(i) + 1 for i in tlst_1[:window_size]])
                        N += 1
                        tlst_1 = tlst_1[window_size:]
                        units.append(window_size)

                    units.append(len(tlst_1))
                    tlst_1 = [int(i) + 1 for i in tlst_1] + [0] * (window_size - len(tlst_1))
                    N += 1
                    input_1.append(tlst_1)

                    while len(tlst_2) > window_size:
                        input_3.append([int(i) + 1 for i in tlst_2[:window_size]])
                        tlst_2 = tlst_2[window_size:]

                    tlst_2 = [int(i) + 1 for i in tlst_2] + [0] * (window_size - len(tlst_2))
                    input_3.append(tlst_2)

                elif count % 4 == 3:
                    tlst = [x for x in line.split(',') if x != ""]

                    tlst_1 = tlst[:-1]
                    tlst_2 = tlst[1:]

                    if drop and len(tlst_1) > window_size:
                        tlst_1 = tlst_1[:window_size]
                        tlst_2 = tlst_2[:window_size]

                    while len(tlst_1) > window_size:
                        input_2.append([int(i) + bis for i in tlst_1[:window_size]])
                        tlst_1 = tlst_1[window_size:]

                    tlst_1 = [int(i) + bis for i in tlst_1] + [0] * (window_size - len(tlst_1))
                    input_2.append(tlst_1)

                    while len(tlst_2) > window_size:
                        input_4.append([int(i) + bis for i in tlst_2[:window_size]])
                        tlst_2 = tlst_2[window_size:]

                    tlst_2 = [int(i) + bis for i in tlst_2] + [0] * (window_size - len(tlst_2))
                    input_4.append(tlst_2)

                count += 1

    E += 1

    input_1 = torch.tensor(input_1)
    input_2 = torch.tensor(input_2)
    input_3 = torch.tensor(input_3)
    input_4 = torch.tensor(input_4)

    if model_type == 'sakt':
        input_1 = input_1 + E * input_2
        return torch.stack((input_1, input_3, input_4), 0), N, E, units
    elif model_type == 'saint':
        return torch.stack((input_1, input_2), 0), N, E, units
    else:
        raise Exception('model type error')

def dataloader(data, batch_size, shuffle: bool):
    data = data.permute(1, 0, 2)   # [N, 3, window]
    lis = list(range(len(data)))
    if shuffle:
        random.shuffle(lis)
    lis = torch.tensor(lis).long()

    ret = []
    num_batches = math.ceil(len(data) / batch_size)

    for i in range(num_batches):
        idx = lis[i * batch_size : min((i + 1) * batch_size, len(data))]
        temp = torch.index_select(data, 0, idx)   # [batch, 3, window]
        ret.append(temp.permute(1, 0, 2))         # [3, batch, window]

    return ret

class NoamOpt:
    def __init__(self, optimizer:torch.optim.Optimizer, warmup:int, dimension:int, factor=0.1):
        self.optimizer = optimizer;
        self._steps = 0;
        self._warmup = warmup;
        self._factor = factor;
        self._dimension = dimension;
        
    def step(self):
        self._steps += 1;
        rate = self._factor * (self._dimension**(-0.5) * min(self._steps**(-0.5), self._steps * self._warmup**(-1.5)));
        for x in self.optimizer.param_groups:
            x['lr'] = rate;