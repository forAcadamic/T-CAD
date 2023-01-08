import torch
from torch.utils.data import DataLoader
from src.learner import Learner, Pathleaner
from src.loss import *
from src.dataset_TAD import *
import os
from sklearn import metrics
from src.util import *
import time
import math

import argparse

parser = argparse.ArgumentParser(description="argparse")
parser.add_argument('--data_path', default='DATA/TAD/', type=str)
parser.add_argument('--epoches', default=500, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--save_path', default='parames', type=str)
parser.add_argument('--thred', default=0.6, type=int)
parser.add_argument('--seg', default=False, type=bool)
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--gpu_id', default='0', type=str)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
epoches = args.epoches
seed_id = args.seed
data_path = args.data_path
batch_size = args.batch_size
seg = args.seg
thred = args.thred
save_path = args.save_path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_id = random.randint(1,10000)
seed_torch(seed=seed_id)
print('seed_id: ', seed_id)

def min_max_normalize(scores, mode='zero'):
    max_score = max(scores)
    min_score = min(scores) if mode == 'min' else 0
    try:
        norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
    except Exception:
        print(scores)
        norm_scores = scores
        print('Min Max Error')
    return norm_scores

normal_train_dataset = Normal_Loader(is_train=1, path=data_path, seg=seg)
normal_test_dataset = Normal_Loader(is_train=0, path=data_path, seg=seg)

anomaly_train_dataset = Anomaly_Loader(is_train=1, path=data_path, seg=seg)
anomaly_test_dataset = Anomaly_Loader(is_train=0, path=data_path, seg=seg)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

model_VAD = Pathleaner(input_dim=31*8, drop_p=0.6).to(device)

optimizer = torch.optim.SGD(model_VAD.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=150,
                                                         eta_min=0,
                                                         last_epoch=-1)
criterion = torch.nn.BCELoss().to(device)

print('length of trains: %3d + %3d'%(len(normal_train_loader), len(anomaly_train_loader)))
print('length of vals: %3d + %3d'%(len(normal_test_loader), len(anomaly_test_loader)))

all_loss = []
all_auc = []

nameTime = time.strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.exists(save_path):
    os.mkdir(save_path)
best_auc = 0
ith_auc = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model_VAD.train()
    train_loss = 0
    train_right = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        cls_normal, patch_normal = normal_inputs
        cls_anomaly, patch_anomaly = anomaly_inputs
        inputs_patch = torch.cat([patch_anomaly, patch_normal], dim=1).to(device)

        patch_cos_ano = torch.cosine_similarity(inputs_patch[:, 1:32, :],
                                                inputs_patch[:, 0:32-1, :], dim=3)
        patch_cos_nor = torch.cosine_similarity(inputs_patch[:, 32+1:32*2, :],
                                                inputs_patch[:, 32: 32*2-1, :], dim=3)

        patch_ano_score = model_VAD(patch_cos_ano.view(batch_size, -1)).squeeze()
        patch_nor_score = model_VAD(patch_cos_nor.view(batch_size, -1)).squeeze()

        ano_loss = criterion(patch_ano_score, torch.ones(batch_size).to(device))
        nor_loss = criterion(patch_nor_score, torch.zeros(batch_size).to(device))

        train_right += int(sum(patch_ano_score > thred) + sum(patch_nor_score <= thred))

        loss_1 = ano_loss + nor_loss

        optimizer.zero_grad()

        loss_1.backward()
        optimizer.step()
        train_loss += loss_1.item()
    print('loss = {:.5f}'.format(train_loss/(len(normal_train_loader)+len(anomaly_train_loader))))
    print('train_acc = {:.5f}'.format(train_right/((batch_idx+1) * batch_size * 2)))
    scheduler.step()
    all_loss.append(train_loss/(len(normal_train_loader)+len(anomaly_train_loader)))

best_acc = 0

def test_abnormal(epoch,best_acc):
    model_VAD.eval()
    num_videos = 0
    gt_all = []
    score_all = []

    with torch.no_grad():
        for i, data in enumerate(anomaly_test_loader):
            num_videos += 1
            cls_anomaly, patch_anomaly, gts, frames, name = data

            patch_cos = torch.cosine_similarity(patch_anomaly[:, 1:32, :],
                                                    patch_anomaly[:, 0: 32 - 1, :], dim=3)
            patch_score = model_VAD(patch_cos.view(1, -1)).squeeze()
            if patch_score > thred:
                score_all.append(1)
            else:
                score_all.append(0)
            gt_all.append(1)

        for i, data2 in enumerate(normal_test_loader):
            num_videos += 1
            cls_nomaly, patch_normal, gts2, frames2, name2 = data2
            patch_cos = torch.cosine_similarity(patch_normal[:, 1:32, :],
                                                patch_normal[:, 0: 32 - 1, :], dim=3)
            patch_score = model_VAD(patch_cos.view(1, -1)).squeeze()
            if patch_score > thred:
                score_all.append(1)
            else:
                score_all.append(0)
            gt_all.append(0)

        acc = sum(np.array(gt_all)==np.array(score_all))/len(gt_all)
        print('accuracy = {:.5f}'.format(acc))

        if acc > best_acc:
            best_acc = acc
            torch.save(model_VAD.state_dict(), save_path + '/model_VAD_best_auc.pth')
        return best_acc

for epoch in range(epoches):
    train(epoch)
    best_acc = test_abnormal(epoch, best_acc)

    with open(save_path + '/best_performance.txt', 'w') as f:
        print('random_id: [%4d]' % seed_id, file=f)
        print('best_acc:[%.4f]' % (best_acc), file=f)
