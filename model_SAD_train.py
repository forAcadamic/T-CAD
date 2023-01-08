import torch
from torch.utils.data import DataLoader
from src.learner import Learner
from src.loss import *
from src.dataset_TAD import *
import os
from sklearn import metrics
from src.util import *
import time
import math
import copy
import argparse

parser = argparse.ArgumentParser(description="argparse")
parser.add_argument('--data_path', default='DATA/TAD/', type=str)
parser.add_argument('--epoches', default=500, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--save_path', default='parames', type=str)
parser.add_argument('--seg', default=False, type=bool)
parser.add_argument('--seed', default=1029, type=int)
parser.add_argument('--gpu_id', default='0', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
epoches = args.epoches
seed_id = args.seed
data_path = args.data_path
batch_size = args.batch_size
seg = args.seg
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

seed_torch(seed=seed_id)
print('seed_id: ', seed_id)

# split the data

normal_train_dataset = Normal_Loader(is_train=1, path=data_path, seg=seg)
normal_test_dataset = Normal_Loader(is_train=0, path=data_path, seg=seg)

anomaly_train_dataset = Anomaly_Loader(is_train=1, path=data_path, seg=seg)
anomaly_test_dataset = Anomaly_Loader(is_train=0, path=data_path, seg=seg)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

print('length of trains: %3d + %3d'%(len(normal_train_loader), len(anomaly_train_loader)))
print('length of vals: %3d + %3d'%(len(normal_test_loader), len(anomaly_test_loader)))

# design the whole model

model_SAD = Learner(input_dim=768, drop_p=0.6).to(device)
optimizer = torch.optim.SGD(model_SAD.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=150,
                                                         eta_min=0,
                                                         last_epoch=-1)
criterion = MIL
Rcriterion = torch.nn.MarginRankingLoss(margin=1.0, reduction='mean')
Rcriterion = Rcriterion.to(device)

all_loss = []
all_auc = []

nameTime = time.strftime('%Y-%m-%d-%H-%M-%S')
if not os.path.exists(save_path):
    os.mkdir(save_path)

best_auc = 0
ith_auc = 0
best_auc_model = copy.deepcopy(model_SAD.state_dict())

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model_SAD.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        cls_normal, patch_normal = normal_inputs
        cls_anomaly, patch_anomaly = anomaly_inputs
        inputs = torch.cat([cls_anomaly, cls_normal], dim=1).to(device)
        outputs = model_SAD(inputs).squeeze()

        semantic_margin_ano = torch.max(outputs[:, 0:32], 1)[0] - torch.min(outputs[:, 0:32], 1)[0]
        semantic_margin_nor = torch.max(outputs[:, 32:32*2], 1)[0] - torch.min(outputs[:, 32:32*2], 1)[0]
        loss_c = Rcriterion(semantic_margin_ano, semantic_margin_nor, torch.tensor([1.]*batch_size).to(device))
        loss_sparsity = torch.mean(torch.sum(outputs[:, :32], 1))
        loss_smooth = torch.mean(torch.sum((outputs[:, :32 - 1] - outputs[:, 1:32]) ** 2, 1))
        loss = loss_c+loss_sparsity*0.00008 + loss_smooth * 0.00008

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('loss = {:.5f}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()
    all_loss.append(train_loss/len(normal_train_loader))

def test_abnormal(epoch, best_auc):
    model_SAD.eval()
    num_videos = 0
    gt_all = np.array([])
    score_all = np.array([])

    with torch.no_grad():
        for i, data in enumerate(anomaly_test_loader):
            num_videos += 1
            cls_anomaly, _, gts, frames = data

            inputs = cls_anomaly.view(-1, cls_anomaly.size(-1)).to(torch.device(device))
            score = model_SAD(inputs)
            score = score.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, math.ceil(frames[0]/16), 33))

            for j in range(32):
                score_list[int(step[j])*16:(int(step[j+1]))*16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames)
                gt_list[s-1:e] = 1

            gt_all = np.concatenate((gt_all, gt_list), axis=0)
            score_all = np.concatenate((score_all, score_list), axis=0)

        for i, data2 in enumerate(normal_test_loader):
            num_videos += 1
            cls_nomaly, _, gts2, frames2 = data2
            inputs2 = cls_nomaly.view(-1, cls_nomaly.size(-1)).to(torch.device(device))
            score2 = model_SAD(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, math.ceil(frames2[0]/16), 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk+1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])

            gt_all = np.concatenate((gt_all, gt_list2), axis=0)
            score_all = np.concatenate((score_all, score_list2), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(gt_all, score_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        fa = get_false_alarm_rate(gt_all, score_all)
        f1 = get_f1_score(gt_all, score_all)

        all_auc.append(auc)

        print('auc = {:.5f}'.format(auc))

        if auc > best_auc:
            best_auc = auc
            best_auc_model = copy.deepcopy(model_SAD.state_dict())

            np.save(save_path + '/gt_all_best_auc.npy', gt_all)
            np.save(save_path + '/score_all_best_auc.npy', score_all)

        return best_auc

for epoch in range(epoches):
    train(epoch)
    print('best_auc: ', best_auc)

    model_SAD.eval()
    num_videos = 0
    gt_all = np.array([])
    score_all = np.array([])

    with torch.no_grad():
        for i, data in enumerate(anomaly_test_loader):
            num_videos += 1
            cls_anomaly, _, gts, frames, name = data

            inputs = cls_anomaly.view(-1, cls_anomaly.size(-1)).to(torch.device(device))
            score = model_SAD(inputs)
            score = score.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, math.ceil(frames[0] / 16), 33))

            for j in range(32):
                score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = gts[k * 2]
                e = min(gts[k * 2 + 1], frames)
                gt_list[s - 1:e] = 1

            gt_all = np.concatenate((gt_all, gt_list), axis=0)
            score_all = np.concatenate((score_all, score_list), axis=0)

        for i, data2 in enumerate(normal_test_loader):
            num_videos += 1
            cls_nomaly, _, gts2, frames2, name2 = data2
            inputs2 = cls_nomaly.view(-1, cls_nomaly.size(-1)).to(torch.device(device))
            score2 = model_SAD(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, math.ceil(frames2[0] / 16), 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
            gt_list2 = np.zeros(frames2[0])

            gt_all = np.concatenate((gt_all, gt_list2), axis=0)
            score_all = np.concatenate((score_all, score_list2), axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(gt_all, score_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        fa = get_false_alarm_rate(gt_all, score_all)
        f1 = get_f1_score(gt_all, score_all)

        all_auc.append(auc)

        print('auc = {:.5f}'.format(auc))

        if auc > best_auc:
            best_auc = auc
            best_auc_model = copy.deepcopy(model_SAD.state_dict())
            np.save(save_path + '/gt_all_best_auc.npy', gt_all)
            np.save(save_path + '/score_all_best_auc.npy', score_all)

np.save(save_path + '/all_loss.npy', all_loss)
np.save(save_path + '/all_auc.npy', all_auc)

best_auc = max(all_auc)
ith_auc = all_auc.index(best_auc)

model_SAD.load_state_dict(best_auc_model)
torch.save(model_SAD.state_dict(), save_path + '/model_SAD_best_auc.pth')

with open(save_path + '/best_performance.txt', 'w') as f:
    print('random_id: [%4d]' % seed_id, file=f)
    print('best_auc:[%4d] %.4f ' % (ith_auc, best_auc), file=f)