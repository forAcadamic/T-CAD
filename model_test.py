from torch.utils.data import DataLoader
from src.dataset_TAD import *
import sys
from src.learner import Learner, Pathleaner
import math
import os
import torch
from src.util import *
from sklearn import metrics
import matplotlib.pyplot as plt
import argparse
plt.rcParams['font.family'] = 'STSong'


parser = argparse.ArgumentParser(description="argparse")
parser.add_argument('--data_path', default='DATA/TAD/', type=str)
parser.add_argument('--result_path', default='result/', type=str)
parser.add_argument('--SAD_model_path', default='parames/model_SAD_best_auc.pth', type=str)
parser.add_argument('--VAD_model_path', default='parames/model_VAD_best_auc.pth', type=str)
parser.add_argument('--seg', default=False, type=bool)
parser.add_argument('--gpu_id', default='0', type=str)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
SAD_model_path = args.SAD_model_path
VAD_model_path = args.VAD_model_path
data_path = args.data_path
result_path = args.result_path
seg = args.seg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

normal_test_dataset = Normal_Loader(is_train=0, path=data_path, seg=seg)
anomaly_test_dataset = Anomaly_Loader(is_train=0, path=data_path, seg=seg)

normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

SAD_model = Learner(input_dim=768, drop_p=0.6)
VAD_model = Pathleaner(input_dim=31*8)
thred_patch = 0.6

video_raw_frame_predict = {}
video_frame_predict = {}
if torch.cuda.is_available():
    SAD_model.to(device).load_state_dict(torch.load(SAD_model_path))
    VAD_model.to(device).load_state_dict(torch.load(VAD_model_path))
else:
    SAD_model.to(device).load_state_dict(torch.load(SAD_model_path, map_location=torch.device('cpu')))
    VAD_model.to(device).load_state_dict(torch.load(VAD_model_path, map_location=torch.device('cpu')))

gt_all = np.array([])
score_all = np.array([])
ano_gt_all = np.array([])
ano_score_all = np.array([])

sc_gt_all = np.array([])
sc_score_all = np.array([])
sc_ano_gt_all = np.array([])
sc_ano_score_all = np.array([])

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

SAD_model.eval()
VAD_model.eval()

with torch.no_grad():
    for i, data in enumerate(anomaly_test_loader):
        cls_anomaly, patch_anomaly, gts, frames, name = data

        inputs = cls_anomaly.view(1, -1, cls_anomaly.size(-1)).to(torch.device(device))
        score = SAD_model(inputs).squeeze()

        score1 = score.cpu().detach().numpy()

        score_list = np.zeros(frames[0])
        step = np.round(np.linspace(0, math.ceil(int(frames[0]) / 16), 33))

        for j in range(32):
            score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score1[j]

        gt_list = np.zeros(frames[0])
        for k in range(len(gts) // 2):
            s = gts[k * 2]
            e = min(gts[k * 2 + 1], frames)
            gt_list[s-1:e] = 1

        patch_cos = torch.cosine_similarity(patch_anomaly[:, 1:32, :],
                                            patch_anomaly[:, 0: 32 - 1, :], dim=3)

        patch_score = VAD_model(patch_cos.view(1, -1)).squeeze()

        # T-SAD The frame level abnormal score for each video
        video_raw_frame_predict[name[0]] = [gt_list, score_list]
        sc_gt_all = np.concatenate((sc_gt_all, gt_list), axis=0)
        sc_score_all = np.concatenate((sc_score_all, score_list), axis=0)
        sc_ano_gt_all = np.concatenate((sc_ano_gt_all, gt_list), axis=0)
        sc_ano_score_all = np.concatenate((sc_ano_score_all, score_list), axis=0)

        max_score = max(score)
        if patch_score > thred_patch:
            score_list = min_max_normalize(score_list)

        gt_all = np.concatenate((gt_all, gt_list), axis=0)
        score_all = np.concatenate((score_all, score_list), axis=0)
        ano_gt_all = np.concatenate((ano_gt_all, gt_list), axis=0)
        ano_score_all = np.concatenate((ano_score_all, score_list), axis=0)

        # T-CAD The frame level abnormal score for each video
        video_frame_predict[name[0]] = [gt_list, score_list]

    for i, data2 in enumerate(normal_test_loader):
        cls_nomaly, patch_normal, gts2, frames2, name2 = data2
        inputs2 = cls_nomaly.to(torch.device(device))
        score = SAD_model(inputs2).squeeze()
        score2 = score.cpu().detach().numpy()
        score_list2 = np.zeros(frames2[0])
        step2 = np.round(np.linspace(0, math.ceil(int(frames2[0]) / 16), 33))
        for kk in range(32):
            score_list2[int(step2[kk]) * 16:(int(step2[kk + 1])) * 16] = score2[kk]
        gt_list2 = np.zeros(frames2[0])

        patch_cos = torch.cosine_similarity(patch_normal[:, 1:32, :],
                                            patch_normal[:, 0: 32 - 1, :], dim=3)

        patch_score = VAD_model(patch_cos.view(1, -1)).squeeze()

        # T-SAD The frame level abnormal score for each video
        video_raw_frame_predict[name2[0]] = [gt_list2, score_list2]
        sc_gt_all = np.concatenate((sc_gt_all, gt_list2), axis=0)
        sc_score_all = np.concatenate((sc_score_all, score_list2), axis=0)

        max_score = max(score)
        if patch_score > thred_patch:
            score_list2 = min_max_normalize(score_list2)

        gt_all = np.concatenate((gt_all, gt_list2), axis=0)
        score_all = np.concatenate((score_all, score_list2), axis=0)

        # T-CAD The frame level abnormal score for each video
        video_frame_predict[name2[0]] = [gt_list2, score_list2]

if not os.path.exists(result_path):
    os.mkdir(result_path)

# Draw the T-SC and T-CAD AUC plots of the complete test set and the abnormal subtest set
lw = 2
sc_fpr, sc_tpr, sc_thresholds = metrics.roc_curve(sc_gt_all, sc_score_all, pos_label=1)
sc_auc = metrics.auc(sc_fpr, sc_tpr)
fpr, tpr, thresholds = metrics.roc_curve(gt_all, score_all, pos_label=1)
auc = metrics.auc(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r', label='Random' + ' (AUC = 0.5000)', alpha=1)
plt.plot(sc_fpr, sc_tpr, lw=lw, alpha=1, color='g', label='T-SAD' +' (AUC = %0.4f)' % sc_auc)
plt.plot(fpr, tpr, lw=lw, alpha=1, color='b', label='T-CAD' +' (AUC = %0.4f)' % auc)

plt.xlim([0.0, 1.0])
plt.ylim([-0.005, 1.005])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(os.path.join(result_path, 'overal_roc.pdf'))
# plt.show()
plt.close()

plt.figure()
sc_ano_fpr, sc_ano_tpr, sc_ano_thresholds = metrics.roc_curve(sc_ano_gt_all, sc_ano_score_all, pos_label=1)
sc_ano_auc = metrics.auc(sc_ano_fpr, sc_ano_tpr)
ano_fpr, ano_tpr, ano_thresholds = metrics.roc_curve(ano_gt_all, ano_score_all, pos_label=1)
ano_auc = metrics.auc(ano_fpr, ano_tpr)

plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r', label='Random' + ' (AUC = 0.5000)', alpha=1)
plt.plot(sc_ano_fpr, sc_ano_tpr, lw=lw, alpha=1, color='g', label='T-SAD' +' (AUC = %0.4f)' % sc_ano_auc)
plt.plot(ano_fpr, ano_tpr, lw=lw, alpha=1, color='b', label='T-CAD' +' (AUC = %0.4f)' % ano_auc)

plt.xlim([0.0, 1.0])
plt.ylim([-0.005, 1.005])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig(os.path.join(result_path,'subset_roc.pdf'))
# plt.show()
plt.close()

# Print various indicators
target_names = ['Normal', 'Anomaly']
print('overal_auc: ' + str(auc))
print('Anomaly Subset_auc: ' + str(ano_auc))

# Draw the anomaly detection diagram for each video
video_raw_frame_predict_path = os.path.join(result_path, 'video_raw_frame_predict')
video_frame_predict_path = os.path.join(result_path, 'video_frame_predict')

if not os.path.exists(video_raw_frame_predict_path):
    os.mkdir(video_raw_frame_predict_path)
if not os.path.exists(video_frame_predict_path):
    os.mkdir(video_frame_predict_path)

for name_item in list(video_frame_predict.keys()):
    gt_raw = video_raw_frame_predict[name_item][0]
    sc_raw = video_raw_frame_predict[name_item][1]
    plt.figure()
    plt.plot(list(range(len(gt_raw))), gt_raw, 'r')
    plt.plot(list(range(len(sc_raw))), sc_raw, 'b')
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(video_raw_frame_predict_path, name_item[:-4]+'.png'), dpi=1000, bbox_inches='tight')
    plt.close()

    gt = video_frame_predict[name_item][0]
    sc = video_frame_predict[name_item][1]
    plt.figure()
    plt.plot(list(range(len(gt))), gt, 'r')
    plt.plot(list(range(len(sc))), sc, 'b')
    plt.ylim([-0.05, 1.05])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(video_frame_predict_path, name_item[:-4] + '.png'), dpi=1000, bbox_inches='tight')
    plt.close()
