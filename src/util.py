from sklearn import metrics
def get_false_alarm_rate(ground_true, pred):
    pred = ((pred > 0.5) + 0).tolist()
    fa_rate = sum(pred > ground_true)/len(pred)
    return fa_rate

def get_f1_score(ground_true, pred):
    pred = ((pred > 0.5) + 0).tolist()
    return metrics.f1_score(y_true=ground_true, pos_label=1, y_pred=pred, average='weighted')

def get_report(ground_true, pred, target_names=['0', '1']):
    pred = ((pred > 0.5) + 0).tolist()
    return metrics.classification_report(ground_true, pred, digits=4, target_names=target_names)

def min_max_norm(input):
    '''input: torch.tensor'''
    norm = input - input.min()
    norm = norm / norm.max()
    return norm.cpu().detach().numpy()
