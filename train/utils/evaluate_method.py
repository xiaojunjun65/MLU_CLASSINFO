import numpy as np

def compute_acc(predict_res, truth):
    predict_res = np.vstack(predict_res)
    truth = np.hstack(truth)
    res = np.argmax(predict_res, axis=1)
    pred_t = 0
    missed_ok = 0
    missed_qipao = 0 
    for p, t in zip(res, truth):
        if p == t:
            pred_t += 1
        else:
            if t == 1:
                missed_ok += 1
            else:
                missed_qipao += 1
    return pred_t / len(truth)