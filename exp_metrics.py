import math

def calc_recall(TPs, TNs, FPs, FNs):
    try:
        return TPs / (TPs + FNs)
    except ZeroDivisionError:
        return 0
    
def calc_precision(TPs, TNs, FPs, FNs):
    try:
        return TPs / (TPs + FPs)
    except ZeroDivisionError:
        return 0

def calc_accuracy(TPs, TNs, FPs, FNs):
    try:
        return (TPs + TNs) / (TPs + TNs + FPs + FNs)
    except ZeroDivisionError:
        return 0

def calc_f1(TPs, TNs, FPs, FNs):
    precision = calc_precision(TPs, TNs, FPs, FNs)
    recall = calc_recall(TPs, TNs, FPs, FNs)
    try:
        return 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0

def calc_mcc(TPs, TNs, FPs, FNs):
    try:
        return (TPs * TNs - FPs * FNs) / math.sqrt((TPs + FPs) * (TPs + FNs) * (TNs + FPs) * (TNs + FNs))
    except ZeroDivisionError:
        return 0

def calc_metrics(TPs, TNs, FPs, FNs):
    return {
        'recall': calc_recall(TPs, TNs, FPs, FNs),
        'precision': calc_precision(TPs, TNs, FPs, FNs),
        'accuracy': calc_accuracy(TPs, TNs, FPs, FNs),
        'f1': calc_f1(TPs, TNs, FPs, FNs),
        'mcc': calc_mcc(TPs, TNs, FPs, FNs)
    }

