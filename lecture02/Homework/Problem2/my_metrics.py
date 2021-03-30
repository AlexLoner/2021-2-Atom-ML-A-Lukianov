import numpy as np
import functools

def analizer(func):
    
    @functools.wraps(func)
    def wrapper(y_true, y_predict, percent=None):
        # Здесь игнорируется часть с тем, чтобы в случае percent = None оценивался порог в 0.5 для всей выборки,
        # так как это само собой должно получиться для классификации на два класса, а для 
        # большего числа классов такое выражение уже некорректно. В общем случае, для k классов 
        # порог будет 1 / k. Поэтому, если percent = None, то просто берется вся выборка.
        
        percent = 100 if percent is None else percent
        assert isinstance(percent, int), "Custom value of variable 'percent' should have INT type"
        assert 1 <= percent <= 100, '"percent" should be integer from [1, 100]'

        part = int(0.01 * percent * y_true.shape[0])
        top_indexes = np.argsort(np.max(y_predict, axis=1))[-part:]
        new_y_predict = np.argmax(y_predict[top_indexes], axis=1)
        new_y_true = y_true[top_indexes]
            
        score = func(new_y_true, new_y_predict, percent)
        return score
    
    return wrapper


@analizer
def accuracy_score(y_true, y_predict, percent=None):
    return np.sum(y_true == y_predict) / y_true.shape[0]


@analizer
def precision_score(y_true, y_predict, percent=None):
    c = ps = 0
    unique = np.unique(y_true)
    if unique.shape[0] == 2:
        unique = [1]
        
    for label in unique:
        c += 1
        TP = (np.where((y_true == label) & (y_predict == label))[0]).shape[0]
        FP = (np.where((y_true != label) & (y_predict == label))[0]).shape[0]
        ps += TP / (TP + FP)
    return ps / c
    
    
@analizer
def recall_score(y_true, y_predict, percent=None):
    c = rs = 0
    unique = np.unique(y_true)
    if unique.shape[0] == 2:
        unique = [1]
    for label in unique:
        TP = (np.where((y_true == label) & (y_predict == label))[0]).shape[0]
        FN = (np.where((y_true == label) & (y_predict != label))[0]).shape[0]
        c += 1
        rs += TP / (TP + FN)
    return rs / c


@analizer
def lift_score(y_true, y_predict, percent=None):
    unique = np.unique(y_true)
    if unique.shape[0] == 2:
        unique = [1]
    c = lift = 0
    for label in unique:
        TP = (np.where((y_true == label) & (y_predict == label))[0]).shape[0]
        FP = (np.where((y_true != label) & (y_predict == label))[0]).shape[0]
        FN = (np.where((y_true == label) & (y_predict != label))[0]).shape[0]
        ps = TP / (TP + FP)
        c += 1
        lift += ps * y_true.shape[0] / (TP + FN)
    return lift / c

@analizer
def f1_score(y_true, y_predict, percent=None):
    unique = np.unique(y_true)
    if unique.shape[0] == 2:
        unique = [1]
    c = f1 = 0
    for label in unique:
        TP = (np.where((y_true == label) & (y_predict == label))[0]).shape[0]
        FP = (np.where((y_true != label) & (y_predict == label))[0]).shape[0]
        FN = (np.where((y_true == label) & (y_predict != label))[0]).shape[0]
        ps = TP / (TP + FP)
        rs = TP / (TP + FN)
        c += 1
        f1 +=  2 * ps * rs / (ps + rs)
    return f1 / c
