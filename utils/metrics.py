def calculate_f1(correct_num, pred_num, y_true_num, verbose=False):
    if correct_num == 0 or pred_num == 0 or y_true_num == 0:
        precise = 0
        recall = 0
        f1 = 0
    else:
        precise = correct_num / pred_num
        recall = correct_num / y_true_num
        f1 = 2 * precise * recall / (precise + recall)
    if verbose:
        return precise, recall, f1
    else:
        return f1


def accuracy(correct_num, y_true_num):
    return correct_num / y_true_num



