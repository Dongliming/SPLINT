from time import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

from util import get_file_list, write_evaluation, write_recommend, get_project_name


def recommend_link(data: pd.DataFrame, X_test: pd.DataFrame, clf: RandomForestClassifier, name, y_test, n):
    """
    recommend trace links for each issue
    :param X_test: test set
    :param clf: classifier
    :param data: all features
    :return:
    """
    # extract issue_id and commit_sha from the whole data
    test_indexes = X_test.index.values
    test_data = data[data.index.isin(test_indexes)]
    issue_id_list = list(test_data['issue_id'].drop_duplicates().values.tolist())
    issue_id_index = {}
    issue_id_linked_index = {}
    index_loc = pd.DataFrame(test_indexes)
    index_loc = index_loc.rename(columns={0: 'raw_indexes'})
    p_count = 0
    for iid in issue_id_list:
        issue_id_index[iid] = test_data[test_data['issue_id'] == iid].index.values
        issue_id_linked_index[iid] = test_data[
            (test_data['issue_id'] == iid) & (test_data['is_linked'] == 1)].index.values
        p_count += len(issue_id_linked_index[iid])
    y_pro = clf.predict_proba(X_test)
    issue_id_recommend = {}
    r_count = 0
    for iid in issue_id_list:
        indexes = list(index_loc[index_loc['raw_indexes'].isin(issue_id_index[iid])].index.values)
        linked_prob = y_pro[indexes][:, 1]
        rec_indexes = [index_loc['raw_indexes'][indexes[x]] for x in np.argsort(-linked_prob)[0: min(n, len(indexes))]]
        issue_id_recommend[iid] = rec_indexes
        r_count += len(issue_id_recommend[iid])

    # calculate p, r, F, MAP, MRR
    tp_count = 0
    sum_ap = 0.0
    sum_reciprocal_rank = 0.0
    for iid in issue_id_list:
        p_indexes = issue_id_linked_index[iid]
        r_indexes = issue_id_recommend[iid]
        tp_count += len(set(p_indexes) & set(r_indexes))

        ap = calculate_AP(r_indexes, p_indexes)
        reciprocal_rank = calculate_reciprocal_rank(r_indexes, p_indexes)
        sum_ap += ap
        sum_reciprocal_rank += reciprocal_rank

    p = tp_count / r_count if r_count != 0 else 0
    r = tp_count / p_count if p_count != 0 else 0
    F = 2 * p * r / (p + r) if p + r != 0 else 0
    MAP = sum_ap / len(issue_id_list)
    MRR = sum_reciprocal_rank / len(issue_id_list)
    best_f1, best_f2, best_f1_p, best_f1_r, details, _ = prc(name, y_test, y_pro[:, 1])

    return p, r, F, MAP, MRR, best_f1, best_f1_p, best_f1_r, best_f2


def calculate_AP(recommend_list, ground_truth):
    hits = 0
    sum_precs = 0
    for i in range(len(recommend_list)):
        if recommend_list[i] in ground_truth:
            hits += 1
            sum_precs += hits / (i + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0


def calculate_reciprocal_rank(recommend_list, groud_truth):
    if len(groud_truth) == 0:
        return 0
    first = groud_truth[0]
    for i in range(len(recommend_list)):
        if first == recommend_list[i]:
            return 1 / (i + 1.0)
    return 0


def prc(fig_name, label, pred):
    precision, recall, thresholds = precision_recall_curve(label, pred)
    # print(len(precision))
    max_f1 = 0
    max_f2 = 0
    max_threshold = 0
    max_f1_p = 0
    max_f1_r = 0
    for p, r, tr in zip(precision, recall, thresholds):
        f1 = f1_score(p, r)
        f2 = f2_score(p, r)
        if f1 >= max_f1:
            max_f1 = f1
            max_f1_p = p
            max_f1_r = r
            max_threshold = tr
        if f2 >= max_f2:
            max_f2 = f2
    # viz = PrecisionRecallDisplay(
    #     precision=precision, recall=recall)
    # viz.plot()
    # plt.savefig(fig_name)
    # plt.close()
    detail = f1_details(max_threshold, label, pred)
    return round(max_f1, 3), round(max_f2, 3), max_f1_p, max_f1_r, detail, max_threshold


def f1_score(p, r):
    return 2 * (p * r) / (p + r) if p + r > 0 else 0


def f2_score(p, r):
    return 5 * (p * r) / (4 * p + r) if p + r > 0 else 0


def f1_details(threshold, label, pred):
    """Return ture positive (tp), fp, tn,fn """
    f_name = "f1_details"
    tp, fp, tn, fn = 0, 0, 0, 0
    for p, l in zip(pred, label):
        if p > threshold:
            p = 1
        else:
            p = 0
        if p == l:
            if l == 1:
                tp += 1
            else:
                tn += 1
        else:
            if l == 1:
                fp += 1
            else:
                fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def model_evaluation(clf, X_test, y_test):
    """
    evaluate the performance of the classifier
    :param clf: classifier
    :param X_test: X for test set
    :param y_test: y for test set
    :return: P,R,F1
    """
    begin = time()
    y_pre = clf.predict(X_test)
    end = time()
    test_cost_time = end - begin
    precision = metrics.precision_score(y_test, y_pre)
    recall = metrics.recall_score(y_test, y_pre)
    F1 = metrics.f1_score(y_test, y_pre)
    tp, fp, tn, fn = 0, 0, 0, 0
    for pre, true in zip(y_pre, y_test):
        if pre == true == 1:
            tp += 1
        elif pre == true == 0:
            tn += 1
        elif pre == 1 and true == 0:
            fp += 1
        elif pre == 0 and true == 1:
            fn += 1
    one_IoU = tp / (fn + tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    try:
        auc = metrics.roc_auc_score(y_test, y_pre)
    except ValueError:
        auc = 0.0
    return (precision, recall, F1, auc, one_IoU, accuracy), test_cost_time


def evaluate_IR_based(data_df: pd.DataFrame, ir_method):
    data_df['pred'] = data_df[ir_method].apply(lambda x: 1 if x >= 0.5 else 0)
    y_pred = data_df['pred'].values
    y_label = data_df['is_linked'].values
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for idx in data_df.index.values:
        pred = data_df['pred'][idx]
        label = data_df['is_linked'][idx]
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        elif pred == 0 and label == 0:
            tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    one_IoU = tp / (fn + tp + fp)
    try:
        auc = metrics.roc_auc_score(y_label, y_pred)
    except ValueError:
        auc = 0.0
    return precision, recall, f1, auc, one_IoU, accuracy


def recommend_IR_based(data_df: pd.DataFrame, ir_method, n):
    test_indexes = data_df.index.values
    issue_id_list = list(data_df['issue_id'].drop_duplicates().values.tolist())
    issue_id_index = {}
    issue_id_linked_index = {}
    index_loc = pd.DataFrame(test_indexes)
    index_loc = index_loc.rename(columns={0: 'raw_indexes'})
    p_count = 0
    for iid in issue_id_list:
        issue_id_index[iid] = data_df[data_df['issue_id'] == iid].index.values
        issue_id_linked_index[iid] = data_df[
            (data_df['issue_id'] == iid) & (data_df['is_linked'] == 1)].index.values
        p_count += len(issue_id_linked_index[iid])
    y_pro = data_df[ir_method].values
    issue_id_recommend = {}
    r_count = 0
    for iid in issue_id_list:
        indexes = list(index_loc[index_loc['raw_indexes'].isin(issue_id_index[iid])].index.values)
        linked_prob = y_pro[indexes]
        rec_indexes = [index_loc['raw_indexes'][indexes[x]] for x in np.argsort(-linked_prob)[0: min(n, len(indexes))]]
        issue_id_recommend[iid] = rec_indexes
        r_count += len(issue_id_recommend[iid])

    # calculate p, r, F, MAP, MRR
    tp_count = 0
    sum_ap = 0.0
    sum_reciprocal_rank = 0.0
    for iid in issue_id_list:
        p_indexes = issue_id_linked_index[iid]
        r_indexes = issue_id_recommend[iid]
        tp_count += len(set(p_indexes) & set(r_indexes))

        ap = calculate_AP(r_indexes, p_indexes)
        reciprocal_rank = calculate_reciprocal_rank(r_indexes, p_indexes)
        sum_ap += ap
        sum_reciprocal_rank += reciprocal_rank

    p = tp_count / r_count if r_count != 0 else 0
    r = tp_count / p_count if p_count != 0 else 0
    F = 2 * p * r / (p + r) if p + r != 0 else 0
    MAP = sum_ap / len(issue_id_list)
    MRR = sum_reciprocal_rank / len(issue_id_list)
    best_f1, best_f2, best_f1_p, best_f1_r, details, _ = prc('name', data_df['is_linked'].values, y_pro)

    return p, r, F, MAP, MRR, best_f1, best_f1_p, best_f1_r, best_f2


def new_IR_evaluation(data_df: pd.DataFrame, ir_method, n):
    issue_list = data_df['issue_id'].drop_duplicates().values.tolist()
    predict_true_index = []
    for issue in issue_list:
        part_df = data_df[data_df['issue_id'] == issue]
        part_df = part_df.sort_values(by=[ir_method], ascending=[False])
        predict_true_index.extend(part_df.head(n).index.values.tolist())
    predict_df = data_df.copy()
    predict_df['predict'] = 0
    predict_df.loc[predict_df.index.isin(predict_true_index), 'predict'] = 1
    y_pred = predict_df['predict'].values
    y_label = predict_df['is_linked'].values
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for idx in predict_df.index.values:
        pred = predict_df['predict'][idx]
        label = predict_df['is_linked'][idx]
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        elif pred == 0 and label == 0:
            tn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    one_IoU = tp / (fn + tp + fp)
    try:
        auc = metrics.roc_auc_score(y_label, y_pred)
    except ValueError:
        auc = 0.0
    return precision, recall, f1, auc, one_IoU, accuracy


if __name__ == '__main__':
    sources = ['oss', 'industry']
    methods = [('VSM', 'a19'), ('LSI', 'a20'), ('LDA', 'a22')]
    for source in sources:
        evaluate_path = '{}_IR_evaluate.csv'.format(source)
        recommend_path = '{}_IR_recommend.csv'.format(source)
        file_list = get_file_list(source)
        for file in file_list:
            for i in range(10):
                project_name = get_project_name(file)
                print('#' * 10, project_name, 'iter', i, '#' * 10)
                data_df = pd.read_csv(file)
                data_df = data_df[data_df['is_linked'].isin([1, 0])]
                indices = data_df.index.values
                np.random.shuffle(indices)
                test_df = data_df[data_df.index.isin(indices[0: int(len(indices) * 0.2)])]

                for method, column in methods:
                    test_df[column].fillna(0, inplace=True)
                    test_df[column].replace(np.inf, 1, inplace=True)
                    for n in [1, 2, 3, 5, 10]:
                        row_head = [project_name, '{}_{}'.format(method, n), '-', '-']
                        evaluate_metrics = new_IR_evaluation(test_df, column, n)
                        evaluation_rows = [row_head + [i] + list(evaluate_metrics) + ['-', '-']]
                        write_evaluation(evaluate_path, evaluation_rows)
