# coding=utf-8

import csv
import math
import os
from time import *

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
from sklearn.ensemble import RandomForestClassifier

from metric import model_evaluation, recommend_link
from util import write_iteration, write_evaluation, write_recommend, get_path, get_file_list, get_project_name

DS_RATE = 4
NUM_CLASS = 2
TGT_PORT = (0.2, 0.05)
ITERATION_NUM = 3
LRENT = 0.25
ALPHA = 1 / 3


def write_res(artifact, label_process, split_method, sampling_method, crossIR, exp_no):
    n = [1, 2, 3, 5, 10]
    model_evaluation_path = '{}_evaluation_res_{}.csv'.format(artifact, exp_no)
    model_recommend_path = '{}_recommend_res_{}.csv'.format(artifact, exp_no)

    path = get_path(artifact)

    file_list = get_file_list(path)

    for file in file_list:
        project_name = get_project_name(file)
        print("#####################################################")
        print('project_name:' + project_name)

        # According the label process, the dataset will be divided into three parts:
        # 1) original dataset; 2) labeled dataset; 3) unlabeled dataset
        data_set = get_data_set(file, crossIR)
        print('-----------------------------------------------------')
        print(label_process, split_method, sampling_method)
        row_res = [project_name, label_process, split_method, sampling_method]
        for i in range(1):
            print('---- round ' + str(i + 1) + '----')
            model_evaluation_metrics, model_recommend_metrics, train_time_cost, test_time_cost = control(
                data_set, label_process, split_method, sampling_method, project_name, i, crossIR)
            for k, metric in enumerate(model_evaluation_metrics):
                rows = [row_res + [i] + list(metric) + [train_time_cost[k], test_time_cost[k]]]
                write_evaluation(model_evaluation_path, rows)

            for metric in model_recommend_metrics:
                for nn in range(len(n)):
                    rows = [row_res + [i, n[nn]] + list(metric[nn])]
                    write_recommend(model_recommend_path, rows)

        print('finish writing this round...')


def control(data_set, label_process, split_method, sampling_method, project_name, round, crossIR, port=TGT_PORT,
            alpha=ALPHA):
    n = [1, 2, 3, 5, 10]

    original_data, labeled_data, unlabeled_X = data_set
    if label_process == 'original':
        data = original_data
    else:
        data = labeled_data

    # model evaluation metrics: Precision，Recall，F1，AUC
    evaluation_metrics, recommend_metrics, train_time_list, test_time_list = [], [], [], []

    if split_method == 'time_split':
        train_begin = time()
        print('data split：' + split_method)
        X_train, y_train, X_test, y_test = time_split(data, crossIR)

        rate = np.sum(y_train == 1) / np.sum(y_train == 0)

        # different sampling method
        print('data sampling：' + sampling_method)
        X_train, y_train = data_sampling(sampling_method, X_train, y_train)

        # train the initial classifier
        print('training initial classifier...')
        clf_initial = random_forrest(X_train, y_train)
        # use different parameters for RQ3
        if label_process == 'ssl':
            evaluation_metrics, recommend_metrics, train_time_list, test_time_list = [], [], [], []
            for p in [(0.2, 0.1), (0.2, 0.05), (0.2, 0.025), (0.15, 0.05), (0.1, 0.05)]:
                for a in [0, 1 / 4, 1 / 3, 1 / 2, 1]:
                    for method in ['SPLINT']:
                        clf_final = ssl(project_name, method, sampling_method, clf_initial, X_train, y_train,
                                        unlabeled_X, p, a, X_test, y_test, crossIR, rate)
                        train_end = time()
                        train_time_cost = train_end - train_begin
                        print('evaluating classifier...')
                        model_evaluation_metrics, test_time_cost = model_evaluation(clf_final, X_test, y_test)

                        print('top n recommend evaluation...')
                        name = 'prc_image_topn/{}_{}_{}_{}_{}.png'.format(project_name, label_process, split_method,
                                                                          sampling_method, str(round))
                        model_recommend_metrics = [recommend_link(original_data, X_test, clf_final, name, y_test, topn)
                                                   for topn in n]
                        evaluation_metrics.append(model_evaluation_metrics)
                        recommend_metrics.append(model_recommend_metrics)
                        train_time_list.append(train_time_cost)
                        test_time_list.append(test_time_cost)

        # use default parameters for SPLINT
        elif label_process == 'SPLINT':
            evaluation_metrics, recommend_metrics, train_time_list, test_time_list = [], [], [], []
            clf_final = ssl(project_name, 'SPLINT', sampling_method, clf_initial, X_train, y_train, unlabeled_X,
                            port, alpha, X_test, y_test, crossIR, rate)
            train_end = time()
            train_time_cost = train_end - train_begin
            print('evaluating classifier...')
            model_evaluation_metrics, test_time_cost = model_evaluation(clf_final, X_test, y_test)

            print('top n recommend evaluation...')
            name = 'prc_image_topn/{}_{}_{}_{}_{}.png'.format(project_name, label_process, split_method,
                                                              sampling_method, str(round))
            model_recommend_metrics = [recommend_link(original_data, X_test, clf_final, name, y_test, topn)
                                       for topn in n]
            evaluation_metrics.append(model_evaluation_metrics)
            recommend_metrics.append(model_recommend_metrics)
            train_time_list.append(train_time_cost)
            test_time_list.append(test_time_cost)

        else:
            evaluation_metrics, recommend_metrics, train_time_list, test_time_list = [], [], [], []
            clf_final = clf_initial
            train_end = time()
            train_time_cost = train_end - train_begin

            print('evaluating classifier...')
            model_evaluation_metrics, test_time_cost = model_evaluation(clf_final, X_test, y_test)

            print('top n recommend evaluation...')
            name = 'prc_image_topn/{}_{}_{}_{}_{}.png'.format(project_name, label_process, split_method,
                                                              sampling_method, str(round))
            model_recommend_metrics = [recommend_link(original_data, X_test, clf_final, name, y_test, topn)
                                       for topn in n]
            evaluation_metrics.append(model_evaluation_metrics)
            recommend_metrics.append(model_recommend_metrics)
            train_time_list.append(train_time_cost)
            test_time_list.append(test_time_cost)

    return evaluation_metrics, recommend_metrics, train_time_list, test_time_list


def random_forrest(X, y):
    """
    use random forest to train the classifier
    :param X: feature
    :param y: label
    :return:
    """
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf


def y_reshape(y_original):
    length = len(y_original)
    y_new = y_original.reshape(length, 1)
    return y_new


# introduce SSL
def ssl(project_name, label_process, sampling_method, clf_initial, X_train, y_train, unlabeled_X, port, alpha, X_test,
        y_test, crossIR, rate):
    def cbst(project_name, clf_initial, train_X, train_y, unlabeled_X, port, alpha, X_test, y_test, crossIR,
             not_adjust=True):
        X_un = unlabeled_X.copy()
        X_train = train_X.copy()
        y_train = train_y.copy()
        method = 'CBST' if not_adjust else 'SPLINT'
        row_header = [project_name, method, sampling_method]
        print(len(X_un))
        print('initial_evaluate')
        evaluation_metric, _ = model_evaluation(clf_initial, X_test, y_test)
        # print(evaluation_metric[0], evaluation_metric[1], evaluation_metric[2], evaluation_metric[3], sep='\t')
        iter_num = 1
        write_iteration(row_header + [iter_num, '-', '-', '-', '-', '-', evaluation_metric[0], evaluation_metric[1],
                                      evaluation_metric[2], evaluation_metric[3], evaluation_metric[4],
                                      evaluation_metric[5]])
        # print('p', 'kc{0, 1}', 'add 0', 'add 1', sep='\t')
        feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
                        'a16', 'a17', 'a18']
        if crossIR:
            feature_name = feature_name + ['a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27']

        iter_port = port[0]
        port_step = port[1]
        for i in range(ITERATION_NUM):
            print('<---Iteration {} for crst--->'.format(i))
            if len(X_un) == 0:
                print('finish this iteration')
                break
            if i > 0:
                iter_port = iter_port + port_step if iter_port + port_step < 1 else 1
            conf_dict, pred_cls_num, unlabeled_prob, unlabeled_label = val(clf_initial, X_un.copy())

            cls_thresh = kc_parameters(conf_dict, pred_cls_num, iter_port)
            # print(cls_thresh)
            pseudo_label = label_generation(unlabeled_prob, cls_thresh)

            # select more pseudo-labels from the minor class
            if not not_adjust:
                u = math.pow(rate, alpha)
                # print(u)
                zero_id_list = pseudo_label == 0
                # print(np.sum(zero_id_list))
                conf = np.amax(unlabeled_prob, axis=1)
                prob_zero = conf[zero_id_list].astype(np.float32)
                index_list = np.where(zero_id_list == True)[0]
                len_thresh = int(math.floor(len(prob_zero) * u))
                # print(len_thresh)
                if len(prob_zero) > 0:
                    sort_index = np.lexsort((-index_list, -prob_zero))[len_thresh:]
                    # print(sort_index)
                    chosen_index = index_list[sort_index]
                    pseudo_label[chosen_index] = -1

            X = X_un.copy()
            X['is_linked'] = pseudo_label
            pseudo_data = X.loc[X['is_linked'] != -1]
            add_0 = len(pseudo_data[pseudo_data['is_linked'] == 0])
            add_1 = len(pseudo_data[pseudo_data['is_linked'] == 1])
            X_pseudo = pseudo_data.loc[:, feature_name]
            y_pseudo = pseudo_data.loc[:, ['is_linked']].values.astype('int').ravel()

            print(iter_port, cls_thresh, len(pseudo_data[pseudo_data['is_linked'] == 0]),
                  len(pseudo_data[pseudo_data['is_linked'] == 1]), sep='\t')

            unlabeled_data = X.loc[X['is_linked'] == -1]
            unlabeled_x = unlabeled_data.loc[:, feature_name]

            X_un = unlabeled_x
            X_train = pd.concat([X_train, X_pseudo])
            y_train = np.concatenate((y_train, y_pseudo))
            X_train, y_train = data_sampling(sampling_method, X_train, y_train)
            clf_initial = random_forrest(X_train, y_train)
            evaluation_metric, _ = model_evaluation(clf_initial, X_test, y_test)

            write_iteration(row_header + [i + 1, alpha, iter_port, cls_thresh, add_0, add_1, evaluation_metric[0],
                                          evaluation_metric[1], evaluation_metric[2], evaluation_metric[3],
                                          evaluation_metric[4], evaluation_metric[5]])

        return clf_initial

    if label_process == 'cbst':
        print('<<<<<CBST...>>>>>>')
        clf_final = cbst(project_name, clf_initial, X_train, y_train, unlabeled_X, port, alpha, X_test, y_test, crossIR,
                         True)
    elif label_process == 'SPLINT':
        print('<<<<<SPLINT...>>>>>')
        clf_final = cbst(project_name, clf_initial, X_train, y_train, unlabeled_X, port, alpha, X_test, y_test, crossIR,
                         False)

    return clf_final


def val(cls: RandomForestClassifier, X_unlabeled):
    unlabeled_prob = cls.predict_proba(X_unlabeled)
    unlabeled_label = cls.predict(X_unlabeled)

    conf = np.amax(unlabeled_prob, axis=1)
    conf_dict = {k: [] for k in range(NUM_CLASS)}
    pred_cls_num = np.zeros(NUM_CLASS)

    for idx_cls in range(NUM_CLASS):
        idx_temp = unlabeled_label == idx_cls
        pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
        if idx_temp.any():
            conf_cls_temp = conf[idx_temp].astype(np.float32)
            len_cls_temp = conf_cls_temp.size
            conf_cls = conf_cls_temp[0:len_cls_temp:DS_RATE]
            conf_dict[idx_cls].extend(conf_cls)

    return conf_dict, pred_cls_num, unlabeled_prob, unlabeled_label


# calculate the kc value to determine which pseudo-labels will be selected
def kc_parameters(conf_dict, pred_cls_num, port):
    cls_thresh = np.ones(NUM_CLASS, dtype=np.float32)
    cls_sel_size = np.zeros(NUM_CLASS, dtype=np.float32)
    cls_size = np.zeros(NUM_CLASS, dtype=np.float32)

    for idx_cls in np.arange(0, NUM_CLASS):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] is not None:
            conf_dict[idx_cls].sort(reverse=True)
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * port))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh - 1]
            conf_dict[idx_cls] = None

    return cls_thresh


# generate pseudo-labels
def label_generation(unlabeled_prob, cls_thresh):
    # label regularization
    soft_pseudo_label = np.power(unlabeled_prob / cls_thresh, 1.0 / LRENT)
    soft_pseudo_label_sum = soft_pseudo_label.sum(1)
    soft_pseudo_label = soft_pseudo_label.transpose(1, 0) / soft_pseudo_label_sum
    soft_pseudo_label = soft_pseudo_label.transpose(1, 0).astype(np.float32)
    weighted_pred_id = np.asarray(np.argmax(soft_pseudo_label, axis=1))
    reg_score = np.sum(-soft_pseudo_label * np.log(unlabeled_prob + 1e-32) + LRENT * soft_pseudo_label * np.log(
        soft_pseudo_label + 1e-32), axis=1)
    sel_score = np.sum(-soft_pseudo_label * np.log(cls_thresh + 1e-32), axis=1)
    pseudo_label = weighted_pred_id.copy()
    pseudo_label[reg_score > sel_score] = -1
    return pseudo_label


def data_sampling(sampling_method, X, y):
    """
    different data sampling method
    :param sampling_method: method name
    :param X: features
    :param y: label
    :return: X,y
    """
    if sampling_method == 'smote':
        X_smote, y_smote = SMOTE(random_state=42, k_neighbors=1).fit_resample(X, y)
        return X_smote, y_smote
    elif sampling_method == "smote_enn":
        X_SMOTE_ENN, y_SMOTE_ENN = SMOTEENN(random_state=0).fit_resample(X, y)
        return X_SMOTE_ENN, y_SMOTE_ENN
    elif sampling_method == 'smote_tomek':
        X_SMOTE_t, y_SMOTE_t = SMOTETomek(random_state=0).fit_resample(X, y)
        return X_SMOTE_t, y_SMOTE_t
    elif sampling_method == 'oss':
        X_oss, y_oss = OneSidedSelection(random_state=0).fit_resample(X, y)
        return X_oss, y_oss
    else:
        return X, y


# split issues into 80%-20%
def time_split(data, crossIR):
    issues = data[['issue_id', 'issue_created']].drop_duplicates()
    issues = issues.sort_values(by=['issue_created'])
    train_issues = issues[:int(0.8 * len(issues))]
    train_issues = train_issues['issue_id'].values.tolist()
    test_issues = issues[int(0.8 * len(issues)):]
    test_issues = test_issues['issue_id'].values.tolist()
    train_data = data[data['issue_id'].isin(train_issues)]
    test_data = data[data['issue_id'].isin(test_issues)]

    X_train, y_train = split_data(train_data, crossIR)
    X_test, y_test = split_data(test_data, crossIR)
    return X_train, y_train, X_test, y_test


def split_data(df, crossIR):
    """
    :param crossIR: True/False use crossIR features or not
    :param df: features
    :return: X,y
    """
    feature_name = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15',
                    'a16', 'a17', 'a18']
    if crossIR:
        feature_name = feature_name + ['a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27']
    X = df.loc[:, feature_name]
    y = df.loc[:, ['is_linked']].values.astype('int').ravel()
    return X, y


def str_2_int(data) -> pd.DataFrame:
    data['a10'].fillna(-1, inplace=True)
    data['a13'].fillna(-1, inplace=True)
    developer_list = list(
        set(data['a10'].drop_duplicates().values.tolist() + data['a13'].drop_duplicates().values.tolist()))
    developer_dict = {}
    for i in range(len(developer_list)):
        developer_dict[developer_list[i]] = i
    data['a10'] = data['a10'].apply(lambda x: developer_dict[x])
    data['a13'] = data['a13'].apply(lambda x: developer_dict[x])

    data['a1'].fillna(-1, inplace=True)
    list1 = list(set(data['a1'].drop_duplicates().values.tolist()))
    dict1 = {}
    for i in range(len(list1)):
        dict1[list1[i]] = i
    data['a1'] = data['a1'].apply(lambda x: dict1[x])

    data['a2'].fillna(-1, inplace=True)
    list2 = list(set(data['a2'].drop_duplicates().values.tolist()))
    dict2 = {}
    for i in range(len(list2)):
        dict2[list2[i]] = i
    data['a2'] = data['a2'].apply(lambda x: dict2[x])
    return data


def get_data_set(path, crossIR):
    """
    return dataset according different label processes
    :param path: file path
    :param crossIR: whether or not use new features
    :return: original_data, labeled_data, unlabeled_data
    """
    df = pd.read_csv(path)
    df = str_2_int(df)
    df = df.sort_values(by="issue_created", ascending=True)
    df['is_linked'].fillna(-1, inplace=True)

    columns = ['a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27']
    for c in columns:
        df[c].fillna(-1, inplace=True)
        df[c].replace(np.inf, 1, inplace=True)
    original_data = df.copy()
    original_data.loc[original_data['is_linked'] == -1] = 0

    labeled_data = df[df['is_linked'].isin([0, 1])]
    unlabeled_data = df[df['is_linked'] == -1]
    unlabeled_X, _ = split_data(unlabeled_data, crossIR)
    return original_data, labeled_data, unlabeled_X
