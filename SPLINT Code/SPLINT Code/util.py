import csv
import os


def write_importance(project, label_process, split_method, sampling_method, crossIR, important_list):
    path = 'important_feature.csv'
    if not os.path.exists(path):
        with open(path, "a", newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            header = ['project_name', 'label_process', 'split_method', 'sampling_method', 'crossIR', 'important_list']
            f_csv.writerow(header)

    with open(path, "a", errors="ignore", newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        row = [project, label_process, split_method, sampling_method, crossIR, important_list]
        f_csv.writerow(row)


def write_iteration(row):
    path = 'ssl_iteration.csv'
    if not os.path.exists(path):
        with open(path, "a", newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            header = ['project_name', 'label_process', 'sampling_method', 'alpha', 'iter_num', 'port', 'k', 'add_0',
                      'add_1', 'P', 'R', 'F1', 'auc', 'one_IoU', 'accuracy']
            f_csv.writerow(header)

    with open(path, "a", errors="ignore", newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(row)


def write_evaluation(model_evaluation_path, rows):
    if not os.path.exists(model_evaluation_path):
        with open(model_evaluation_path, "a", newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            header = ['project_name', 'label_process', 'split_method', 'sampling_method', 'round_num', 'P', 'R', 'F1',
                      'auc', 'one_IoU', 'accuracy', 'train_cost', 'test_cost']
            f_csv.writerow(header)
    with open(model_evaluation_path, "a", errors="ignore", newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)


def write_recommend(model_recommend_path, rows):
    if not os.path.exists(model_recommend_path):
        with open(model_recommend_path, "a", newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            header = ['project_name', 'label_process', 'split_method', 'sampling_method', 'round_num', 'topn', 'P', 'R',
                      'F1', 'MAP', 'MRR', 'best_F1', 'best_F1_P', 'best_F1_R', 'best_F2']
            f_csv.writerow(header)
    with open(model_recommend_path, "a", errors="ignore", newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)


def get_path(artifact):
    """
    :param artifact: oss/industry
    :return: path
    """
    if artifact == 'oss':
        return './oss'
    elif artifact == 'industry':
        return './industry'
    else:
        return "please input a true artifact name"


def get_file_list(filepath):
    arr = []
    for root, dirs, files in os.walk(filepath):
        for fn in files:
            arr.append(root + '/' + fn)
    return arr


def get_project_name(file_name):
    project_name = file_name.split('/')[-1].split('.')[0]
    return project_name
