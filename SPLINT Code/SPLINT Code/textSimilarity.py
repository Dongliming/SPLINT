from Similarity import *
import json
import os
import pandas as pd

COMMIT_PATH = ['raw_data/commit/{}.json', 'raw_data/commit/{}_origin.json']
ISSUE_DIR = 'raw_data/issue/{}/{}'
ISSUE_PATH = 'raw_data/issue/{}/{}/{}.json'
OLD_FEATURE_PATH = 'oss_1/{}_{}.csv'
NEW_FEATURE_PATH = 'oss/{}_{}.csv'
MODEL_PATH = 'Similarity/model_output/{}/{}_{}.txt'


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content


def get_issues_from_dir(dir_path):
    issues = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            issues.append(file.split('.')[0])
    return issues


def get_artifacts(project, abbr, kind, old_commits, old_issues):
    commit_path = [path.format(project) for path in COMMIT_PATH]
    issue_dir = ISSUE_DIR.format(abbr, kind)

    all_commits = {}
    for p in commit_path:
        all_commits.update(read_json(p))
        # print(len(all_commits.keys()))

    commit_artifacts = []
    for sha in all_commits.keys():
        commit_artifacts.append((sha, all_commits[sha]['message']))

    issues = get_issues_from_dir(issue_dir)
    if kind == 'improvement':
        issues.extend(get_issues_from_dir(ISSUE_DIR.format(abbr, 'enhancement')))
    issue_artifacts = []
    for issue_id in issues:
        issue_path = ISSUE_PATH.format(abbr, kind, issue_id)
        issue_content = read_json(issue_path)
        issue_artifacts.append((issue_id, issue_content['description']))

    commit_artifacts = [commit for commit in commit_artifacts if commit[0] in old_commits]
    issue_artifacts = [issue for issue in issue_artifacts if issue[0] in old_issues]

    return commit_artifacts, issue_artifacts


def generate_vsm_model(corpus_name, commit_artifacts, issue_artifacts):
    vsm = VSM.VSM(corpus_name, commit_artifacts, issue_artifacts)
    return vsm.generate_model(n_grams=(2, 3))


def generate_lsi_model(corpus_name, commit_artifacts, issue_artifacts):
    lsi = LSI.LSI(corpus_name, commit_artifacts, issue_artifacts)
    return lsi.generate_model()


def generate_jsd_model(corpus_name, commit_artifacts, issue_artifacts):
    jsd = JensenShannon.JensenShannon(corpus_name, commit_artifacts, issue_artifacts)
    return jsd.generate_model()


def generate_lda_model(corpus_name, commit_artifacts, issue_artifacts):
    lda = LDA.LDA(corpus_name, commit_artifacts, issue_artifacts)
    return lda.generate_model()


def generate_nmf_model(corpus_name, commit_artifacts, issue_artifacts):
    nmf = NMF.NMF(corpus_name, commit_artifacts, issue_artifacts)
    return nmf.generate_model()


def generate_vsm_lda_model(corpus_name, commit_artifacts, issue_artifacts, vsm, lda):
    if vsm is None:
        vsm = generate_vsm_model(corpus_name, commit_artifacts, issue_artifacts)

    if lda is None:
        lda = generate_lda_model(corpus_name, commit_artifacts, issue_artifacts)
    vsm_lda = Combined_IR.CombinedIR(corpus_name, commit_artifacts, issue_artifacts)
    return vsm_lda.generate_model(vsm, lda)


def generate_jsd_lda_model(corpus_name, commit_artifacts, issue_artifacts, jsd, lda):
    if jsd is None:
        jsd = generate_jsd_model(corpus_name, commit_artifacts, issue_artifacts)

    if lda is None:
        lda = generate_lda_model(corpus_name, commit_artifacts, issue_artifacts)
    jsd_lda = Combined_IR.CombinedIR(corpus_name, commit_artifacts, issue_artifacts)
    return jsd_lda.generate_model(jsd, lda)


def generate_vsm_nmf_model(corpus_name, commit_artifacts, issue_artifacts, vsm, nmf):
    if vsm is None:
        vsm = generate_vsm_model(corpus_name, commit_artifacts, issue_artifacts)

    if nmf is None:
        nmf = generate_nmf_model(corpus_name, commit_artifacts, issue_artifacts)
    vsm_nmf = Combined_IR.CombinedIR(corpus_name, commit_artifacts, issue_artifacts)
    return vsm_nmf.generate_model(vsm, nmf)


def generate_jsd_nmf_model(corpus_name, commit_artifacts, issue_artifacts, jsd, nmf):
    if jsd is None:
        jsd = generate_jsd_model(corpus_name, commit_artifacts, issue_artifacts)

    if nmf is None:
        nmf = generate_nmf_model(corpus_name, commit_artifacts, issue_artifacts)
    jsd_nmf = Combined_IR.CombinedIR(corpus_name, commit_artifacts, issue_artifacts)
    return jsd_nmf.generate_model(jsd, nmf)


def generate_vsm_jsd_model(corpus_name, commit_artifacts, issue_artifacts, vsm, jsd):
    if vsm is None:
        vsm = generate_vsm_model(corpus_name, commit_artifacts, issue_artifacts)

    if jsd is None:
        jsd = generate_jsd_model(corpus_name, commit_artifacts, issue_artifacts)
    vsm_jsd = Combined_IR.CombinedIR(corpus_name, commit_artifacts, issue_artifacts)
    return vsm_jsd.generate_model(vsm, jsd)


def generate_sim_model(name, corpus_name, commits, issues, model_dict):
    commits_id = [commit[0] for commit in commits]
    issues_id = [issue[0] for issue in issues]
    if os.path.exists(MODEL_PATH.format(corpus_name, corpus_name, name)):
        model = Model.SimilarityModel(name, corpus_name, commits_id, issues_id)
        model.read_file(MODEL_PATH.format(corpus_name, corpus_name, name))
        return model
    if name == 'VSM':
        return generate_vsm_model(corpus_name, commits, issues)
    elif name == 'LSI':
        return generate_lsi_model(corpus_name, commits, issues)
    elif name == 'JSD':
        return generate_jsd_model(corpus_name, commits, issues)
    elif name == 'LDA':
        return generate_lda_model(corpus_name, commits, issues)
    elif name == 'NMF':
        return generate_nmf_model(corpus_name, commits, issues)
    elif name == 'VSM+LDA':
        return generate_vsm_lda_model(corpus_name, commits, issues, model_dict['VSM'], model_dict['LDA'])
    elif name == 'JSD+LDA':
        return generate_jsd_lda_model(corpus_name, commits, issues, model_dict['JSD'], model_dict['LDA'])
    elif name == 'VSM+NMF':
        return generate_vsm_nmf_model(corpus_name, commits, issues, model_dict['VSM'], model_dict['NMF'])
    elif name == 'JSD+NMF':
        return generate_jsd_nmf_model(corpus_name, commits, issues, model_dict['JSD'], model_dict['NMF'])
    elif name == 'VSM+JSD':
        return generate_vsm_jsd_model(corpus_name, commits, issues, model_dict['VSM'], model_dict['JSD'])
    return None


def new_sim_feature_for_oss():
    projects = ['derby', 'maven', 'pig', 'drools', 'groovy', 'infinispan']
    abbrs = ['DERBY', 'MNG', 'PIG', 'DROOLS', 'GROOVY', 'ISPN']
    kinds = ['bug']
    sim_method = ['VSM', 'LSI', 'JSD', 'LDA', 'NMF',
                  'VSM+LDA', 'JSD+LDA', 'VSM+NMF', 'JSD+NMF', "VSM+JSD"]

    new_columns = ['a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27']

    for i in range(len(projects)):
        for kind in kinds:
            project = projects[i]
            abbr = abbrs[i]
            corpus_name = project + '_' + kind
            print('now: ', corpus_name)
            old_features = pd.read_csv(OLD_FEATURE_PATH.format(project, kind))

            commits = old_features['commit_hash'].drop_duplicates().values
            issues = old_features['issue_id'].drop_duplicates().values

            commit_artifacts, issue_artifacts = get_artifacts(projects[i], abbrs[i], kind, commits, issues)
            commits = [commit[0] for commit in commit_artifacts]
            issues = [issue[0] for issue in issue_artifacts]

            model_dict = {}
            for method in sim_method:
                model_dict[method] = None

            for method in sim_method:
                model = generate_sim_model(method, corpus_name, commit_artifacts, issue_artifacts, model_dict)
                if model is not None:
                    model_dict[method] = model
                    model.write_file()
                else:
                    print('generate {} model failed'.format(method))

            for col in new_columns:
                old_features[col] = 0.0

            for idx in old_features.index.values:
                commit_id = old_features['commit_hash'][idx]
                issue_id = old_features['issue_id'][idx]
                if commit_id not in commits or issue_id not in issues:
                    continue
                if idx % 1000 == 0:
                    print(corpus_name, idx)

                for j in range(len(new_columns)):
                    method = sim_method[j + 1]
                    column = new_columns[j]
                    # print(model_dict[method])
                    sim = model_dict[method].get_value(commit_id, issue_id) if model_dict[method] is not None else 0
                    old_features[column][idx] = sim
            old_features.to_csv(NEW_FEATURE_PATH.format(project, kind), index=False)


if __name__ == '__main__':
    new_sim_feature_for_oss()
