from controller import write_res


def experiment_01(artifact):
    # baseline vs. SPLINT
    write_res(artifact, 'labeled', 'time_split', 'smote', False, 'RQ1')
    write_res(artifact, 'SPLINT', 'time_split', 'smote', True, 'RQ1')


def experiment_02(artifact):
    # SPLINT vs. SPLINT with noCrossIR vs. SPLINT with noSSL
    write_res(artifact, 'SPLINT', 'time_split', 'smote', True, 'RQ2')
    write_res(artifact, 'SPLINT', 'time_split', 'smote', False, 'RQ2')
    write_res(artifact, 'labeled', 'time_split', 'smote', True, 'RQ2')


def experiment_03(artifact):
    # different parameters of SPLINT
    write_res(artifact, 'ssl', 'time_split', 'smote', True, 'RQ3')


if __name__ == '__main__':
    experiment_01('oss')
    experiment_01('industry')
    experiment_02('oss')
    experiment_02('industry')
    experiment_03('oss')
    experiment_03('industry')
