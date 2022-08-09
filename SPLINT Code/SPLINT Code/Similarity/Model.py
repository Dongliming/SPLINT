import os
import numpy as np


class SimilarityModel:
    def __init__(self, name, corpus_name, source_ids, target_ids):
        """
        :param name: the name of the IR method
        :param corpus_name: project name
        :param source_ids:
        :param target_ids:
        """

        self.name = name
        self.corpus_name = corpus_name
        self.source_ids = source_ids
        self.target_ids = target_ids

        self.similarity_matrix = dict()
        for sid in source_ids:
            self.similarity_matrix[sid] = dict()
            for tid in target_ids:
                self.similarity_matrix[sid][tid] = None

    def get_name(self):
        return self.name

    def get_corpus_name(self):
        return self.corpus_name

    def get_source_names(self):
        return self.source_ids

    def get_target_names(self):
        return self.target_ids

    def set_value(self, source, target, similarity):
        if source not in self.source_ids:
            raise KeyError("source artifact \'" + source + "\' doesn't exist")
        if target not in self.target_ids:
            raise KeyError("target artifact \'" + target + "\' doesn't exist")

        self.similarity_matrix[source][target] = similarity

    def get_value(self, source, target):
        if source not in self.source_ids:
            raise KeyError("source artifact \'" + source + "\' doesn't exist")
        if target not in self.target_ids:
            raise KeyError("target artifact \'" + target + "\' doesn't exist")

        return self.similarity_matrix[source][target]

    def get_all(self):
        for source in self.get_source_names():
            for target in self.get_target_names():
                if self.similarity_matrix[source][target] is not None:
                    yield source, target

    def get_all_values(self):
        return [self.similarity_matrix[source][target] for source, target in self.get_all()]

    def write_file(self):
        dir_path = 'Similarity/model_output/{}'.format(self.corpus_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        path = 'Similarity/model_output/{}/{}_{}.txt'.format(self.corpus_name, self.corpus_name, self.name)
        with open(path, 'w+', encoding='utf-8') as f:
            for source, target in self.get_all():
                line = '{} {} {}'.format(source, target, self.get_value(source, target))
                f.write(line.strip() + '\n')

    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line) <= 1:
                    continue
                info = line.split(" ")
                source_id = info[0]
                target_id = info[1]
                similarity = info[2]
                if similarity == 'inf':
                    similarity = np.inf
                elif similarity == 'nan':
                    similarity = np.nan
                else:
                    similarity = float(similarity)
                self.similarity_matrix[source_id][target_id] = similarity
