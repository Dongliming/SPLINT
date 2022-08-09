from .util import preprocess
from .Model import SimilarityModel


class Method:
    """
    the base class
    """
    EXISTING_METHOD = []

    def __init__(self, corpus_name, sources, targets):
        """
        :param corpus_name: project
        :param sources: (source_id, text)
        :param targets: (target_id, text)
        """
        self.corpus_name = corpus_name
        self.source_id = []
        self.source_text = []
        for source in sources:
            self.source_id.append(source[0])
            self.source_text.append(source[1])

        self.target_id = []
        self.target_text = []
        for target in targets:
            self.target_id.append(target[0])
            self.target_text.append(target[1])

        for method in Method.EXISTING_METHOD:
            if method.corpus_name == self.corpus_name:
                self.source_processed = method.source_processed
                self.target_processed = method.target_processed
                break
        else:
            source_tokenized = preprocess(self.source_text)
            target_tokenized = preprocess(self.target_text)
            self.source_processed = [' '.join([x for x in source_artifact]) for source_artifact
                                     in source_tokenized]
            self.target_processed = [' '.join([x for x in target_artifact]) for target_artifact
                                     in target_tokenized]
            Method.EXISTING_METHOD.append(self)

    def new_model(self, name):
        model = SimilarityModel(name, self.corpus_name, self.source_id, self.target_id)
        return model
