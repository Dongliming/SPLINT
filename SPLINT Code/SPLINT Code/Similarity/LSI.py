from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
import numpy as np

from Similarity.Method import Method


class LSI(Method):
    def generate_model(self, n_topics=None):
        print("Generating new LSI model")

        vectorizer = CountVectorizer()
        all_artifacts = self.source_processed + self.target_processed
        dtm = vectorizer.fit_transform(all_artifacts).astype('d')

        if n_topics is None:
            n_components = min(dtm.shape) - 1
        else:
            n_components = min(dtm.shape) - 1

        model = self.new_model("LSI")

        lsi = TruncatedSVD(n_components, algorithm='arpack')

        dtm_lsi = lsi.fit_transform(dtm)
        dtm_lsi = Normalizer(copy=False).fit_transform(dtm_lsi)

        similarity_matrix = np.asarray(np.asmatrix(dtm_lsi) * np.asmatrix(dtm_lsi).T)

        sources = model.get_source_names()
        targets = model.get_target_names()

        source_len = len(sources)

        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                j += source_len
                similarity = similarity_matrix[i][j]
                model.set_value(source, target, similarity)

        print("Done generating LSI model")
        return model
