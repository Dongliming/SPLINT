from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF as NonnegativeMatrixFactorization

from Similarity.Method import Method


class NMF(Method):
    def generate_model(self, n_topics=10):
        print("Generating new NMF model")

        model = self.new_model("NMF")

        vectorizer = TfidfVectorizer()
        all_artifacts = self.source_processed + self.target_processed
        tfidf_matrix = vectorizer.fit_transform(all_artifacts)
        nmf_model = NonnegativeMatrixFactorization(n_components=n_topics).fit(tfidf_matrix)
        doc_topic_matrix = nmf_model.transform(tfidf_matrix)
        sources = model.get_source_names()
        targets = model.get_target_names()

        num_sources = len(sources)

        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                j += num_sources

                source_topic_vector = doc_topic_matrix[i]
                target_topic_vector = doc_topic_matrix[j]

                for k in range(len(source_topic_vector)):
                    if source_topic_vector[k] == 0:
                        source_topic_vector[k] = 0.00001
                for k in range(len(target_topic_vector)):
                    if target_topic_vector[k] == 0:
                        target_topic_vector[k] = 0.00001

                similarity = 1 / entropy(source_topic_vector, target_topic_vector)

                model.set_value(source, target, similarity)

        print("Done generating NMF model")
        return model
