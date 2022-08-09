from math import sqrt

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from Similarity.Method import Method


class LDA(Method):
    def generate_model(self, n_topics=10):
        print("Generating new LDA model")

        model = self.new_model("LDA")

        all_artifacts = self.source_processed + self.target_processed

        vectorizer = CountVectorizer()
        tf_matrix = vectorizer.fit_transform(all_artifacts)

        lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online').fit(tf_matrix)
        doc_topic_matrix = lda_model.transform(tf_matrix)

        sources = model.get_source_names()
        targets = model.get_target_names()

        num_sources = len(sources)

        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                j += num_sources

                source_topic_vector = doc_topic_matrix[i]
                target_topic_vector = doc_topic_matrix[j]

                similarity = 1 - (1 / sqrt(2)) * sqrt(
                    sum([(sqrt(source_topic_vector[i]) - sqrt(target_topic_vector[i])) ** 2
                         for i in range(len(source_topic_vector))]))

                model.set_value(source, target, similarity)

        print("Done generating LDA model")
        return model
