import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

from Similarity.Method import Method


class VSM(Method):

    def generate_model(self, n_grams=(1, 1)):
        print("Generating new VSM model")
        model = self.new_model("VSM")

        source_tokens = [source.split(" ") for source in self.source_processed]
        target_tokens = [target.split(" ") for target in self.target_processed]

        source_n_gram = [self._n_gram(token, n_grams) for token in source_tokens]
        target_n_gram = [self._n_gram(token, n_grams) for token in target_tokens]

        all_n_gram = source_n_gram + target_n_gram
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(all_n_gram)
        array = matrix.toarray()

        source_num = len(source_n_gram)

        sources = model.get_source_names()
        targets = model.get_target_names()

        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                j += source_num

                source_vector = array[i]
                target_vector = array[j]

                similarity = np.dot(source_vector, target_vector) / norm(source_vector) * norm(target_vector)

                model.set_value(source, target, similarity)

        print("Done generating VSM model")
        return model

    def _n_gram(self, token, n_grams):
        min_n, max_n = n_grams
        if max_n == 1:
            return token
        original_tokens = token
        tokens = []
        n_original_tokens = len(original_tokens)
        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens.append(" ".join(original_tokens[i: i + n]).replace(" ", ""))

        return " ".join(tokens)
