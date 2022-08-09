from .Method import Method
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy


class JensenShannon(Method):

    def generate_model(self):
        print('Generating new Jensen-Shannon model')

        model = self.new_model('JSD')

        vectorizer = CountVectorizer()
        texts = self.source_processed + self.target_processed
        source_num = len(self.source_processed)
        target_num = len(self.target_processed)
        matrix = vectorizer.fit_transform(texts)
        array = matrix.toarray().astype(float)

        source_array = array[0:source_num]
        target_array = array[source_num:]

        # source_matrix = vectorizer.fit_transform(self.source_processed)
        # target_matrix = vectorizer.fit_transform(self.target_processed)

        # source_array = source_matrix.toarray().astype(float)
        for vector in source_array:
            vec_sum = vector.sum()
            for i in range(len(vector)):
                vector[i] /= vec_sum

        # target_array = target_matrix.toarray().astype(float)
        for vector in target_array:
            vec_sum = vector.sum()
            for i in range(len(vector)):
                vector[i] /= vec_sum

        sources = model.get_source_names()
        targets = model.get_target_names()

        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                source_vector = source_array[i]
                target_vector = target_array[j]

                m = (source_vector + target_vector) / 2
                similarity = 1 - ((entropy(source_vector, m) + entropy(target_vector, m)) / 2)

                model.set_value(source, target, similarity)

        print("Done generating Jensen-Shannon model")

        return model
