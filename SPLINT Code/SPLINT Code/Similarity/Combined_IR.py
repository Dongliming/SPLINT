from Similarity.Method import Method
from Similarity.Model import SimilarityModel
import numpy as np
from Similarity.util import mean, var


class CombinedIR(Method):

    def generate_model(self, model1: SimilarityModel, model2: SimilarityModel, parameter=0.5):
        name1 = model1.get_name()
        name2 = model2.get_name()
        print("Generating new Orthogonal model: " + name1 + "+" + name2)

        if model1.get_corpus_name() != model2.get_corpus_name():
            raise ValueError("需要给出文本来自同一项目的两个模型进行结合！！！")

        all_val1 = model1.get_all_values()

        all_val2 = model2.get_all_values()

        mean1 = mean(all_val1)
        var1 = var(all_val1)
        mean2 = mean(all_val2)
        var2 = var(all_val2)

        lambda_ = parameter

        combined_model = self.new_model(name1 + "+" + name2)
        for source, target in model1.get_all():
            sim1 = (model1.get_value(source, target) - mean1) / var1

            sim2 = (model2.get_value(source, target) - mean2) / var2

            sim_combined = (lambda_ * sim1) + ((1 - lambda_) * sim2)

            combined_model.set_value(source, target, sim_combined)

        print("Done generating orthogonal model: " + name1 + "+" + name2)
        return combined_model
