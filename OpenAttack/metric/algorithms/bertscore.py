from ...tags import *
from .base import AttackMetric
from ...utils import format_sentence
from evaluate import load


class BERTScore(AttackMetric):
    NAME = "BERTScore"
    TAGS = { TAG_English }

    def __init__(self):
        """
        https://huggingface.co/spaces/evaluate-metric/bertscore
        :Language: english
        
        """
        self.bertscore = load("bertscore")
        
    def calc_score(self, sentA, sentB):
        """
        Calculates semantic sim..

        :param sentA: The first sentence.
        :type sentA: str
        :param sentB: The second sentence.
        :type sentB: str
        :return: The BARTScore fluency score.
        :rtype: float
        """
        sentA, sentB = format_sentence(sentA), format_sentence(sentB)
        results = self.bertscore.compute(predictions=[sentA], references=[sentB], lang="en", 
                                         use_fast_tokenizer=True)
        return results['f1'][0]

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)
    