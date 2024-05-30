import warnings
from .base import AttackMetric
import numpy as np
from ...tags import *
from ...data_manager import DataManager
from ...utils import format_sentence

## TODO use a pytorch model instead

class UniversalSentenceEncoder(AttackMetric):

    NAME = "Semantic Similarity"

    TAGS = { TAG_English }

    def __init__(self):
        """
        Universal Sentence Encoder in tensorflow_hub.
        `[pdf] <https://arxiv.org/pdf/1803.11175>`__
        `[page] <https://tfhub.dev/google/universal-sentence-encoder/4>`__

        :Data Requirements: :py:data:`.AttackAssist.UniversalSentenceEncoder`
        :Package Requirements:
            * **tensorflow** >= 2.0.0
            * **tensorflow_hub**
        :Language: english
        
        """
        
        import tensorflow_hub as hub
        
        self.embed = hub.load( DataManager.load("AttackAssist.UniversalSentenceEncoder") )

    def calc_score(self, sentA : str, sentB : str) -> float:
        """
        Args:
            sentA: The first sentence.
            sentB: The second sentence.

        Returns:
            Cosine distance between two sentences.
        
        """
        sentA, sentB = format_sentence(sentA), format_sentence(sentB)
        ret = self.embed([sentA, sentB]).numpy()
        sim = ret[0].dot(ret[1]) / (np.linalg.norm(ret[0]) * np.linalg.norm(ret[1]))
        if hasattr(sim, '__len__'):
            warnings.warn(f"Warning: USE returned a vector {sim}instead of a scalar. Returning the first element.")
            return sim[0]
        return sim

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)
