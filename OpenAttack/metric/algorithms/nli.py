import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from .base import AttackMetric
from ...tags import *
from ...data_manager import DataManager
from ...utils import format_sentence


class EntailmentProbability(AttackMetric):
    NAME = "Entailment Probability"
    TAGS = { TAG_English }

    def __init__(self, model_name_or_path="howey/electra-small-mnli"):
        """
        Entailment Probability metric based on a pretrained NLI model.

        :param model_name_or_path: The name or path of a pretrained NLI model.
        :type model_name_or_path: str
        :Data Requirements: :py:data:`.AttackAssist.EntailmentProbability.model_name_or_path`
        :Package Requirements: transformers
        :Language: english
        """
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        self.model.eval()
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        if   model_name_or_path  == "microsoft/deberta-base-mnli"     :   self.entail_label = 2
        elif model_name_or_path  == "howey/electra-small-mnli"        :   self.entail_label = 0
        else:  self.entail_label = self.model.config.label2id["entailment"]

    def calc_score(self, sentA, sentB):
        """
        Calculates the probability of entailment between two sentences.

        :param sentA: The first sentence.
        :type sentA: str
        :param sentB: The second sentence.
        :type sentB: str
        :return: The probability of entailment.
        :rtype: float
        """
        sentA, sentB = format_sentence(sentA), format_sentence(sentB)
        inputs = self.tokenizer(sentA, sentB, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            prob = logits.softmax(1)[0][self.entail_label].item()
        return prob

    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            return self.calc_score(input["x"], adversarial_sample)
