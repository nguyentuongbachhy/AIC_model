import re, spacy, nltk
from nltk.corpus import wordnet
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
# nltk.download("wordnet")


class TextProcessor():
    def __init__(self):
        self.re = re
        self.nlp = nlp
        self.synsets = wordnet.synsets
        self.stopwords = STOP_WORDS
    
    def normalize_text(self, text:str):
        text= text.lower()
        text = self.re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_stopwords(self, text:str):
        tokens = text.split()
        filtered_tokens = [
            word for word in tokens if word not in self.stopwords
        ]
        return " ".join(filtered_tokens)

    def lemmatize_text(self, text:str):
        doc = self.nlp(text)
        lemmatized = [token.lemma_ for token in doc]
        return " ".join(lemmatized)
    
    def get_synonyms(self, word:str):
        synonyms = []
        for syn in self.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        return synonyms[0] if synonyms else word
    
    def replace_with_synonyms(self, text:str):
        tokens = text.split()
        synonym_replaced = [
            self.get_synonyms(word) for word in tokens
        ]
        return " ".join(synonym_replaced)
    
    def __call__(self, text:str):
        text = self.normalize_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        text = self.replace_with_synonyms(text)
        return text
