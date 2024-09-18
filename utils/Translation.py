import translate
import string
from deep_translator import GoogleTranslator
from transformers import pipeline


class Translation():
    def __init__(self, from_lang='vi', to_lang='en', mode='deep_translator'):
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang
        self.translator = None

        if self.__mode == 'deep_translator':
            self.translator = GoogleTranslator(source=self.__from_lang, target=self.__to_lang)
        else:
            self.translator = translate.Translator(from_lang=self.__from_lang, to_lang=self.__to_lang)

    def preprocessing(self, text):
        if isinstance(text, str):
            return text.lower()
        return text
    
    def __call__(self, text):
        text = self.preprocessing(text)
        if not isinstance(text, str):
            raise ValueError('Input text should be a string')
        return self.translator.translate(text)

class TextPreprocessing():
    def __init__(self, stopWords_path='D:/demo-ai-challenge/model/dict/english-stopwords-dash.txt'):
        with open(stopWords_path, 'r') as f:
            lines = f.readlines()
        self.stop_words = set(line.strip() for line in lines)

    def remove_stop_words(self, text):
        if isinstance(text, str):
            tokens = text.split()
            return " ".join([w for w in tokens if w not in self.stop_words])
        return text
    
    def lower_casing(self, text):
        if isinstance(text, str):
            return text.lower()
        return text
    
    def upper_casing(self, text):
        if isinstance(text, str):
            return text.upper()
        return text
    
    def remove_punctuation(self, text):
        if isinstance(text, str):
            return text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def text_norm(self, text):
        if isinstance(text, str):
            text = self.remove_punctuation(text)
            return text
        return text
    
    def __call__(self, text):
        text = self.lower_casing(text)
        text = self.remove_stop_words(text)
        text = self.text_norm(text)

        return text