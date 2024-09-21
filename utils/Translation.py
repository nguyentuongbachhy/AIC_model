import translate
from deep_translator import GoogleTranslator

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