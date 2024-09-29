import string, re, inflect, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt_tab')
# nltk.download('woednet')
# nltk.download('stopwords')

class TextProcessor():
    def __init__(self):
        self.re = re
        self.p = inflect.engine()
        self.word_tokenize = word_tokenize
        self.stop_words = set(stopwords.words("english"))
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.porter_stemmer = PorterStemmer()

    def lowercase(self, text:str):
        return text.lower()
    
    def convert_number(self, text:str):
        temp_str = text.split()
        new_string = []

        for word in temp_str:
            if word.isdigit():
                temp = self.p.number_to_words(word)
                new_string.append(temp)
            else:
                new_string.append(word)

        return " ".join(new_string)

    def remove_punctuation(self, text:str):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_whitespace(self, text:str):
        return ' '.join(text.split())
    
    def tokenize(self, text:str):
        word_tokens = self.word_tokenize(text)
        return word_tokens
    
    def remove_stopwords(self, tokens):
        filtered_text = [token for token in tokens if token not in self.stop_words]
        return filtered_text
    
    def stemming(self, tokens):
        stem_text = [self.porter_stemmer.stem(token) for token in tokens]
        return stem_text

    def lemmatizer(self, tokens):
        lemm_text = [self.wordnet_lemmatizer.lemmatize(token) for token in tokens]
        return lemm_text


    def __call__(self, text:str):
        text = self.remove_whitespace(text)
        text = self.remove_punctuation(text)
        text = self.convert_number(text)
        text = self.lowercase(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stemming(tokens)
        tokens = self.lemmatizer(tokens)
        return " ".join(tokens)