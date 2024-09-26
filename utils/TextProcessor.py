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
    
    def remove_stopwords(self, text:str):
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return filtered_text
    
    def stemming(self, tokens):
        stem_text = [self.porter_stemmer(token) for token in tokens]
        return stem_text

    def lemmatizer(self, tokens):
        lemm_text = [self.wordnet_lemmatizer.lemmatize(token) for token in tokens]
        return lemm_text


    def __call__(self, text:str):
        text = self.lowercase(text)
        text = self.convert_number(text)
        text = self.remove_punctuation(text)
        text = self.remove_whitespace(text)
        tokens = self.remove_stopwords(text)
        tokens = self.stemming(tokens)
        tokens = self.lemmatizer(tokens)
        return " ".join(tokens)