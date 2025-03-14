import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.tokenize import word_tokenize
import contractions
import unicodedata
import chardet
from unidecode import unidecode
import emoji
from bs4 import BeautifulSoup

class textPreprocessing():

    def __init__(self):
        pass
    
    def preprocess_text(self, text, keep_first_n=0, keep_last_n=0,
                    stemming=False, lemmatize=False, stop_words=True, languages=['english']):
    
        # Step 1: Noisy Entity Removal
        text = self.process_text_encoding(text.encode('utf-8'))  # Fix encoding issue
        text = self.remove_urls(text)  # Remove URLs

        # Step 2: Keep First and Last N Lines (If Required)
        if keep_first_n != 0 and keep_last_n == 0:
            text = self.keep_first_n_lines(text, n=keep_first_n)
        elif keep_first_n == 0 and keep_last_n != 0:
            text = self.keep_last_n_lines(text, n=keep_last_n)
        elif keep_first_n != 0 and keep_last_n != 0:
            first_part = self.keep_first_n_lines(text, n=keep_first_n)
            last_part = self.keep_last_n_lines(text, n=keep_last_n)
            text = first_part + ' \n ' + last_part

        # Step 3: Text Normalization
        text = self.remove_html_tags(text)  # Remove HTML tags
        text = self.expand_contractions(text)  # Expand contractions
        text = self.to_lowercase(text)  # Convert to lowercase

        if stop_words:
            for language in languages:
                text = self.remove_stopwords(text, language)  # Remove stopwords

        text = self.remove_punctuation(text)  # Remove punctuation
        text = self.remove_digits(text)  # Remove numbers
        
        if stemming:
            text = self.apply_stemming(text)  # Apply stemming
        elif lemmatize:
            text = self.lemmatize_text_nltk(text)  # Apply lemmatization
    
        # Step 4: Word Standardization
        text = self.remove_emojis(text)  # Remove emojis
        text = self.replace_special_symbols(text)  # Replace special symbols
        text = self.remove_non_alphanumeric(text)  # Remove non-alphanumeric characters

        text = self.normalize_whitespace(text)

        return text

    def to_lowercase(self, text):
        return text.lower()

    def process_text_encoding(self, text_bytes):
        detected_encoding = chardet.detect(text_bytes)['encoding']
        decoded_text = text_bytes.decode(detected_encoding, errors='ignore')  # Ignore errors
        normalized_text = unidecode(decoded_text)  # Convert to ASCII-friendly text
        return normalized_text

    def remove_punctuation(self, text):
        text = re.sub(r'[^\w\s]', '', text) 
        return text

    def expand_contractions(self, text):
        return contractions.fix(text)

    def remove_digits(self, text):
        return re.sub(r'\d+', '', text)

    def keep_first_n_lines(self, text, n=1):
        # Normalize HTML line breaks by replacing <br> and <br /> with '\n'
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Split by '\n' and extract the first n lines
        lines = text.split('\n')
        return '\n'.join(lines[:n]) if len(lines) >= n else text

    def keep_last_n_lines(self, text, n=1):
        # Normalize HTML line breaks by replacing <br> and <br /> with '\n'
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        # Split by '\n' and extract the last n lines
        lines = text.split('\n')
        return '\n'.join(lines[-n:]) if len(lines) >= n else text


    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def apply_stemming(self, text):
        stemmer = PorterStemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatize_text_nltk(self, text):
        """
        Lemmatizes the words in the given string using NLTK's WordNetLemmatizer.
        """
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(text)  # Tokenize the text into words
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]  # Lemmatize each word
        return ' '.join(lemmatized_words)

    def remove_stopwords(self, text, language='english'):

        # Load the stopwords for the specified language
        stop_words = set(stopwords.words(language))
        
        # Tokenize the text into words
        words = text.split()
        
        # Filter out stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Rejoin the filtered words into a single string
        return ' '.join(filtered_words)

    def normalize_whitespace(self, text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        return text.strip()

    def replace_special_symbols(self, text):
        text = text.replace("&", " and ")
        text = text.replace("@", " at ")
        return text

    def remove_non_alphanumeric(self, text):
        return re.sub(r'[^A-Za-z0-9\s]', '', text)

    def remove_emojis(self, text):
        return emoji.replace_emoji(text, replace='')  # Removes emojis completely

    def remove_html_tags(self, text):
        return BeautifulSoup(text, "html.parser").get_text()