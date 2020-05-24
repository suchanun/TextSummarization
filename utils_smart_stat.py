# pickle.dump(presum_filenames,open('Presentation/data/cnn_filenames_1084.pkl','wb'))
import os
import numpy as np
from nltk.corpus import stopwords
import nltk.data
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re

print(nltk.__version__)
nltk.download('stopwords')
stoplist = stopwords.words('english')
class Ranker:
    '''for cnn data'''

    def __init__(self, folder_name, dataset, suffix='.story', n_docs=5000, stoplist=stopwords.words('english'),
                 smooth_idf=True, sentence_tokenize=tokenizer.tokenize, word_tokenize=word_tokenize):
        if dataset.lower() == 'cnn':
            filenames = (filename for filename in os.listdir(folder_name) if filename.endswith(suffix))
            lemmatized_texts = (
            self.lemmatize(open("{}/{}".format(folder_name, filename)).read().split("@highlight", 1)[0]) for filename in
            filenames)
        else:
            newsroom_data = self.read_newsroom_data(folder_name)
            lemmatized_texts = (self.lemmatize(data['text']) for data in newsroom_data)

        self.sentence_tokenize = sentence_tokenize
        self.word_tokenize = word_tokenize
        self.n_docs = n_docs
        self.stoplist = stoplist
        self.smooth_idf = smooth_idf

        count_vect = self.get_new_countvect()
        self.docs_word_freq = count_vect.fit_transform(lemmatized_texts).toarray()  # (docs, words)
        self.docs_word_freq = np.where(self.docs_word_freq > 0, 1, 0)
        self.docs_word_freq = np.sum(self.docs_word_freq, axis=0)
        # self.docs_word_freq /= np.sum(self.docs_word_freq)
        self.docs_vocab = count_vect.vocabulary_

    def get_vocabs(self):
        return self.vocab

    def get_new_countvect(self):
        return CountVectorizer(tokenizer=self.word_tokenize,
                               preprocessor=lambda x: re.sub(r'(\d[\d\.])+', '', x.lower()), stop_words=self.stoplist)

    def process_text(self, text):
        text = text.lower()
        lemmatized_text = self.lemmatize(text)
        word_count_vectorizer = self.get_new_countvect()
        word_count = word_count_vectorizer.fit_transform([lemmatized_text]).toarray()[0]
        vocab = word_count_vectorizer.vocabulary_
        sentences = self.sentence_tokenize(text)
        n_sentences = len(sentences)
        text_info = dict()
        #         text_info['word_count'] = word_count
        text_info['word_freq'] = word_count / np.sum(word_count)
        print(text_info['word_freq'].shape)
        text_info['vocab'] = vocab
        text_info['tfidf'] = self.get_tfidf_vect(text_info['word_freq'], text_info['vocab'])
        text_info['sentences'] = sentences
        text_info['n_sentences'] = n_sentences
        return text_info

    def get_tfidf_vect(self, word_freq, vocab):
        d = np.log((1 + self.n_docs) / (1 + 0)) + 1
        tfidf = np.zeros(len(vocab))
        for word in vocab:
            idx = vocab[word]
            tf = word_freq[idx]
            if re.search('[a-zA-Z]', word) is None:
                tf = 0
            if word not in self.docs_vocab:
                idf = d
            else:
                if self.docs_word_freq[self.docs_vocab[word]] > self.n_docs:
                    print(self.docs_word_freq[self.docs_vocab[word]])
                    print(self.n_docs)
                    print(word)
                    print("heyyyyyyyyyyyyyyyy")
                idf = np.log((1 + self.n_docs) / (1 + self.docs_word_freq[self.docs_vocab[word]])) + 1
            tfidf[idx] = tf * idf
            if (tfidf[idx] < 0):
                print(word)
                print('awwwwwwww')
        print(tfidf)
        print(vocab)
        return normalize([tfidf])[0]

    def get_score(self, sentence, text_info, words_already_in_summ, k, m, min_sentence_len):
        # sentence is lowercased.
        tfidf_vect, vocab = text_info['tfidf'], text_info['vocab']
        words = [word for word in self.word_tokenize(sentence) if
                 (word not in stoplist) and (re.search('[a-zA-Z]', word) is not None)]
        words = [self.lemmatize(word, word=True) for word in words]
        print('sentence == {}'.format(sentence))
        print('words = ')
        print(words)
        n_words = 0
        score = 0
        for word in words:
            if word not in vocab:
                continue
            n_words += 1
            idx = vocab[word]
            tfidf = tfidf_vect[idx]
            if word in words_already_in_summ:
                tfidf *= k
            score += tfidf

        if n_words < min_sentence_len:
            if n_words == 0:
                print("{} {}".format(0, sentence))
                return 0
            print("---> {} {}".format((score / n_words) * m, sentence))
            return (score / n_words) * m
        words_already_in_summ.update(words)
        return score / n_words

    def rank_sentences(self, text, k=0.5, min_sentence_len=3, m=0.8):
        text_info = self.process_text(text)
        print("text_info['vocab']")
        print(text_info['vocab'])
        sentences = [(i, sentence) for i, sentence in enumerate(text_info['sentences'])]
        selected_sentences = []
        words_in_summ = set()
        n_selected = 0
        while n_selected < text_info['n_sentences']:
            candidate_sentences = [
                (i, sentence, self.get_score(sentence, text_info, words_in_summ, k, m, min_sentence_len)) for
                i, sentence in sentences]
            i, selected_sentence, score = max(candidate_sentences, key=lambda tup: tup[2])
            selected_sentences.append((i, selected_sentence, score))
            sentences.remove((i, selected_sentence))
            words_in_summ.update(self.word_tokenize(selected_sentence))
            n_selected += 1
        return selected_sentences, text_info['tfidf'], text_info['vocab']

    def read_newsroom_data(self, path, n=5000):
        data = []
        with gzip.open(path) as f:
            for i, ln in enumerate(f):
                if i >= n:
                    break
                obj = json.loads(ln)
                data.append(obj)
        return data

    def lemmatize(self, text, word=False):
        '''text is lowercased.'''
        '''word=False : lemmatize text to put in word count calculator'''
        if word:
            if word in self.stoplist:
                return word
            return WordNetLemmatizer().lemmatize(text)
        words = self.word_tokenize(text)
        return ' '.join([WordNetLemmatizer().lemmatize(word) if word not in self.stoplist else word for word in words])



