import os
import numpy as np
from nltk.corpus import stopwords
import nltk.data
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re
import streamlit as st
import plotly.graph_objects as go

from string import punctuation

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stoplist = stopwords.words('english')


def word_tokenize_custom(text):
    return [w.strip(punctuation) for w in word_tokenize(text)]
class Ranker:

    def __init__(self, folder_name, dataset, suffix='.story', n_docs=5000, stoplist=stopwords.words('english'),
                 smooth_idf=True, sentence_tokenize=tokenizer.tokenize, word_tokenize=word_tokenize_custom):
        self.sentence_tokenize = sentence_tokenize
        self.word_tokenize = word_tokenize#lambda text: [w.strip(punctuation) for w in word_tokenize(text)]
        self.n_docs = n_docs
        self.stoplist = []
        for word in stoplist:
            for token in self.word_tokenize(word):
                self.stoplist.append(token)  # (re.sub('^[^a-zA-Z]*|[^a-zA-Z]*$','',token))
                self.stoplist.append(token.strip(punctuation))
        self.stoplist = set(self.stoplist)
        if dataset.lower() == 'cnn':
            filenames = (filename for filename in os.listdir(folder_name) if filename.endswith(suffix))
            lemmatized_texts = (
            self.lemmatize(open("{}/{}".format(folder_name, filename)).read().split("@highlight", 1)[0]) for filename in
            filenames)
        else:
            newsroom_data = self.read_newsroom_data(folder_name)
            lemmatized_texts = (self.lemmatize(data['text']) for data in newsroom_data)


        self.smooth_idf = smooth_idf

        count_vect = self.get_new_countvect()
        self.docs_word_freq = count_vect.fit_transform(lemmatized_texts).toarray()  # (docs, words)
        self.docs_word_freq = np.where(self.docs_word_freq > 0, 1, 0)
        self.docs_word_freq = np.sum(self.docs_word_freq, axis=0)
        # self.docs_word_freq /= np.sum(self.docs_word_freq)
        self.docs_vocab = count_vect.vocabulary_


    def get_new_countvect(self):#preprocessor=lambda x: re.sub(r'(\d[\d\.])+', '', x.lower())
        return CountVectorizer(tokenizer=self.word_tokenize, stop_words=self.stoplist)

    def process_text(self, text):
        ori_text_nodouble_newline = re.sub(r'\n+', '\n', text).strip()
        sentences = []
        for paragraph in ori_text_nodouble_newline.split('\n'):
            sentences.extend(self.sentence_tokenize(paragraph))  # self.sentence_tokenize(ori_text) self.sentence_tokenize(text)
        n_sentences = len(sentences)


        text = text.lower()
        lemmatized_text = self.lemmatize(text)
        word_count_vectorizer = self.get_new_countvect()
        word_count = word_count_vectorizer.fit_transform([lemmatized_text]).toarray()[0]
        vocab = word_count_vectorizer.vocabulary_

        text_info = dict()
        text_info['word_freq'] = word_count / np.sum(word_count)
        text_info['vocab'] = vocab
        text_info['n_vocab'] = len(vocab)
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
                idf = np.log((1 + self.n_docs) / (1 + self.docs_word_freq[self.docs_vocab[word]])) + 1
            tfidf[idx] = tf * idf

        return normalize([tfidf])[0]
    def get_keywords(self,tfidf,vocab,n):  ## bug when n > num of words
        tfidf_tup = [(i,val) for i,val in enumerate(tfidf)]
        tfidf_tup = sorted(tfidf_tup,key=lambda tup:tup[1],reverse=True)
        keywords = []
        indices = [tfidf_tup[i][0] for i in range(n)]

        for idx in indices:
            for word in vocab:
                if vocab[word] == idx:
                    keywords.append(word)
                    break
            else:
                continue
            continue
        return keywords



    def get_score(self, sentence, text_info, words_already_in_summ, k, m, min_sentence_len):

        sentence = sentence.lower()

        tfidf_vect, vocab = text_info['tfidf'], text_info['vocab']
        words = [word for word in self.word_tokenize(sentence) if
                 (word not in self.stoplist) and (re.search('[a-zA-Z]', word) is not None)]
        words = [self.lemmatize(word, is_word=True) for word in words]

        n_words = 0
        score = 0
        for word in words:
            if (word not in vocab) or not re.search('[a-zA-Z]', word): # word doesn't contain any alpha letter
                continue
            n_words += 1
            idx = vocab[word]
            tfidf = tfidf_vect[idx]
            if word in words_already_in_summ:
                tfidf *= k
            score += tfidf

        if n_words < min_sentence_len:
            if n_words == 0:
                return 0
            return (score / n_words) * m
        words_already_in_summ.update(words)
        return score / n_words
   # def get_keywords(self,text_info,n):


    def rank_sentences(self, text, k=0.5, min_sentence_len=4, m=0.3,n_keywords=5): #keyword
        text_info = self.process_text(text)
        n_keywords = min(n_keywords,text_info['n_vocab'])
        # keywords = self.get_keywords(text,tfidf,vocab)
        keywords = self.get_keywords(text_info['tfidf'], text_info['vocab'], n_keywords)
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
        return selected_sentences,keywords

    def lemmatize(self, text, is_word=False):
        '''text is lowercased.'''
        '''word=False : lemmatize text to put in word count calculator'''
        def get_wordnet_pos(_word):
            """Map POS tag to first character lemmatize() accepts"""
            nltk.pos_tag([_word])
            tag = nltk.pos_tag([_word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        text = text.lower()
        if is_word:
            if (text in self.stoplist) or (not text):
                return text
            return WordNetLemmatizer().lemmatize(text,get_wordnet_pos(text))
        words = self.word_tokenize(text)
        return ' '.join([WordNetLemmatizer().lemmatize(word,get_wordnet_pos(word)) for word in words if ((word not in self.stoplist) and word) ])


class Displayer:
    @staticmethod
    def show_custom(info, n_sentences):
        text = info['text']#.replace('``', '""')
        top_n_my_model = info['my_model_result'][:n_sentences]

        st.subheader('Our Summary')
        sorted_summ = sorted(top_n_my_model, key=lambda sentence: sentence[0])
        for sentence in sorted_summ:
            st.markdown('{}\n'.format(sentence[1]))
        st.subheader('Keywords')
        st.markdown(', '.join(info['keywords']))
        sentences_group = dict()
        sentences_group[0] = [sentence[1] for sentence in top_n_my_model]
        highlighted_text = Highlighter.get_highlighted_html(text, sentences_group)
        st.subheader('Highlights Displayed')
        st.markdown(highlighted_text, unsafe_allow_html=True)

    @staticmethod
    def display_figure(x,y,title,xaxis_title,yaxis_title,mode='lines+markers'):

        fig = go.Figure(data=go.Scatter(x=x, y=y, mode=mode ))
        fig.update_layout(title=title,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        st.plotly_chart(fig)
    @staticmethod
    def show(info, n_sentences,benchmark=False,pos=0):

        text = info['text']#.replace('``', '""')
        top_n_my_model = info['my_model_result'][:n_sentences] #self.cnn_tfidf_ret[doc_i][:self.n_sentences]
        top_n_presumm = info['presumm_result'][:n_sentences]
        top_n_presumm = sorted(top_n_presumm, key=lambda sentence: text.find(sentence))
        sentences_group = dict()
        #.replace('``', '""'
        sentences_group[0] = [sentence[1] for sentence in top_n_my_model] # mine
        sentences_group[1] = [sentence for sentence in top_n_presumm]


        highlighted_text = Highlighter.get_highlighted_html(text,sentences_group) if not benchmark else text

        if 'title' in info:
            st.header(info['title'])
        st.markdown(highlighted_text.replace('``', '""'), unsafe_allow_html=True)
        xs = list(range((len(info['my_model_result']))))
        sortedByPos = sorted(info['my_model_result'], key=lambda tup: tup[0])
        st.subheader('Reference Summary')
        st.markdown('{}'.format(
            info['ref_summary']))
        sorted_summ = sorted(top_n_my_model, key=lambda sentence: sentence[0])
        if not benchmark:
            Displayer.display_figure(x=xs, y=[tup[2] for tup in sortedByPos], title='score by position',
                                 xaxis_title='position', yaxis_title='score')

        if benchmark:
            if pos==0:
                st.subheader(0)
                for sentence in sorted_summ:
                    st.markdown('{}\n'.format(sentence[1]))
                st.subheader(1)
                for sentence in top_n_presumm:
                    st.markdown('{}\n'.format(sentence))
            else:
                st.subheader(0)
                for sentence in top_n_presumm:
                    st.markdown('{}\n'.format(sentence))
                st.subheader(1)
                for sentence in sorted_summ:
                    st.markdown('{}\n'.format(sentence[1]))
        else:
            st.subheader('Our Summary')

            for sentence in sorted_summ:
                st.markdown('{}\n'.format(sentence[1]))
            st.subheader('Keywords')
            #for word in info['keywords']:
            st.markdown(', '.join(info['keywords']))
            st.subheader('PreSumm\'s Summary')
            for sentence in top_n_presumm:
                st.markdown('{}\n'.format(sentence))



class Highlighter:

    @staticmethod
    def get_highlighted_html(text,sentences_group, intersect_group_id=2):
        highlighted_indices = Highlighter.get_highlight_indices(text, sentences_group, intersect_group_id)
        return Highlighter.compute_highlighted_text(text, highlighted_indices)

    @staticmethod
    def get_highlight_indices( text, sentences_group, intersect_group_id):
        def has_duplicate( start_end_indices, start_idx):
            for index in start_end_indices:
                if index[0] == start_idx:
                    return True
            return False

        text = text.lower()
        indices_with_group_id = []
        indices_by_groupID = dict()
        for group_id in sentences_group:
            indices_by_groupID[group_id] = set()
            for sentence in sentences_group[group_id]:
                sentence = sentence.lower().strip()
                start_idx = text.find(sentence)
                while has_duplicate(indices_by_groupID[group_id], start_idx):
                    start_idx = text.find(sentence, start_idx + len(sentence))
                if start_idx == -1:
                    continue
                index = (start_idx, start_idx + len(sentence))
                indices_by_groupID[group_id].add(index)
        intersect_indices = set.intersection(*[indices_by_groupID[gid] for gid in sentences_group])
        for group_id in sentences_group:
            indices = indices_by_groupID[group_id]
            for index in indices:
                if index not in intersect_indices:
                    indices_with_group_id.append(index + (group_id,))
        for idx in intersect_indices:
            indices_with_group_id.append(idx + (intersect_group_id,))
        ans =  sorted(indices_with_group_id, key=lambda my_tuple: my_tuple[0])

        return ans

    @staticmethod
    def compute_highlighted_text(text, indices, colors={0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}):
        highlighted_text = ''
        last_pos = 0

        for index in indices:
            start, end, color_id = index
            color_code = colors[color_id]

            highlighted_text += text[last_pos:start] + '<span style="background-color: {}">'.format(color_code) + text[
                                                                                                                  start:end] + '</span>'
            last_pos = end
        highlighted_text += text[last_pos:]
        return highlighted_text







class Highlighter:

    @staticmethod
    def get_highlighted_html(text,sentences_group, intersect_group_id=2):
        highlighted_indices = Highlighter.get_highlight_indices(text, sentences_group, intersect_group_id)
        return Highlighter.compute_highlighted_text(text, highlighted_indices)

    @staticmethod
    def get_highlight_indices( text, sentences_group, intersect_group_id):
        def has_duplicate( start_end_indices, start_idx):
            for index in start_end_indices:
                if index[0] == start_idx:
                    return True
            return False

        text = text.lower()
        indices_with_group_id = []
        indices_by_groupID = dict()
        for group_id in sentences_group:
            indices_by_groupID[group_id] = set()
            for sentence in sentences_group[group_id]:
                sentence = sentence.lower().strip()
                start_idx = text.find(sentence)
                while has_duplicate(indices_by_groupID[group_id], start_idx):
                    start_idx = text.find(sentence, start_idx + len(sentence))
                if start_idx == -1:
                    continue
                index = (start_idx, start_idx + len(sentence))
                indices_by_groupID[group_id].add(index)
        intersect_indices = set.intersection(*[indices_by_groupID[gid] for gid in sentences_group])
        for group_id in sentences_group:
            indices = indices_by_groupID[group_id]
            for index in indices:
                if index not in intersect_indices:
                    indices_with_group_id.append(index + (group_id,))
        for idx in intersect_indices:
            indices_with_group_id.append(idx + (intersect_group_id,))
        ans =  sorted(indices_with_group_id, key=lambda my_tuple: my_tuple[0])

        return ans

    @staticmethod
    def compute_highlighted_text(text, indices, colors={0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}):
        highlighted_text = ''
        last_pos = 0

        for index in indices:
            start, end, color_id = index
            color_code = colors[color_id]

            highlighted_text += text[last_pos:start] + '<span style="background-color: {}">'.format(color_code) + text[
                                                                                                                  start:end] + '</span>'
            last_pos = end
        highlighted_text += text[last_pos:]
        return highlighted_text



