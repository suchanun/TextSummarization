import streamlit as st
import re


class Presentation:
    def __init__(self,cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,
                 presumm_newsroom,tfidf_newsroom,ref_newsroom_data,n_sentences=5,colors= {0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}):
        self.cnn_ref_text = cnn_ref_text
        self.cnn_ref_summaries = cnn_ref_summaries
        self.cnn_tfidf_ret = cnn_tfidf_ret
        self.cnn_presumm_ret = cnn_presumm_ret
        self.presumm_newsroom = presumm_newsroom
        self.tfidf_newsroom = tfidf_newsroom
        self.ref_newsroom_data = ref_newsroom_data
        self.colors = colors
        self.n_sentences = n_sentences
        self.dataset = ''
    def set_n_sentences(self, n):
        self.n_sentences = n

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_highlight_indices(self, text, sentences_group, intersect_group_id=2):
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
        return sorted(indices_with_group_id, key=lambda my_tuple: my_tuple[0])


    def get_highlight_html(self,text, indices):
        highlighted_text = ''
        last_pos = 0
        for index in indices:
            start, end, color_id = index
            color_code = self.colors[color_id]
            highlighted_text += text[last_pos:start] + '<span style="background-color: {}">'.format(color_code) + text[
                                                                                                                  start:end] + '</span>'
            last_pos = end
        highlighted_text += text[last_pos:]
        return highlighted_text


    def show(self, doc_i):
        st.header('{} {}'.format(self.dataset, doc_i))
        if self.dataset == 'Newsroom':
            self.show_newsroom(doc_i)
        else:
            self.show_cnn(doc_i)


    def show_newsroom(self, i):
        text = self.ref_newsroom_data[i]['text']
        top_n_tfidf = self.tfidf_newsroom[i][:self.n_sentences]
        top_n_presumm = self.presumm_newsroom[i][:self.n_sentences]
        top_n_presumm = sorted(top_n_presumm,key=lambda sentence: text.find(sentence))

        sentences_group = dict()
        sentences_group[0] = [sentence[1] for sentence in top_n_tfidf]
        sentences_group[1] = top_n_presumm

        highlighted_indices = self.get_highlight_indices(text, sentences_group)
        highlighted_text = self.get_highlight_html(text, highlighted_indices)

        st.markdown(highlighted_text, unsafe_allow_html=True)


        st.subheader('Reference Summary')
        st.markdown(re.sub("(.{64})", "\\1\n", self.ref_newsroom_data[i]['summary'], 0, re.DOTALL))

        st.subheader('My Summary')

        sorted_tfidf = sorted(top_n_tfidf, key=lambda sentence: sentence[0])
        for sentence in sorted_tfidf:
            st.markdown('{}\n'.format(sentence[1]))
        st.subheader('PreSumm Summary')
        for sentence in top_n_presumm:
            st.markdown('{}\n'.format(sentence))


    def show_cnn(self, doc_i):
        text = self.cnn_ref_text[doc_i].replace('``', '""')
        top_n_tfidf = self.cnn_tfidf_ret[doc_i][:self.n_sentences]
        top_n_presumm = self.cnn_presumm_ret[doc_i][:self.n_sentences]
        top_n_presumm = sorted(top_n_presumm, key=lambda sentence: text.find(sentence))
        sentences_group = dict()
        sentences_group[0] = [sentence[1].replace('``', '""') for sentence in top_n_tfidf]
        sentences_group[1] = [sentence.replace('``', '""') for sentence in top_n_presumm]
        highlighted_indices = self.get_highlight_indices(text, sentences_group)
        highlighted_text = self.get_highlight_html(text, highlighted_indices)

        st.markdown(highlighted_text, unsafe_allow_html=True)

        st.subheader('Reference Summary')
        st.markdown('{}'.format(
            self.cnn_ref_summaries[doc_i]))
        st.subheader('My Summary')

        sorted_summ = sorted(top_n_tfidf, key=lambda sentence: sentence[0])
        for sentence in sorted_summ:
            st.markdown('{}\n'.format(sentence[1]))

        st.subheader('PreSumm Summary')
        for sentence in top_n_presumm:
            st.markdown('{}\n'.format(sentence))