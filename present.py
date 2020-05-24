import streamlit as st
import numpy as np
import pickle
from utils import *
from utils_smart_stat import *

st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Simple tf-idf", "Smart stat with tf-idf", "Show the source code"])
colors_dict = {0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}

def main():
    st.title('Rank sentences by {}'.format(app_mode))
    st.markdown('<p style="font-size:17px"><span style="background-color: {}">My summary</span>'.format(colors_dict[0]),
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:17px"><span style="background-color: {}">Their summary</span>'.format(colors_dict[1]),
        unsafe_allow_html=True)
    st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),
                unsafe_allow_html=True)
    if app_mode == 'Simple tf-idf':
        run_app_simple()
    elif app_mode == 'Smart stat with tf-idf':
        run_app_smart()



def run_app_smart():

    #presentation.set_dataset(dataset)
    dataset = st.radio('Dataset', ('Newsroom', 'CNN'))
    n_sentences = st.number_input('number of sentences in summary', value=5, min_value=1)
    #presentation.set_n_sentences(n_sentences)
    n_articles = 1000

    article_number = st.number_input(label='{} article article number'.format(dataset), value=0,
                                     format='%d', max_value=n_articles - 1, min_value=0)

    st.markdown('or')
    random = st.button('random an article')
    data = load_data()
    ranker = data['smart_ranker']

    # class Displayer:
    #     @staticmethod
    #     def show_cnn(info, n_sentences):
    #
    # # info['text'] == cnn_ref_text[doc_i]
    # # info['my_model_result'] == cnn_tfidf_ret[doc_i]
    # # info['presumm_result'] == cnn_presumm_ret[doc_i]
    # # info['cnn_ref_summary'] == cnn_ref_summaries[doc_i]

    if random:

        doc_i = np.random.randint(1000)
        st.header('{} {}'.format(dataset, doc_i))
        # print(type(doc_i))
        info = dict()
        info['text'] = data['cnn_ref_text'][doc_i]
        info['my_model_result'] = ranker.rank_sentences(info['text'])

        print(info['my_model_result'])
        info['presumm_result'] = data['cnn_presumm_ret'][doc_i]
        info['cnn_ref_summary'] = data['cnn_ref_summaries'][doc_i]
        Displayer.show_cnn(info,n_sentences)
    else:
        print(type(article_number))
        #presentation.show(article_number,n_sentences)

def run_app_simple():

    # cnn_ref_text, cnn_ref_summaries, cnn_tfidf_ret, cnn_presumm_ret,
    # presumm_newsroom, tfidf_newsroom, ref_newsroom_data

    data = load_data()
    # cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,
    #              presumm_newsroom,tfidf_newsroom,ref_newsroom_data
    presentation = Presentation(data['cnn_ref_text'],data['cnn_ref_summaries'],data['cnn_tfidf_ret'],data['cnn_presumm_ret'],data['presumm_newsroom'],data['tfidf_newsroom'],data['ref_newsroom_data'])
    #
    # st.title('Rank sentences by TF-IDF')
    # st.markdown('<p style="font-size:17px"><span style="background-color: {}">My summary</span>'.format(colors_dict[0]),unsafe_allow_html=True)
    # st.markdown('<p style="font-size:17px"><span style="background-color: {}">Their summary</span>'.format(colors_dict[1]),unsafe_allow_html=True)
    # st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),unsafe_allow_html=True)

    dataset = st.radio('Dataset',('Newsroom','CNN'))
    presentation.set_dataset(dataset)
    n_sentences = st.number_input('number of sentences in summary',value=5,min_value=1)
    presentation.set_n_sentences(n_sentences)
    n_articles = 1000

    article_number = st.number_input(label='{} article article number'.format(dataset),value=0,
        format='%d',max_value=n_articles-1,min_value=0)

    st.markdown('or')
    random = st.button('random an article')

    if random:
        doc_i = np.random.randint(1000)

        print(type(doc_i))
        presentation.show(doc_i)

    else:

        print(type(article_number))
        presentation.show(article_number)
@st.cache(allow_output_mutation=True)
def load_data():
    # load_data
    # newsroom
    data = dict()
    data['presumm_newsroom'] = pickle.load(open('./data/newsroom_presumm_1000_by_sentence_score','rb'))
    data['tfidf_newsroom'] = pickle.load(open('./data/newsroom_sorted_sentences1000_v2','rb'))
    data['ref_newsroom_data'] = pickle.load(open('./data/newsroom_ref_data1000_pickle','rb'))
    # load_data
    # cnn
    data['cnn_ref_text'] = pickle.load( open('./data/cnn_ref_text','rb'))
    data['cnn_ref_summaries'] = pickle.load(open('./data/cnn_ref_summaries','rb'))
    data['cnn_tfidf_ret'] = pickle.load(open('./data/cnn_tfidf_ret','rb'))
    data['cnn_presumm_ret'] = pickle.load(open('./data/cnn_presumm_1084_by_sentence_score','rb'))#pickle.load(open('./data/cnn_presumm_ret','rb'))
    # simple_presentation = Presentation(cnn_ref_text, cnn_ref_summaries, cnn_tfidf_ret, cnn_presumm_ret,
    #                             presumm_newsroom, tfidf_newsroom, ref_newsroom_data)
    data['smart_ranker'] = pickle.load(open('./data/ranker.pkl','rb'))
    return data #[cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,presumm_newsroom,tfidf_newsroom,ref_newsroom_data]

if __name__ == "__main__":
    main()