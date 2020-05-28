import streamlit as st
import numpy as np
import pickle
from utils import *
from utils_smart_stat import *

st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Simple tf-idf", "Smart stat with tf-idf"])
colors_dict = {0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}

def main():
    st.title('Rank sentences by {}'.format(app_mode))
    if app_mode == 'Simple tf-idf':

        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">My summary</span>'.format(colors_dict[0]),
            unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">Their summary</span>'.format(colors_dict[1]),
            unsafe_allow_html=True)
        st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),
                    unsafe_allow_html=True)
        run_app_simple()
    elif app_mode == 'Smart stat with tf-idf':
        run_app_smart()



def run_app_smart():

    #presentation.set_dataset(dataset)
    dataset = st.sidebar.radio('Dataset', ( 'CNN', 'Custom Input')) #'Newsroom',
    apply_location = st.sidebar.checkbox("Apply location stat")
    one_minus_k = st.sidebar.slider(
        'keywords: important ---> diverse',
        0.0, 1.0, value=0.8#(25.0, 75.0)
    )
    k = 1-one_minus_k
    m = st.sidebar.slider(
        'm: Select a range of values',
        0.0, 1.0, value=0.15  # (25.0, 75.0)
    )

    n_sentences = st.sidebar.number_input('number of sentences in summary', value=5, min_value=1)
    if dataset != 'Custom Input':
        n_articles = 1000

        article_number = st.sidebar.number_input(label='{} article article number'.format(dataset), value=0,
                                                 format='%d', max_value=n_articles - 1, min_value=0)

    data = load_data()
    ranker = data['smart_ranker']


    if dataset == 'CNN':
        st.sidebar.markdown('or')
        random = st.sidebar.button('random an article')
        if random:
            doc_i = np.random.randint(1000)
        else:
            doc_i = article_number

        st.header('{} {}'.format(dataset, doc_i))

        info = dict()
        info['text'] = data['cnn_ref_text'][doc_i]

        info['my_model_result'] = ranker.rank_sentences(info['text'],k=k,m=m)

        info['presumm_result'] = data['cnn_presumm_ret'][doc_i]
        info['cnn_ref_summary'] = data['cnn_ref_summaries'][doc_i]
        Displayer.show_cnn(info,n_sentences)
        xs = list(range((len(info['my_model_result']))))
        sortedByPos = sorted(info['my_model_result'], key=lambda tup: tup[0])
        Displayer.display_figure(x=xs, y=[tup[2] for tup in sortedByPos], title='score by position',
                                 xaxis_title='position', yaxis_title='score')

    elif dataset == 'Custom Input':

        custom_text = st.text_area('text to be summarized',height=400)
        if  custom_text != '':
            info = dict()
            info['text'] = custom_text
            info['my_model_result'] = ranker.rank_sentences(custom_text,k=k,m=m)
            Displayer.show_custom(info, n_sentences)
            xs = list(range((len(info['my_model_result']))))
            sortedByPos = sorted(info['my_model_result'],key=lambda tup: tup[0])
            Displayer.display_figure(x=xs,y=[tup[2] for tup in sortedByPos],title='score by position',xaxis_title='position',yaxis_title='score')


def run_app_simple():

    data = load_data()
    presentation = Presentation(data['cnn_ref_text'],data['cnn_ref_summaries'],data['cnn_tfidf_ret'],data['cnn_presumm_ret'],data['presumm_newsroom'],data['tfidf_newsroom'],data['ref_newsroom_data'])

    dataset = st.sidebar.radio('Dataset',('Newsroom','CNN'))
    presentation.set_dataset(dataset)
    n_sentences = st.sidebar.number_input('number of sentences in summary',value=5,min_value=1)
    presentation.set_n_sentences(n_sentences)
    n_articles = 1000

    article_number = st.sidebar.number_input(label='{} article article number'.format(dataset),value=0,
        format='%d',max_value=n_articles-1,min_value=0)

    st.sidebar.markdown('or')
    random = st.sidebar.button('random an article')

    if random:
        doc_i = np.random.randint(1000)

        # print(type(doc_i))
        presentation.show(doc_i)

    else:

        # print(type(article_number))
        presentation.show(article_number)
@st.cache(allow_output_mutation=True)
def load_data():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
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