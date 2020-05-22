import streamlit as st
import numpy as np
import pickle
from utils import *

# load_data
# newsroom
presumm_newsroom = pickle.load(open('./data/newsroom_presumm_1000_by_sentence_score','rb'))
tfidf_newsroom = pickle.load(open('./data/newsroom_sorted_sentences1000_v2','rb'))
ref_newsroom_data = pickle.load(open('./data/newsroom_ref_data1000_pickle','rb'))

# load_data
# cnn
cnn_ref_text = pickle.load( open('./data/cnn_ref_text','rb'))
cnn_ref_summaries = pickle.load(open('./data/cnn_ref_summaries','rb'))
cnn_tfidf_ret = pickle.load(open('./data/cnn_tfidf_ret','rb'))
cnn_presumm_ret = pickle.load(open('./data/cnn_presumm_1084_by_sentence_score','rb'))#pickle.load(open('./data/cnn_presumm_ret','rb'))

colors_dict = {0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}

presentation = Presentation(cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,
                 presumm_newsroom,tfidf_newsroom,ref_newsroom_data)

st.title('Rank sentences by TF-IDF')
st.markdown('<p style="font-size:17px"><span style="background-color: {}">My summary</span>'.format(colors_dict[0]),unsafe_allow_html=True)
st.markdown('<p style="font-size:17px"><span style="background-color: {}">Their summary</span>'.format(colors_dict[1]),unsafe_allow_html=True)
st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),unsafe_allow_html=True)
# st.markdown('<p style="color:blue; font-size:18px">This is a paragraph.</p>',unsafe_allow_html=True)

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
    presentation.show(doc_i)

else:
    presentation.show(article_number)
