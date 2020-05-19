import streamlit as st
import numpy as np
import pickle
import re

# load_data
# newsroom
presumm_newsroom = pickle.load(open('./data/newsroom_presumm_1000_dict','rb'))
#tfidf_newsroom = pickle.load(open('./data/newsroom_tfidf_1000_dict','rb'))
tfidf_newsroom = pickle.load(open('./data/newsroom_sorted_sentences1000_v2','rb'))
ref_newsroom_data = pickle.load(open('./data/newsroom_ref_data1000_pickle','rb'))

# load_data
# cnn
cnn_ref_text = pickle.load( open('./data/cnn_ref_text','rb'))
cnn_ref_summaries = pickle.load(open('./data/cnn_ref_summaries','rb'))
cnn_tfidf_ret = pickle.load(open('./data/cnn_tfidf_ret','rb'))
cnn_presumm_ret = pickle.load(open('./data/cnn_presumm_ret','rb'))


# cnn_ret = pickle.load( open( "./data/cnn_tokenized1.ret", "rb" ) )
# cnn_ref_summ = pickle.load(open("./data/ref_summaries1","rb"))
# # tfidfn = pickle.load( open( "./data/tfidf_score_1", "rb" ) )
# cnn_filenames = pickle.load(open('./data/cnn_filenames1','rb'))
# cnn_presum_ret = pickle.load(open('./data/presum_ret_1','rb'))
# presum_ret_filenames = list(cnn_presum_ret.keys())

st.title('rank sentences by TF-IDF')
st.markdown('<p style="font-size:17px">This <span style="background-color: #D0F15F">my summary</span> a <span style="background-color: #90F9E3">their summary</span> <span style="background-color: #E9B2ED">both</span>.</p>hjh',unsafe_allow_html=True)
st.markdown('<p style="color:blue; font-size:18px">This is a paragraph.</p>',unsafe_allow_html=True)
dataset = st.radio('Dataset',('Newsroom','CNN'))
next = st.button('random another article')
n_articles = 1000
n_sentences = 5
number = st.number_input(label='newsroom article number',value=0,
    format='%d',max_value=n_articles-1,min_value=0)
#new_article = st.button('go to article {}'.format(number))

#st.markdown('$ \underline{fd}<h2>Foo <em>bar</em></h2>$')

def get_highlight_indices(text, sentences_group, intersect_group_id=2):
    text = text.lower()
    indices_with_group_id = []
    
    indices_by_groupID = dict()
    for group_id in sentences_group:
        indices_by_groupID[group_id] = set()
        for sentence in sentences_group[group_id]:
            sentence = sentence.lower().strip()
            start_idx = text.find(sentence)
            if start_idx == -1:
                continue
            index = (start_idx, start_idx+len(sentence))
            indices_by_groupID[group_id].add(index)
    intersect_indices = set.intersection(*[indices_by_groupID[gid] for gid in sentences_group])
    for group_id in sentences_group:
        indices = indices_by_groupID[group_id]
        for index in indices:
            if index not in intersect_indices:
                indices_with_group_id.append(index+(group_id,))
    for idx in intersect_indices:
        indices_with_group_id.append(idx+(intersect_group_id,))
    return sorted(indices_with_group_id, key=lambda my_tuple: my_tuple[0]) # sorted by order of sentence in ori text
def get_highlight_html(text,indices,colors={0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}):
    print('indices:')
    print(indices)
    highlighted_text = ''
    last_pos = 0
    for index in indices:
        start,end,color_id = index
        color_code = colors[color_id]
        highlighted_text += text[last_pos:start] +'<span style="background-color: {}">'.format(color_code) + text[start:end] + '</span>'
        last_pos = end
    return highlighted_text
            
            

    
def show(doc_i):
    st.header('{} {}'.format(dataset,doc_i))
    if dataset == 'Newsroom':
        show_newsroom(doc_i)
    else:
        show_cnn(doc_i)

def show_newsroom(i):
    # st.header('newsroom {}'.format(i))
    #<span style="background-color: #D0F15F">paragraph</span>
    text = ref_newsroom_data[i]['text']
    top_n = tfidf_newsroom[i][:n_sentences]
    
    sentences_group = dict()
    sentences_group[0] = [sentence[1] for sentence in top_n]
    sentences_group[1] = presumm_newsroom[i]

    highlighted_indices =get_highlight_indices(text,sentences_group)
    highlighted_text = get_highlight_html(text,highlighted_indices)

    st.markdown(highlighted_text,unsafe_allow_html=True)

    # st.markdown('{}'.format(ref_newsroom_data[i]['text']),unsafe_allow_html=True)
    st.subheader('Reference Summary')
    st.markdown(re.sub("(.{64})", "\\1\n", ref_newsroom_data[i]['summary'], 0, re.DOTALL))
    # top_n = tfidf_newsroom[i][:n_sentences]
    st.subheader('My Summary')
    print(top_n)
    sorted_summ =sorted(top_n, key=lambda sentence: sentence[0])
    for sentence in sorted_summ:
        st.markdown('{}\n'.format(sentence[1]))
    st.subheader('PreSumm Summary')
    #st.text('---------------------------------------------------presum summary---------------------------------------------------')
    for sentence in presumm_newsroom[i]:
        st.markdown('{}\n'.format(sentence))

def show_cnn(doc_i):
#     cnn_ref_text = pickle.load( open('./Presentation/data/cnn_ref_text','rb'))
# cnn_ref_summaries = pickle.load(open('./Presentation/data/cnn_ref_summaries','rb'))
# cnn_tfidf_ret = pickle.load(open('./Presentation/data/cnn_tfidf_ret','rb'))
# cnn_presumm_ret = pickle.load(open('./Presentation/data/cnn_presumm_ret','rb'))
    text = cnn_ref_text[doc_i].replace('``', '""')
    top_n = cnn_tfidf_ret[doc_i][:n_sentences]
    sentences_group = dict()
    sentences_group[0] = [sentence[1].replace('``', '""') for sentence in top_n]
    sentences_group[1] = [sentence.replace('``', '""') for sentence in cnn_presumm_ret[doc_i]]
    highlighted_indices = get_highlight_indices(text,sentences_group)
    highlighted_text = get_highlight_html(text,highlighted_indices)
    st.markdown(highlighted_text,unsafe_allow_html=True)
    #st.markdown('{}'.format(cnn_ref_text[doc_i]))
    st.subheader('Reference Summary')
    st.markdown('{}'.format(cnn_ref_summaries[doc_i]))#(re.sub("(.{64})", "\\1\n", ref_newsroom_data[i]['summary'], 0, re.DOTALL))
    st.subheader('My Summary')
    print(top_n)
    sorted_summ =sorted(top_n[:n_sentences], key=lambda sentence: sentence[0])
    for sentence in sorted_summ:
        st.markdown('{}\n'.format(sentence[1]))

    st.subheader('PreSumm Summary')
    for sentence in cnn_presumm_ret[doc_i]:
        st.markdown('{}\n'.format(sentence))
   
if next:
    doc_i = np.random.randint(1000)
    show(doc_i)
# elif new_article:
#     show_newsroom(number)
else:
    show(number)
