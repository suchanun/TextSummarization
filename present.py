import pickle
from utils import *
from utils_smart_stat import *

st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Simple tf-idf", "Smart stat with tf-idf","benchmark"])
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
    else:
        run_app_smart()



def run_app_smart():

    dataset = st.sidebar.radio('Dataset', ( 'Newsroom','CNN', 'Custom Input'))
    one_minus_k = st.sidebar.slider(
        'keywords: important ---> diverse',
        0.0, 1.0, value=0.3
    )
    k = 1-one_minus_k
    data = load_data()
    ranker = data['smart_ranker']
    # m = st.sidebar.slider(
    #     'm: Select a range of values',
    #     0.0, 1.0, value=0.15  # (25.0, 75.0)
    # )

    n_sentences = st.sidebar.number_input('number of sentences in summary', value=5, min_value=1)
    if dataset != 'Custom Input':
        n_articles = 1000


        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">My summary</span>'.format(colors_dict[0]),
            unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">Their summary</span>'.format(colors_dict[1]),
            unsafe_allow_html=True)
        st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),
                    unsafe_allow_html=True)

        if app_mode != 'benchmark':
            article_number = st.sidebar.number_input(label='{} article article number'.format(dataset), value=0,
                                                     format='%d', max_value=n_articles - 1, min_value=0)
            st.sidebar.markdown('or')
            random = st.sidebar.button('random an article')
            if random:
                doc_i = np.random.randint(1000)

            else:
                doc_i = article_number
        else:
            article_number = st.sidebar.number_input(label='{} article article number'.format(dataset), value=0,
                                                     format='%d', max_value=100-1, min_value=0)
            doc_i = data['b_indices'][article_number]

        st.header('{} {}'.format(dataset, doc_i))
        info = dict()
        if dataset == 'CNN':
            info['text'] = data['cnn_ref_text'][doc_i]
            info['my_model_result'],info['keywords'] = ranker.rank_sentences(info['text'],k=k)#m=m
            info['presumm_result'] = data['cnn_presumm_ret'][doc_i]
            info['ref_summary'] = data['cnn_ref_summaries'][doc_i]
            # Displayer.show(info, n_sentences)
        elif dataset == 'Newsroom':
            info['text'] = data['ref_newsroom_data'][doc_i]['text']
            info['title'] = data['ref_newsroom_data'][doc_i]['title']
            info['presumm_result'] = data['presumm_newsroom'][doc_i]
            info['my_model_result'],info['keywords'] = ranker.rank_sentences(info['text'], k=k)  # m=m
            info['ref_summary'] = data['ref_newsroom_data'][doc_i]['summary']
            # Displayer.show(info, n_sentences)
        if app_mode == 'benchmark':
            Displayer.show(info, n_sentences,benchmark=True,pos=data['b_pos'][article_number])
        else:
            Displayer.show(info, n_sentences)

    else:
        custom_text = st.text_area('text to be summarized',height=400)
        if  custom_text != '':
            info = dict()
            info['text'] = custom_text
            info['my_model_result'],info['keywords']= ranker.rank_sentences(custom_text,k=k)#,m=m)
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
    data['smart_ranker'] = pickle.load(open('./data/ranker_fixed_stoplist_punc.pkl','rb')) #ranker_fixed_stoplist.pkl
    data['b_indices'] = pickle.load(open('./data/benchmark_indices.pkl','rb'))
    data['b_pos'] = pickle.load(open('./data/benchmark_positions.pkl','rb'))
    return data #[cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,presumm_newsroom,tfidf_newsroom,ref_newsroom_data]

if __name__ == "__main__":
    main()