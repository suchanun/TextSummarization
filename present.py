import pickle

from utils_smart_stat import *
import SessionState
import pythainlp
from thai_utils import *

st.sidebar.title("What to do")
app_mode = st.sidebar.selectbox('language', ('English', 'Thai (an experiment)'))

colors_dict = {0:'#D0F15F',1:'#90F9E3',2:'#E9B2ED'}
state = SessionState.get(current_article_num=0,current_thai_article_num=0)


def main():
    st.title('Extractive Summarization')
    if app_mode == 'Thai (an experiment)':

        run_thai_app()

    elif app_mode == 'English':
        run_app_smart()


def all_stopwords(text):
    words = word_tokenize(text.lower())
    for word in words:
        if re.search('[a-zA-Z]', word) and word not in stoplist:
            return False
    return True


def run_app_smart():

    dataset = st.sidebar.radio('Dataset', ( 'Newsroom','CNN', 'Custom Input'))
    one_minus_k = st.sidebar.slider(
        'keywords: importance ---> diversity',
        0.0, 1.0, value=0.3
    )
    k = 1-one_minus_k
    data = load_data()
    ranker = data['smart_ranker']

    n_sentences = st.sidebar.number_input('number of sentences in summary', value=5, min_value=1)
    if dataset != 'Custom Input':
        n_articles = 1000


        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">Our summary</span>'.format(colors_dict[0]),
            unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:17px"><span style="background-color: {}">Presumm\'s summary</span>'.format(colors_dict[1]),
            unsafe_allow_html=True)
        st.markdown('<p style="font-size:17px"><span style="background-color: {}">Both</span>'.format(colors_dict[2]),
                    unsafe_allow_html=True)


        article_number = st.sidebar.number_input(label='{} article number'.format(dataset), value=state.current_article_num,
                                                 format='%d', max_value=n_articles - 1, min_value=0)
        st.sidebar.markdown('or')
        random = st.sidebar.button('random an article')
        if random:
            doc_i = np.random.randint(1000)
            state.current_article_num = doc_i
        else:
            doc_i = article_number



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
        Displayer.show(info, n_sentences)

    else:
        custom_text = st.text_area('text to be summarized',height=400)

        custom_text = custom_text.strip()

        if  custom_text != '' and re.search('[a-zA-Z]', custom_text):
            if all_stopwords(custom_text):
                st.markdown("Nothing to be summarized")
            else:
                info = dict()
                info['text'] = custom_text
                info['my_model_result'],info['keywords']= ranker.rank_sentences(custom_text,k=k)#,m=m)
                Displayer.show_custom(info, n_sentences)
                xs = list(range((len(info['my_model_result']))))
                sortedByPos = sorted(info['my_model_result'],key=lambda tup: tup[0])
                Displayer.display_figure(x=xs,y=[tup[2] for tup in sortedByPos],title='score by position',xaxis_title='position',yaxis_title='score')

def process_thai_text_for_highlight(text):
    ori_text_nodouble_newline = re.sub(r'\n+', '\n', text).strip()
    processed_text = ''
    for paragraph in ori_text_nodouble_newline.split('\n'):
        psentences = pythainlp.tokenize.sent_tokenize(paragraph)
        processed_text += ' '.join(psentences) + '\n'
    return processed_text#.strip()
def run_thai_app():
    df,ranker = load_thai_data()
    n_groups = st.sidebar.number_input('number of sentences in summary', value=5, min_value=1)
    article_number = st.sidebar.number_input('article number', value=state.current_thai_article_num, min_value=0, max_value=999)
    st.sidebar.markdown('or')
    random = st.sidebar.button('random an article')
    if random:
        doc_i = np.random.randint(1000)
        state.current_thai_article_num = doc_i
    else:
        doc_i = article_number
    st.header('Prachathai {}'.format(doc_i))
    item = df.iloc[doc_i]
    text = item['body_text']
    sentences_with_scores, paragraphs = ranker.rank_phrases(text, n_groups)
    info = dict()
    info['title'] = item['title']
    info['text'] = process_thai_text_for_highlight(text)#text
    info['paragraphs'] = paragraphs
    info['sentences_with_scores'] = sentences_with_scores
    ThaiDisplayer.show_summary_and_text(info)

@st.cache(allow_output_mutation=True)
def load_thai_data():
    return pickle.load(open('data/prachathai1000.pkl', 'rb')), pickle.load(open('data/thai_ranker.pkl', 'rb'))
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
    #data['smart_ranker'] = pickle.load(open('./data/ranker_fixed_stoplist_punc.pkl','rb')) #ranker_fixed_stoplist.pkl
    data['smart_ranker'] =pickle.load(open('./data/tfidf_fixed_lemma_final_ver.pkl','rb'))


    return data #[cnn_ref_text,cnn_ref_summaries,cnn_tfidf_ret,cnn_presumm_ret,presumm_newsroom,tfidf_newsroom,ref_newsroom_data]

if __name__ == "__main__":
    main()