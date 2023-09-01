__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os
from PIL import Image

#제목
st.title("ChatPDF")
st.write("---")

# 방 이미지
img = Image.open('cyworld-room.jpg')
st.image(img)

#파일 업로드
#uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])

#파일 업로드
# ["samsung_tv_manual.pdf", "lg_ac_manual.pdf", "winix_humidifier_manual.pdf"]
tv_file = PyPDFLoader("samsung_tv_manual.pdf")
ac_file = PyPDFLoader("lg_ac_manual.pdf")
hm_file = PyPDFLoader("winix_humidifier_manual.pdf")

menu = ['TV', '에어컨', '가습기']    #options
choice_box = st.radio('type1 : radio', menu)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
# choice_box = st.selectbox('type2 : selectbox', menu)

def document_to_db(uploaded_file, size):    # 문서 크기에 맞게 사이즈 지정하면 좋을 것 같아서 para 넣었어용
    pages = uploaded_file.load_and_split()
    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = size,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)
    return db

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    db_tv = document_to_db(tv_file, 500)
    db_ac = document_to_db(ac_file, 500)
    hm_tv = document_to_db(hm_file, 300)
 
    #Question
    st.header("기기를 선택하고 PDF에게 질문해보세요!!")

    if choice_box == menu[0]:
        data = tv_file.load()
        st.write(f"samsung_tv_manual.pdf : {len(data)}개의 페이지")
        st.write("---")

        question = st.text_input('질문을 입력하세요')
        if st.button('TV에게 질문하기'):
            with st.spinner('Wait for it...'):
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
                result = qa_chain({"query": question})
                st.write(result["result"])


    
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])
