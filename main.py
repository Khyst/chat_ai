# Execute "streamlit run main.py" every steps

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os, sys, datetime, time

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage, 
    HumanMessage,
    AIMessage,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain

import openai

from PyPDF2 import PdfReader

from streamlit_modal import Modal

import streamlit.components.v1 as components


def init():

    load_dotenv()
    
    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(
        page_title="Workation Assisntant",
        page_icon="😎"
    )

def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_the_documents(path):

    loader = PyPDFLoader(path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = text_splitter.split_documents(documents)

    return documents

def create_vector_stores_with_embedding(documents):
    
    embeddings = OpenAIEmbeddings()

    vectorstore=Chroma.from_documents(
        documents,
        embedding=embeddings,
    )

    return vectorstore

def conversation_chain(vectorstore):

    llm = ChatOpenAI(max_tokens=3000, max_retries=3000, model="gpt-3.5-turbo-16k", temperature=0.9)
    
    streaming_llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], max_tokens=3000, max_retries=3000, model="gpt-3.5-turbo-16k", temperature=0.9)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    
    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever()
    
    retriever.search_kwargs['max_token_limit'] = 4096

    conversation_chain = ConversationalRetrievalChain(
        question_generator=question_generator,
        retriever = retriever,
        combine_docs_chain=doc_chain,
        memory=memory,
    )

    return conversation_chain

def handle_user_input(user_input):

    with st.spinner("Processing..."):
        response = st.session_state.conversation({'question' : user_input})
    
    st.session_state.chat_history = response['chat_history']
    
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i)+'_user')
        else:
            message(msg.content, is_user=False, key=str(i)+'_chatbot')
    
def handle_csv_data(country, region, options_1, options_2, startDate_obj, endDate_obj):
    import pandas as pd
    import numpy as np
    import random

    # ----------------------------------------------------------------------
    work_acco = pd.read_csv("docs/workation_accommodation.csv", encoding='cp949')
    work_rest = pd.read_csv("docs/workation_restaurant.csv", encoding='cp949')
    work_tour = pd.read_csv("docs/workation_tourism.csv", encoding='utf8')
    # ----------------------------------------------------------------------

    startDate = '03/08/2023'
    endDate = '04/08/2023'

    selectedRegion = []
    selectedRegion.append(country[0])
    selectedRegion.append(region[0])
    print(selectedRegion)

    selectedTag = options_1
    print(selectedTag)

    format_str = '%d/%m/%Y'  

    days = int(endDate_obj.strftime("%d")) - int(startDate_obj.strftime("%d")) + 1
        
    # Accommodation Filterting
    # ----------------------------------------------------------------------
    mask_1 = (work_acco["행정구역"] == selectedRegion[0]) & (work_acco["지역"] == selectedRegion[1])
    mask_2 = (work_acco["카테고리_1"].isin(selectedTag)) | (work_acco["카테고리_2"].isin(selectedTag))

    print(work_acco)
    user_acco = work_acco.loc[mask_1 & mask_2, :]
    print(user_acco)
    
    seleectedIndex = random.choice(user_acco.index.tolist())
    recommend_acco = user_acco.loc[seleectedIndex, :]

    # ----------------------------------------------------------------------

    # Tourism Filterting
    # ----------------------------------------------------------------------
    selectedTag = options_2
    mask_1 = (work_tour["행정구역"] == selectedRegion[0]) & (work_tour["지역"] == selectedRegion[1])
    mask_2 = (work_tour["태그"].isin(selectedTag))

    user_tour = work_tour.loc[mask_1 & mask_2, :]

    # ----------------------------------------------------------------------

    # Restaurant Filterting
    # ----------------------------------------------------------------------

    selectedRegion = ["제주시", "제주시내"]

    mask_1 = (work_rest["행정구역"] == selectedRegion[0]) & (work_rest["지역"] == selectedRegion[1])

    user_rest = work_rest.loc[mask_1, :]

    # ----------------------------------------------------------------------
    recommend_rest_list = []
    recommend_tour_list = []

    f = open("chat_query.txt", "w", encoding="cp949")

    f.write("아래 양식을 참고해서, 워케이션 계획을 가독성 있고 친화적이게 알려줘. 각 관광지, 워케이션에 대해 상세하게 알려주면 좋겠고, 중간 중간에 여행에 대한 부가정보도 추가해주면 좋아. \n")

    f.write("(양식)\n")
    f.write("---\n")
    f.write("<"+startDate_obj.strftime("%y-%m-%d") + "부터 " + endDate_obj.strftime("%y-%m-%d") + "까지의 워케이션 일정>\n")

    f.write(f'1. 추천된 워케이션')
    f.write(f'워케이션 장소 : {recommend_acco["워케이션 이름"]}\n')
    f.write(f'워케이션 종류 : {recommend_acco["카테고리_1"]}, {recommend_acco["카테고리_2"]}\n')
    f.write(f'주소 : {recommend_acco["주소"]}\n')
    f.write(f'전화번호 : {recommend_acco["주소"]}\n')
    f.write(f'정보 : {recommend_acco["소개"]}\n')
    f.write(f'예약하기 : {recommend_acco["예약링크"]}\n')

    for i in range(days):
        # 숙박 장소
        f.write(f"[{i+1}번째 날]\n\n")
        
        # 추천된 맛집 장소
        # user_rest
        f.write("2. 추천된 관광지 및 맛집")

        for _ in range(2):
            seleectedRestIndex = random.choice(list(set(user_rest.index.tolist()) - set(recommend_rest_list)))
            recommend_rest_list.append(seleectedRestIndex)
            recommend_rest = work_rest.loc[seleectedRestIndex, :]
            f.write(f'추천된 맛집 : {recommend_rest["식당 이름"]}\n')
            f.write(f'주소 : {recommend_rest["주소"]}\n')

        for _ in range(2):
            seleectedTourIndex = random.choice(list(set(user_tour.index.tolist()) - set(recommend_tour_list)))
            recommend_tour_list.append(seleectedTourIndex)
            recommend_tour = work_tour.loc[seleectedTourIndex, :]

            f.write(f'추천된 관광지 : {recommend_tour["관광지 이름"]}\n')
            f.write(f'종류 : {recommend_tour["태그"]}\n')
            f.write(f'주소 : {recommend_tour["주소"]}\n')
            f.write(f'전화번호 : {recommend_tour["전화번호"]}\n')
            f.write(f'소개 : {recommend_tour["소개"]}\n')


    f.write("---\n")

    f.close()

    f = open("chat_query.txt", "r")

    chat_query = f.read()

    st.session_state.chat_history = []

    try:
        response = st.session_state.conversation({'question' : chat_query})

        st.session_state.chat_history = response['chat_history']

        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                #ex_msg = "입력 예시: \n" + msg.content
                #print("입력 예시: \n", msg.content)
                #message(ex_msg, is_user=False, key=str(i)+'_chatbot')
                pass
            else:
                print(msg.content)
                #st.write('Hello, *World!* :sunglasses:')
                with st.container():
                    st.write(msg.content)
                pass

    except openai.InvalidRequestError :
        message("사용할 수 있는 Token의 갯수를 초과하였습니다. 페이지를 새로 고침하거나, 조건을 다르게 해서 적어주세요.")    

def user_input():
    message("안녕하세요, 저는 여러분의 워케이션 어시스턴트에요. 여러분이 가고 싶어하는 여행 정보에 대해 간략히 알려주세요")
    #message("(워케이션 시작일, 종료일, 숙박(워케이션 장소 종류), 선호하는 식당 종류, 선호하는 레저 종류) 가 들어가게 애기해주세요.")
    
    start_date = st.date_input("워케이션 시작일자를 선택해주세요", datetime.date(2023, 7, 6))
    end_date = st.date_input("워케이션 종료일자 선택해주세요", datetime.date(2023, 7, 6))

    country = ["제주시", "서귀포시"]

    region = st.multiselect(
        '워케이션을 가려는 상세 지역을 선택해주세요(한곳만 선택해주세요)',
        ["제주시내", "서귀포시내", "조천", "구좌", "안덕", "성산", ],
        max_selections=1
        )

    # options_1 = st.multiselect(
    #     '선호하는 워케이션 장소의 유형을 선택해주세요',
    #     )
    
    options_1 = ["도심형", "소도시형", "해양형", "산악형", "스테이형", "호텔형", "워케이션 특화형"]
    
    options_２ = st.multiselect(
        '워케이션에서 즐기려고 하는 태그를 선택해주세요',
        ["골프", "다이빙", "서핑", "승마", "역사유적", "오름", "요가", "이색체험", "자연경관", "전시/미술관", "카트/ATV", "테마파크", "해변", "카페", "보트"],
        )
    
    return country, region, options_1, options_2, start_date, end_date
    
def main():

    init()
    
    chat = ChatOpenAI(temperature=0)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="너는 정말 유익한 정보를 알려주는 친구야."),
        ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Your workation Buddy : Woody 😀")
    st.write("당신의 워케이션을 도와주는 Woody를 사용해보세요")
    st.write("(Beta) 현재는 제주 지역을 한정하여서 서비스를 오픈하였습니다.")

    # User Usage
    country, region, options_1, options_2, start_date, end_date = user_input()

    if st.button('워케이션 계획해줘!'):
        with st.spinner("Preparing..."):
            # 1. Get the documents
            documents = get_the_documents("./docs/workation_1.pdf")
            # 2. Create Our Vector stores with embedding
            vectorstore = create_vector_stores_with_embedding(documents)
            # Create Converstation Chain (or Agent)
            st.session_state.conversation = conversation_chain(vectorstore)        
            
        with st.spinner("Loading..."):
            try:
                handle_csv_data(country, region, options_1, options_2, start_date, end_date)
            except IndexError:
                message("조건을 만족하는 충분한 데이터를 찾지 못하였습니다. 조건을 다르게 해서 적어주세요")

if __name__ == '__main__':
    main()
