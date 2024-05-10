from tavily import TavilyClient
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
from langchain_groq import ChatGroq
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import streamlit as st

# """
# First set up your API keys. you would need an API for OpenAI and another one for Tavily.
# You could also replace the llm used in this context with another one such as llama3 70B which works just fine.
# """

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

tavily = TavilyClient(api_key='Your_API')
llm = ChatOpenAI(model='gpt-4-turbo', temperature=1)


def scrape_text(url: str):
    """
    Scrapes the text from any url
    :param url:
    :return:
    """
    # Send a GET request to the webpage
    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the content of the request with BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract all text from the webpage
            page_text = soup.get_text(separator=" ", strip=True)

            # Print the extracted text
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)  # noqa: T201
        return f"Failed to retrieve the webpage: {e}"


#####
# Lets create a query. we'll ask for some query keywords and then add some of our own.
# you can certainly change the "background" queries to better suit your search.
#####

name = st.text_input('Enter the name of the professor/Dr youre trying to look up')
second_query = st.text_input('Add another search query if necessary')
start = st.button('Start')

if second_query:
    query = f'{name} AND {second_query}'
else:
    query = name

query_list = [query, f'{name} AND googlescholar', f' {name} AND sport science', f'{name} AND sport science AND email']

st.markdown('Your Query:')
st.write(query)

if start:
    st.write('Wait a minute or two')

    urls = []

    for q in query_list:
        response_list = (tavily.search(query=q, search_depth="advanced", max_results=3))
        context = [{obj["url"]: obj["content"]} for obj in response_list['results']]
        for item in context:
            for url, text in item.items():
                urls.append(url)

    overall_context = []
    end_text = ""

    for url in urls:
        overall_context.append(scrape_text(url))

    for t in overall_context:
        end_text += t
    end_text = end_text

    ####
    # my embedding model doesn't support more than a million tokens so i had to set this limit
    ###

    if len(end_text) >= 1000000:
        end_text = end_text[:1000000]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, length_function=len,
                                              is_separator_regex=False)
    #####
    # make sure to put the end text inside a [] otherwise you'll get an error!
    ####

    splits = splitter.create_documents([end_text])
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
        collection_name="academic_rag"
    )

    ###
    # Yeah i know... this prompt can indeed get better. but it works just fine!
    ##

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to
                    answer four questions as short as possible based on the question:
                    1. what is the position of this dr/professor?
                    2.what are his areas of interest in research?
                    3.write one to three of his recent research articles.
                    4. what is his or her email?
                    5.what is his or her full name?
    
                    Original question name of the professor: {question}
                    be really cautious when choosing answers, specially in regards to full names and email addresses.
                    for example if a name is A Martín-Rodríguez then the full name would be Alexandra MARTÍN-RODRÍGUEZ
                    and not Vicente Javier CLEMENTE-SUÁREZ since the name doesnt start with A.
                    """,
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """
        based on the context below , answer four questions as short as possible:
        {context}
        {name}
        questions:
        1. what is the position of this dr/professor?
        2.what are his or hers areas of interest in research?
        3.write 1 to 3 of his recent research articles.
        4. what is his or her email?
        5.what is his or her full name?
    
        i expect a an answer like this:
        1.position of this dr/professor: answer to the first question\n
        2.His or her areas of interest in research: answer to the 2nd question\n
        3.Some of his recent research articles: answer to the 3rd question\n
        4.His or her email: answer to the 4th question\n
        5.His or her full name: answer to the 5th question
    
        if you dont know the answer to any of these questions say that you dont know and dont make stuff up.
        be really cautious when choosing answers, specially in regards to full names and email addresses.
        for example if a name is A Martín-Rodríguez then the full name would be Alexandra MARTÍN-RODRÍGUEZ.
        """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
            {"context": retriever, "question": RunnablePassthrough(), "name": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    answers = chain.invoke({'question': query, 'name': name})
    st.write(answers)
    vector_db.delete_collection()
