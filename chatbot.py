from IPython.display import Markdown
import textwrap
import google.generativeai as genai
import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI

import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
#from langchain import PromptTemplate # Import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document as LangChainDocument # Import Document from langchain.schema and rename it
from langchain.chains.question_answering import load_qa_chain
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
warnings.filterwarnings("ignore")


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Prompt the user to enter the API key
GOOGLE_API_KEY ='AIzaSyDro7A5ky_aURZeBbFC_nJdGES2Q1svQNc'#'AIzaSyBs7fa0hdQsl2cPEQYwLhsO5INKQZxt0Vk'

# Store the API key as an environment variable
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Retrieve the stored API key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')#'AIzaSyBs7fa0hdQsl2cPEQYwLhsO5INKQZxt0Vk'  #AIzaSyAp_54Vs7Ur6OmXxRhz_60ndtBmcDQMDjA
genai.configure(api_key=GOOGLE_API_KEY)
#model = genai.GenerativeModel(model_name = "gemini-pro")
#llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17",google_api_key=GOOGLE_API_KEY,
                             temperature=0.2,convert_system_message_to_human=True)







pdf_loader = PyPDFLoader("your pdf path.pdf")
pages = pdf_loader.load_and_split()
pages[0].page_content

# Iterate through each Document object in the 'pages' list
modified_pages = []
for page in pages:
    modified_content = page.page_content.replace("\n", " ")  # Replace newlines in each page's content
    # Use LangChainDocument when creating new Document objects
    modified_pages.append(LangChainDocument(page_content=modified_content, metadata=page.metadata))

# Now 'modified_pages' contains the pages with newlines replaced by spaces

text_splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=70)
context = "\n\n".join(str(p.page_content) for p in modified_pages)
texts = text_splitter.split_text(context)
print("befor embeddings ")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
print("embedings done ")
vector_index = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":3})


print("vector done")
template = """Utilisez les éléments de contexte suivants pour répondre à la question à la fin. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas; ne cherchez pas à inventer une réponse. Gardez la réponse aussi concise que possible.
{context}
Question : {question}
Réponse utile :"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)# Run chain
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("result  ")

def final_result(question):
    result = qa_chain({"query": question})
    return result["result"]
print("am here ")
demo = gr.Interface(fn=final_result, inputs="text", outputs="text")
demo.launch()
if __name__ == "__main__":
    demo.launch()

'''
def final_result(question):
    result = qa_chain({"query": question})
    return result["result"]
'''
