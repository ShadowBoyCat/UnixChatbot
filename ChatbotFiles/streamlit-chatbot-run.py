from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import chromadb
from langchain.prompts.chat import ChatPromptTemplate

from chromadb.config import Settings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

import os
from uuid import uuid4

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Test111 - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

compartment_id = "ocid1.tenancy.oc1..aaaaaaaawtp43a6h35uyjcnmpityq57d26vuib2ngji4hdfg2sz7utozyjva"
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
llm = OCIGenAI(
    model_id = "cohere.command-light",
    service_endpoint = endpoint,
    compartment_id = compartment_id,
    model_kwargs = {"max_tokens":1000}
)

embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint=endpoint,
    compartment_id=compartment_id,
    model_kwargs={"truncate":True},
)

db = FAISS.load_local("/home/ubuntu/src/faiss_index", embeddings, allow_dangerous_deserialization=True)

retv = db.as_retriever(search_type="similarity", search_kwargs={"k":5})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a very knowledgeable Chatbot that answers questions accurately"),
    ("human", "{question}")
])

history = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key="answer")

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retv, memory=memory, return_source_documents=True,)

import streamlit as st

st.set_page_config(page_title="Unix Chatbot")
st.title("Welcome to my Unix Chatbot")
st.subheader("Ask me any question about Unix")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)
    
if x := st.chat_input():
    st.chat_message("human").write(x)
    
    response = qa({"question": x})
    st.chat_message("ai").write(response["answer"])
    
    #with st.expander("Source Documents"):
        #for doc in response["source_documents"]:
            #st.write(doc.page_content)
