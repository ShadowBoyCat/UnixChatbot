from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
import os

# Set the correct path for loading PDFs
pdf_loader = PyPDFDirectoryLoader("../pdf-docs")
pages_dir = pdf_loader.load()

loaders =[pdf_loader]
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
all_documents = text_splitter.split_documents(documents)

print(f"Total number of documents : {len(all_documents)}")

compartment_id = "ocid1.tenancy.oc1..aaaaaaaawtp43a6h35uyjcnmpityq57d26vuib2ngji4hdfg2sz7utozyjva"
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint=endpoint,
    compartment_id=compartment_id,
    model_kwargs={"truncate":True},
)

batch_size=96

num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)

texts = ["FAISS is an important library", "LangChain supports FAISS", "Linux commands include but are not limited to: mkdir, rm, sh, and chmod"]
db = FAISS.from_texts(texts, embeddings)
retv = db.as_retriever()

for batch_num in range(num_batches):
    start_index = batch_num*batch_size
    end_index = (batch_num+1)*batch_size
    
    batch_documents = all_documents[start_index:end_index]
    
    retv.add_documents(batch_documents)
    
    print(start_index, end_index)

# Save the FAISS index in the correct directory
db.save_local("../fiass-index/faiss_index")
