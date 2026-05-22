#activationg gpu rtx 4050
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch as ts
ts.cuda.get_device_name(0)

### INGESTION PIPELINE ###

from langchain_community.document_loaders import PyMuPDFLoader as pd , DirectoryLoader as dl

#pdf ingestion and data loading

loader = dl(
    path=r"C:\Users\njnin\OneDrive\Desktop\code\__pycache__\rag_data" , 
    glob="*.pdf",
    loader_cls=pd
)
docs = loader.load()


# creatinng chunks 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(docuMENT):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunk_doc = text_splitter.split_documents(docuMENT)
    return chunk_doc
chunks=splitter(docs)

# embeddding 

from sentence_transformers import SentenceTransformer 

class embedding_part():

    def __init__(self , model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name ,device= "cuda")
        print(f"activating model : {self.model_name} dimension = {self.model.get_embedding_dimension()}")

    def generate_embedding(self , text):
        embed = self.model.encode(text , show_progress_bar=True)
        print("embeding shape : " , embed.shape)
        return embed

embedding = embedding_part()

# vector DB

import chromadb
import uuid

class vector_DB():

    def __init__(self ,collection_name = "pdf_documents",  prestistant_directary = r"C:\Users\njnin\OneDrive\Desktop\code\__pycache__\vector_DB"):

        self.collection_name = collection_name
        self.prestistant_directary = prestistant_directary
        self.client = None
        self.collection = None

        self._initialization_DB()

    def _initialization_DB(self):

        os.makedirs(self.prestistant_directary , exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.prestistant_directary)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name , 
            metadata = {"discription": " this is Vector DB where vector embedded chunks are stored "}
            )        
        print( "Collection _ Name : ",self.collection_name)
        print("doc count : " , self.collection.count())
    
    
    ids = []
    all_metadata = []
    embedding_list = []
    document_content = [] 

    def adding_documents(self , documents , embedding):

        for i , (doc,emd) in enumerate(zip(documents , embedding)):
            
            doc_id = f"doc_id{uuid.uuid1()}"
            self.ids.append(doc_id)

            metadata =dict(doc.metadata)
            metadata["index"] = i
            metadata["content_lenght"] = len(doc.page_content)
            self.all_metadata.append(metadata)
            self.embedding_list.append(emd.tolist())
            self.document_content.append(doc.page_content)

        self.collection.add(
            ids = self.ids,
            metadatas=self.all_metadata,
            documents=self.document_content,
            embeddings=self.embedding_list
            )
        return self.collection

# embedding the chunks 
embedded_chunks = []
for cnk in chunks:
    
    cnk = embedding.generate_embedding(cnk.page_content)
    embedded_chunks.append(cnk)

# adding into the vector data base
vector_database = vector_DB()
print(vector_database.adding_documents(chunks , embedded_chunks))



        
        
