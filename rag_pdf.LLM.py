
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
        self.model = SentenceTransformer(self.model_name)
        print(f"activating model : {self.model_name} dimension = {self.model.get_embedding_dimension()}")

    def generate_embedding(self , text):
        embed = self.model.encode(text , show_progress_bar=True)
        print("embeding shape : " , embed.shape())
        return embed

embedding = embedding_part()
print(embedding)

# vector DB

import chromadb
import os

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

vector_database = vector_DB()

print(vector_database)

        
        