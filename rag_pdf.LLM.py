import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch as ts
ts.cuda.get_device_name(0)

### INGESTION PIPELINE ###

from langchain_community.document_loaders import PyMuPDFLoader as pymu , DirectoryLoader as dl

#pdf ingestion and data loading

loader = dl(
    path=r"C:\Users\njnin\OneDrive\Desktop\code\__pycache__\rag_data" , 
    glob="*.pdf",
    loader_cls=pymu
)
docs = loader.load()


# creatinng chunks 

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(docuMENT):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
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
        embed = self.model.encode(text , show_progress_bar=True ,batch_size=32)
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
    


    def adding_documents(self , documents , embedding):

         ids = []
         all_metadata = []
         embedding_list = []
         document_content = [] 

         for i , (doc,emd) in enumerate(zip(documents , embedding)):
           
            doc_id = f"doc_id{uuid.uuid1()}"
            ids.append(doc_id)

            metadata =dict(doc.metadata)
            metadata["index"] = i
            metadata["content_lenght"] = len(doc.page_content)
            all_metadata.append(metadata)
            embedding_list.append(emd.tolist())
            document_content.append(doc.page_content)

         self.collection.add(
            ids = ids,
            metadatas=all_metadata,
            documents=document_content,
            embeddings=embedding_list
            )
         return self.collection


texts = [c.page_content for c in chunks]
embedded_chunks = embedding.generate_embedding(texts)

vector_database = vector_DB()
if vector_database.collection.count() == 0:
    vector_database.adding_documents(chunks, embedded_chunks)
else:
    print(f"DB already has {vector_database.collection.count()} docs!")

### retrival pipeline !

class retrival :

    def __init__(self , embedding_part ,vector_DB):

        self.embedding_part  = embedding_part
        self.vector_DB = vector_DB
    
    def retrival_part (self , query , top_k = 5 ,score_threshold = 0.0):

        # query to embedding
        query_embed = self.embedding_part.generate_embedding([query])[0]

        #simantic search 
        results = self.vector_DB.collection.query(
            query_embeddings = [query_embed.tolist()],
            n_results  = top_k
                ) 

        # cosine similarities
        retrived_doc = []

        if results["documents"] and results["documents"][0]:

            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i , (doc_id , metadata , document , distance) in enumerate(zip(ids,metadatas, documents ,distances)):

                similarity_score = 1/(1+distance)

                if similarity_score >= score_threshold :

                    retrived_doc.append({
                        "id":doc_id,
                        "metadata" : metadata,
                        "document" : document,
                        "distance" : distance,
                        "similarity_score" : similarity_score,
                        "rank" : i+1
                    })

        else :
            print("no ducments found")
        
        return retrived_doc
    

retriever = retrival(embedding, vector_database)
results = retriever.retrival_part(
    "retrieval augmented generation",
    score_threshold=0.0
)
for doc in results:
    print(f"\nRank: {doc['rank']}")
    print(f"Similarity: {doc['similarity_score']}")
    print(f"Content: {doc['document'][:200]}")
