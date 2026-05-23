import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch as ts
ts.cuda.get_device_name(0)

### INGESTION PIPELINE ###

from langchain_community.document_loaders import PyMuPDFLoader as pymu , DirectoryLoader as dl

#pdf ingestion and data loading

loader = dl(
    path=r".\rag_data", 
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
"""___________________________________________________________________________________________________"""
"""results = retriever.retrival_part(
    "retrieval augmented generation",
    score_threshold=0.0
)
for doc in results:
    print(f"\nRank: {doc['rank']}")
    print(f"Similarity: {doc['similarity_score']}")
    print(f"Content: {doc['document'][:200]}")
    ___________________________________________________________________________________________________________-"""

# Integrationn with LLM 

from transformers import AutoTokenizer , AutoModelForCausalLM ,BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline 

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained( model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map =("auto"),
    quantization_config=bnb_config,
    torch_dtype = "auto"
)

pipe = pipeline(
    "text-generation",
    model,
    tokenizer = tokenizer,
    max_new_tokens = 500
)

llm = HuggingFacePipeline(pipeline = pipe)

def ask_rag(qry):

    docs = retriever.retrival_part(
        qry ,
        score_threshold=0.35
    )

    context =[]

    for doc in docs:
        source = os.path.basename(doc["metadata"].get("file_path" ,"Unknown file"))
        page = doc["metadata"].get("page" ,"Unknown page")
        text = doc["document"]
        context.append(f"Source : {source} Page : {page} /n{text}")
    prompt =  f"""
You are a retrieval-based assistant.

Rules:

1. Answer ONLY using the provided context.
2. If answer not present:
   "I could not find sufficient information in the provided documents."
3. If question is unrelated:
   "This question appears unrelated to the uploaded documents."
4. Never invent facts, names, dates or numbers.
5. Answer only supported information.
6. Always mention the source (file + page) when giving an answer

context = {context}

query = {qry}

Answer : """
    response = llm.invoke(prompt)

    return response

qry = input("ENTER YOUR QUESTION : ")
anwer =ask_rag (qry)

print("LEMME THINKING BRUHH PLZ WAIT...")
print(f"hmmm...alright herewegooo....{anwer}")

    









        



        
        
