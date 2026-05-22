# rag_pdf.LLM

# documenting the entire project 
** Ingestion pipeline 
 > Extracting the documents
 > coverting them into chunks
 > embedding the chunks
 > using chroma db to store

# Extrating the documents 
  Used langchain_community to load the directory and loaded them in PyMuloader ( why PyMU ? cuz to extract the images from doc in case needed)
# Converting them into chunks 
  Used RecursiveCharacterTextSplitter from langchain_text_splitter 
    chunk_size = 500,
    chunk_overlap = 50 ( for not loosing the meaning of a sentence in previous chunk or forth )
# Embedding the Chunks
  I created mannual class for embedding, using SentenceTransformer model_name = "all-MiniLM-L6-v2"  to build my foundation aslo to not write the the embedding      part again for query .
# Chromadb as VectoreDB
  To store vector

  
  
     
    
