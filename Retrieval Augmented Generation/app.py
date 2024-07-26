import os
import qdrant_client
from pypdf import PdfReader
from langchain_qdrant import Qdrant
from concurrent.futures import ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import openai

os.environ['OPENAI_API_KEY'] = "replace_your_api_key"



openai.api_key = "replace_your_api_key"

client = QdrantClient(
    "paste_your_qdrant_cluster_link",
    api_key="replace_your_qdrant_key",
)
embeddings = OpenAIEmbeddings(openai_api_key="replace_your_api_key")

def addData():
    PATH = "C:/Users/ram/openai/python/content/"

    def load_documents():
        documents = []
        filenames = [filename for filename in os.listdir(PATH) if filename.endswith(".pdf")]
        
        if not filenames:
            print("No PDF files found in the directory.")
            return documents
        
        print(f"Found PDF files: {filenames}")
        
        def process_file(filename):
            file_path = os.path.join(PATH, filename)
            try:
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    page_content = page.extract_text()
                    if page_content:
                        metadata = {"filename": filename, "page_number": i}
                        document = Document(content=page_content, page_content=page_content, metadata=metadata)
                        documents.append(document)
                        print(f"Processed page {i} of {filename}")
            except Exception as e:
                print(f"Failed to process file {filename}: {e}")
        
        with ThreadPoolExecutor() as executor:
            executor.map(process_file, filenames)
        
        return documents

    def split_text(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True,
        )
        return text_splitter.split_documents(documents)

    documents = load_documents()
    if not documents:
        print("No documents loaded.")
        return
    
    chunks = split_text(documents)
    if not chunks:
        print("No chunks created from documents.")
        return
    
    print('Loaded and split documents into chunks')

    client = QdrantClient(
        "paste_your_qdrant_cluster_link",
        api_key="replace_your_qdrant_key",
        timeout=300
    )
    
    vector_store = Qdrant(
        client=client,
        collection_name="Agri",
        embeddings=embeddings
    )

    vector_store.add_documents(chunks)


def aiRequest(userQues,pastQuestion):
    client = qdrant_client.QdrantClient(
        "paste_your_qdrant_cluster_link",
        api_key="replace_your_qdrant_key",
    )

template = """
If the question contains 'hello' or 'hi', then send a welcome message. 
Use the following pieces of context to provide answers related to regenerative farming practices. Refer to past user questions and combine relevant information to answer the current question effectively. Explain the regenerative farming practices and steps for implementation. If you don't know the answer, simply say that you don't know. Do not make up an answer. Keep the response concise and check if the question is connected to previous ones to provide a cohesive answer.
{context}
"""
    Past Question:{pastQues} 
 
    Current Question: {question}"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    qra_client = Qdrant(client=client, collection_name='Agri', embeddings=embeddings)
    result = qra_client.similarity_search_with_relevance_scores(query=userQues, k=5)

    prompt = QA_CHAIN_PROMPT.format(context=result, question=userQues,pastQues=pastQuestion)

    model = ChatOpenAI(
        api_key="your_openai_key",
    )
    
    return model.invoke(prompt)

