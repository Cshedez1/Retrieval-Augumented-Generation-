import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        try:
            documents = [uploaded_file.getvalue().decode("utf-8")]  # Assuming file is uploaded in a readable text format
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        
        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create a vectorstore from documents
        db = Chroma.from_documents(texts, embeddings)
        
        # Create retriever interface
        retriever = db.as_retriever()
        
        # Create QA chain
        qa = RetrievalQA(llm=OpenAI(openai_api_key=openai_api_key), retriever=retriever)
        
        # Run the query
        return qa.run(query_text)

    return "No file uploaded."


# Page title
st.set_page_config(page_title='🔗🦉 Ask the Doc App🖥')
st.title('🔗🦉 Ask the Doc App🖥')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='txt')

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

# Form input and query
result = None
with st.form('myform'):
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    submitted = st.form_submit_button('Submit')
    
    if submitted:
        if openai_api_key and uploaded_file and query_text:
            with st.spinner('Calculating...'):
                result = generate_response(uploaded_file, openai_api_key, query_text)
        else:
            st.error("Please make sure all fields are filled correctly and try again.")

if result:
    st.info(result)
