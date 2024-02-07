import streamlit as st 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import openai

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    # Function call that takes a filepath (file parameter the function receives)
    # Splits it into two parts: the name and the extension.
    # Name would be the file name without its file extension.
    # Extension is the file extension, starting with the period (.) character, like .txt or .jpg
    name, extension = os.path.splitext(file)
    
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported')
        return None

    # When the .load method is called, it performs the action of loading data from somewhere, 
    # which could be a file, a database, or any other data source.   
    data = loader.load()
    return data

# This line defines the function chunk_data with three parameters:
# data: the text or dataset you want to split into chunks.
# chunk_size: an optional parameter that determines how large each chunk will be, defaulting to 256 characters.
# chunk_overlap: another optional parameter that determines how many characters will overlap between consecutive chunks, defaulting to 20 characters.
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # This instance is configured with the chunk_size and chunk_overlap values provided to the chunk_data function, meaning it will use these values to determine how to split the text.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # This method applies the splitting logic to the data, dividing it into chunks as specified by the chunk_size and chunk_overlap
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=3, model_choice= "gpt-3.5-turbo", temperature=1, max_tokens=2048):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model=model_choice, temperature=temperature)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(q)
    return answer  

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def test_openai_api_key(api_key):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
            model="text-davinci-003",  # Ensure this model is available in your plan
            prompt="Hello, world!",
            max_tokens=5
        )
        print("API Test Response:", response)  # Log the response
        return True
    except Exception as e:
        print("API Test Error:", e)  # Log detailed error
        return False

if __name__ == "__main__":
    # import os
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv(), override='True')

    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')

    # Access the API key using st.secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    with st.sidebar:
        user_api_key = st.text_input('OpenAI API Key:' , type='password', help="Enter your API key for OpenAI here.")
        api_key_valid = False

        # Check if the provided API key is valid
        if user_api_key:
            api_key_valid = test_openai_api_key(user_api_key)
            if api_key_valid:
                st.success("API key is valid.")
                api_key = user_api_key
                st.session_state['api_key_valid'] = True
            else:
                st.error("Invalid API key. Please enter a valid OpenAI API key.")
                st.session_state['api_key_valid'] = False     
        else:
            st.error("Please enter your OpenAI API key to use this application")
            api_key = None
        
        # Error message display logic
        if st.session_state.get('attempted_file_process', False) and not st.session_state.get('api_key_valid',False):
            st.error("Unable to process the file without an OpenAI API key.")
        # else:
        #     st.session_state['attempted_file_process'] = False
            
        # Widgets 
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf','docx','txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100 , max_value=2048, value=512, on_change=clear_history)
        k=st.number_input('k', min_value=1, max_value =20, value=3, on_change=clear_history) 

        add_data = st.button('Add Data', on_click=clear_history)

        # Reading, Chunking and Embedding Data
        if uploaded_file and add_data:
            st.session_state['attempted_file_process'] = True
            if api_key_valid:
                # Start the file processing
                with st.spinner('Reading, chunking and embedding file ....'):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    # Chunking
                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks:{len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embedding_cost: 4f}')

                    # Embeddings
                    vector_store = create_embeddings(chunks)

                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully.')
                
                # Clear the file processing flag after successful processing
                st.session_state['attempted_file_process'] = False
            else:
                # If the API key is not valid, display the error message
                st.error("Unable to process the file without an OpenAI API key")
        else:
            # If no file is being processed, ensure the flag is not set
            st.session_state['attempted_file_process'] = False

    # Asking and Getting Answer
    q = st.text_input('Ask a question about the content of your file')
    # if q:
    #     if 'vs' in st.session_state:
    #         vector_store = st.session_state.vs
    #         # st.write(f'k: {k}')
    #         answer = ask_and_get_answer(vector_store, q, k)
    #         st.text_area('LLM Answer: ', value=answer)

    if q and api_key_valid:
        # ... [Question processing]
        with st.spinner('Fetching the answer...'):
            try:
                if 'vs' in st.session_state:
                    vector_store = st.session_state.vs
                    answer = ask_and_get_answer(vector_store, q, k)
                    st.text_area('LLM Answer: ', value=answer, height=150)
            except Exception as e:
                    st.error(f"An error occurred: {e}")
                    answer = f"An error occurred: {e}"  # Set answer to the error message

            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            # Current question and answer
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)

            # Check if AuthenticationError is present in the openai.error module

st.write("OpenAI Library Version:", openai.__version__)
