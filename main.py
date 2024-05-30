import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util
import google.generativeai as palm
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
import os

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    :heart: Welcome to The World of Chatting With Pdf 
    Make You Ans Your Question According to the PDF 
    So , You Wont need to Read Whole PDF For some Specific question :heart:
 
    - [My Linkdin](linkedin.com/in/ritesh-tambe-aa1a70205)
    - [Project Github Link]()


    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Ritesh Tambe')

def main():
    st.header(":bookmark_tabs: Chat with PDF 💬")

    # upload a PDF file
    
    pdf = st.file_uploader(":point_right: Upload your PDF :point_down:", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        pdf_text = []
        for page in pdf_reader.pages:
          pdf_text.append(page.extract_text())



        
    if pdf is not None:
    # Now, you can use pdf_text as your documents
        documents = pdf_text
        # Load a pre-trained sentence embedding model
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Sample list of sentences
        sentences = documents

        #  User input sentence
        if not None:
            user_input = st.text_input("Ask questions about your PDF file:")
        else :
            print("Question Is Not Provided")

        # Encode sentences and user input
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        user_input_embedding = model.encode(user_input, convert_to_tensor=True)

        # Calculate cosine similarities between user input and sentences
        cosine_scores = util.pytorch_cos_sim(user_input_embedding, sentence_embeddings)

        # Sort sentences by similarity score
        sorted_scores, sorted_indices = cosine_scores[0].sort(descending=True)
        for i in range(1):  # Adjust the number of top similar sentences you want to display
            similar_sentence_index = sorted_indices[i]
            similar_sentence = sentences[similar_sentence_index]
            similarity_score = sorted_scores[i]
            # print(f"{similar_sentence}: Similarity Score = {similarity_score:.4f}")
        messages=['hello answer the quesAIzaSyCpzY5teMmWo8xQYQtFdTvNK72d9JgHEXstion with the reference contextgiven by the user']

        palm.configure(api_key="AIzaSyCpzY5teMmWo8xQYQtFdTvNK72d9JgHEXs")

        defaults = {
            'model': 'models/chat-bison-001',
            'temperature': 0.9,
            'candidate_count': 1,
            'top_k': 40,
            'top_p': 0.95,
        }
        context = "be a question answer solver assistant which answers the given questions with the given reference context only and if it is not in the"
        examples = [
            [  "hello answer the question with the reference in the given context",
                "ya sure , give me the reference context"
            ],
            [
                'who is jack , reference context - jack was a fruit seller in the tokyo',
                'jack was a fruit seller'
            ]
        ]
        def lang_model(question):
            messages.append(question)
            response = palm.chat(
                **defaults,
                context=context,
                examples=examples,
                messages=messages)
            g=response.last # Response of the AI to your most recent request
            def replace(string, replacements):
                new_string = ""
                for character in string:
                    if character in replacements:
                        new_string += replacements[character]
                    else:
                        new_string += character
                return new_string
            string=g
            replacements = {"\n": " ", "\r": " ", " ": " ", "\t": " "}
            new_string = replace(string,replacements)
            messages.append(new_string)
            return new_string
        st.write(lang_model(f'answer the question in 50 words with given context ,question-{user_input},reference_context-{similar_sentence}'))
    else:
        st.write(":broken_heart: PDF IS NOT UPLOADED :broken_heart:")

 

if __name__ == '__main__':
    main()