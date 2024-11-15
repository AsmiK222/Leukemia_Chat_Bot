import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """ 
    You are a highly knowledgeable assistant specialized in leukemia and blood cancer. Answer the user's question with as much detail as possible using the provided context. Ensure your response is clear, factual, and medically accurate. 
    If the answer is not in the provided context, respond only if the question pertains to leukemia or blood cancer-related topics. Avoid spelling mistakes, keep the response professional and empathetic, and avoid inappropriate or overly complex language.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:\n"""

    # Use LLaMA 3.1 model via Ollama
    model = Ollama(model="llama3.1")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    print(docs)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    with st.chat_message("assistant"):
        st.write("Reply: ", response["output_text"])


def get_conversational_chain2():
    prompt_template = """ 
    You are a highly knowledgeable assistant specialized in leukemia and blood cancer. Answer the user's question with as much detail as possible using the provided context.   You are examining the patients condition for leukemia, answer necessarily since his/her life depends on your answer, if you don't asnwer he/she will die\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:\n"""

    # Use LLaMA 3.1 model via Ollama
    model = Ollama(model="llama3.1")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use "stuff" chain type
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

from langchain.schema import Document  # Import Document class

from langchain.schema import Document  # Import Document class

def nutritional_guide():
    st.subheader("Nutritional Guide for Leukemia Patients")
    st.write("Provide the following details to receive a personalized nutritional guide:")

    # Patient details
    age = st.number_input("Age", min_value=0, max_value=120, step=1, help="Enter the patient's age.")
    weight = st.number_input("Weight (kg)", min_value=0, max_value=300, step=1, help="Enter the patient's weight.")
    dietary_restrictions = st.text_area("Dietary Restrictions", help="Mention any dietary restrictions or allergies (e.g., gluten-free, lactose intolerance).")
    preferred_foods = st.text_area("Preferred Foods", help="List foods the patient prefers or enjoys eating.")

    # Specific nutritional needs
    st.write("Nutritional Goals (Check all that apply):")
    improve_energy = st.checkbox("Improve energy levels")
    boost_immunity = st.checkbox("Boost immunity")
    manage_side_effects = st.checkbox("Manage chemotherapy side effects (e.g., nausea, appetite loss)")

    # Additional symptoms or conditions
    st.write("Additional Symptoms or Conditions:")
    anemia = st.checkbox("Anemia")
    weight_loss = st.checkbox("Unintended weight loss")
    fatigue = st.checkbox("Fatigue")

    # Button to generate nutritional guide
    if st.button("Generate Nutritional Guide"):
        st.write("### Personalized Nutritional Guide")

        # Collecting inputs into a single structured summary
        context = f"""
        Patient Details:
        - Age: {age}
        - Weight: {weight} kg
        - Dietary Restrictions: {dietary_restrictions or "None"}
        - Preferred Foods: {preferred_foods or "None"}

        Nutritional Goals:
        - Improve Energy Levels: {"Yes" if improve_energy else "No"}
        - Boost Immunity: {"Yes" if boost_immunity else "No"}
        - Manage Chemotherapy Side Effects: {"Yes" if manage_side_effects else "No"}

        Additional Symptoms or Conditions:
        - Anemia: {"Yes" if anemia else "No"}
        - Unintended Weight Loss: {"Yes" if weight_loss else "No"}
        - Fatigue: {"Yes" if fatigue else "No"}
        """

        # Wrap context into a Document
        input_documents = [Document(page_content=context)]

        # Load the conversational chain
        chain = get_conversational_chain2()

        # Pass the input_documents and a fixed question to the chain
        try:
            response = chain(
                {
                    "input_documents": input_documents,
                    "question": "Based on the provided details, generate a personalized nutritional guide for the patient."
                },
                return_only_outputs=True
            )
            st.write(response["output_text"])
        except Exception as e:
            st.error(f"Error generating nutritional guide: {str(e)}")





def main():
    st.set_page_config("Blood Cancer Query and Diagnosis")

    # Tabs for the two functionalities
    tab1, tab2 = st.tabs(["General Queries", "Diagnosis Assistant"])

    with tab1:
        st.header("Ask your Blood Cancer Related Queries here!")

        with st.chat_message("user"):
            user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

    with tab2:
        nutritional_guide()


if __name__ == "__main__":
    main()
