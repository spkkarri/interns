import streamlit as st
import time
import random
import json
import seaborn as sns
import matplotlib.pyplot as plt
from langchain.schema import Document
from agent1 import fetch_pdf_from_zotero, download_pdf, extract_text_from_pdf, summarization_agent
from agent2 import extract_topics, agent2, generate_questions_directly
from agent3 import preprocess_original_content, validation_agent, embed_content
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import re

def safe_run(agent, text, max_retries=5, initial_delay=1):
    """Handles retry logic for running an agent task."""
    retries = 0
    while retries < max_retries:
        try:
            result = agent.run(f"Summarize this text chunk using the Summarization Tool: {text}")
            return result
        except Exception as e:
            wait_time = initial_delay * (2 ** retries) + random.uniform(0, 1)
            st.warning(f"Error: {str(e)}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            retries += 1
    st.error("Exceeded retries. Falling back to direct summarization.")
    from agent1 import summarize_text  # Fallback
    return summarize_text(text, agent.llm)

def main():
    st.title("AI RESEARCH ASSISTANT")
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Summarize PDF", "Generate Questions", "Validate Results"])

    # Session state to persist data
    if "original_content" not in st.session_state:
        st.session_state.original_content = ""
    if "final_summary" not in st.session_state:
        st.session_state.final_summary = ""
    if "questions" not in st.session_state:
        st.session_state.questions = []
    if "summarized_answers" not in st.session_state:
        st.session_state.summarized_answers = []
    if "original_answers" not in st.session_state:
        st.session_state.original_answers = []

    if page == "Home":
        st.header("Welcome")
        st.write("This app allows you to summarize(agent1) research papers from Zotero, generate questions(agent2), and validate results(agent3) using AI agents.")

    elif page == "Summarize PDF":
        st.header("Summarize PDF from Zotero")
        titles, keys = fetch_pdf_from_zotero()
        if titles and keys:
            selected_title = st.selectbox("Select a PDF to summarize", titles)
            selected_key = keys[titles.index(selected_title)]
            if st.button("Summarize"):
                with st.spinner("Downloading and processing PDF..."):
                    pdf_path = download_pdf(selected_key)
                    if pdf_path:
                        original_content = extract_text_from_pdf(pdf_path)
                        if original_content:
                            st.session_state.original_content = original_content
                            chunk_size = 10000
                            chunks = [Document(page_content=original_content[i:i + chunk_size]) for i in range(0, len(original_content), chunk_size)]
                            summarized_chunks = []
                            progress_bar = st.progress(0)
                            for i, chunk in enumerate(chunks):
                                summary = safe_run(summarization_agent, chunk.page_content)
                                summarized_chunks.append(summary)
                                progress_bar.progress((i + 1) / len(chunks))
                                time.sleep(1)
                            st.session_state.final_summary = "\n\n".join(summarized_chunks)
                            st.success("Summarization complete!")
                            st.subheader("Summary")
                            st.text_area("Final Summary", st.session_state.final_summary, height=300)
                            with open("summary.txt", "w", encoding="utf-8") as f:
                                f.write(st.session_state.final_summary)
                            st.download_button("Download Summary", data=st.session_state.final_summary, file_name="summary.txt")
                            os.unlink(pdf_path)
                        else:
                            st.error("Failed to extract text from PDF.")
        else:
            st.error("No PDFs available in Zotero.")

    elif page == "Generate Questions":
        st.header("Generate Questions and Answers")
        if st.session_state.original_content:
            if st.button("Extract Topics and Generate Questions"):
                with st.spinner("Extracting topics and generating questions..."):
                    topics = extract_topics(st.session_state.original_content)
                    st.subheader("Extracted Topics")
                    for topic in topics:
                        st.write(topic)
                    questions_raw = agent2.run(f"Use the Question Generator tool to generate three questions from these topics: {', '.join(topics)}")
                    generated_questions = [q.strip() for q in questions_raw.split("\n") if q.strip() and re.match(r"^\d+\.\s*.+", q)]
                    st.session_state.questions = [re.sub(r"^\d+\.\s*", "", q) for q in generated_questions[:3]]
                    st.subheader("Generated Questions")
                    for i, q in enumerate(st.session_state.questions, 1):
                        st.write(f"{i}. {q}")

            if st.session_state.questions:
                if st.button("Generate Answers"):
                    with st.spinner("Generating answers..."):
                        st.session_state.summarized_answers = []
                        st.session_state.original_answers = []
                        for i, question in enumerate(st.session_state.questions, 1):
                            # Summarized Answer
                            summary_payload = json.dumps({"question": question, "summary_content": st.session_state.final_summary})
                            summarized_answer = agent2.run(f"Use the Summary Content Answer tool with the following JSON input: {summary_payload}")
                            st.session_state.summarized_answers.append(summarized_answer)
                            # Original Answer
                            original_answer_input = f"question: {question} | content: {st.session_state.original_content[:1000]}"
                            original_answer = agent2.run(f"Use the Original Content Answer tool with the following input: {original_answer_input}")
                            st.session_state.original_answers.append(original_answer)
                            st.subheader(f"Question {i}: {question}")
                            st.write(f"**Summarized Answer:** {summarized_answer}")
                            st.write(f"**Original Answer:** {original_answer}")
        else:
            st.warning("Please summarize a PDF first on the 'Summarize PDF' page.")

    elif page == "Validate Results":
        st.header("Validate Results")
        if (
            st.session_state.final_summary 
            and st.session_state.summarized_answers 
            and st.session_state.original_answers
        ):
            if st.button("Run Validation"):
                with st.spinner("Validating summary and answers..."):
                    try:
                        # Preprocess original content
                        original_data = preprocess_original_content(st.session_state.original_content)

                        # Prepare input
                        summary_validation_input = {
                            "original_data": original_data, 
                            "summary": st.session_state.final_summary
                        }

                        # Call validation agent
                        summary_validation_raw = validation_agent.run(
                            f"Use the Summary Validation tool with the following inputs and return the result as JSON: {json.dumps(summary_validation_input)}"
                        )

                        # Ensure valid JSON response
                        try:
                            summary_result = json.loads(summary_validation_raw.strip())
                        except json.JSONDecodeError:
                            st.error("Validation agent returned invalid JSON.")
                            st.stop()

                        # Check if expected keys exist
                        required_keys = {"keyword_overlap", "topic_overlap", "valid"}
                        if not all(key in summary_result for key in required_keys):
                            st.error("Unexpected response format from validation agent.")
                            st.json(summary_result)  # Display raw output for debugging
                            st.stop()

                        # Display validation results
                        st.subheader("Summary Validation")
                        st.write(f"**Keyword Overlap:** {summary_result['keyword_overlap']}")
                        st.write(f"**Topic Overlap:** {summary_result['topic_overlap']}")
                        st.write(f"**Valid:** {summary_result['valid']}")

                        # If summary is valid, print a success message
                        if summary_result["valid"]:
                            st.success("✅ The summary is valid and of high quality!")

                    except Exception as e:
                        st.error(f"An error occurred during validation: {e}")

                    # Q&A Validation
                    qa_validation_input = {
                        "summarized_answers": st.session_state.summarized_answers,
                        "original_answers": st.session_state.original_answers
                    }
                    qa_validation_raw = validation_agent.run(
                        f"Use the Q&A Validation tool with the following inputs and return the result as JSON: {json.dumps(qa_validation_input)}"
                    )
                    qa_result = json.loads(qa_validation_raw) if qa_validation_raw.startswith("{") else None
                    st.subheader("Q&A Validation")
                    if qa_result:
                        st.write(f"Similarity Score: {qa_result['similarity']}")
                        # Plot heatmap
                        summarized_embeddings = [embed_content(ans) for ans in st.session_state.summarized_answers if ans.strip()]
                        original_embeddings = [embed_content(ans) for ans in st.session_state.original_answers if ans.strip()]
                        summarized_embeddings = [e for e in summarized_embeddings if e is not None]
                        original_embeddings = [e for e in original_embeddings if e is not None]
                        if summarized_embeddings and original_embeddings:
                            similarity_matrix = cosine_similarity(np.array(summarized_embeddings), np.array(original_embeddings))
                            fig, ax = plt.subplots()
                            sns.heatmap(similarity_matrix, annot=True, cmap="viridis", fmt=".2f", ax=ax)
                            ax.set_title("Cosine Similarity Heatmap: Q&A")
                            ax.set_xlabel("Original Answers")
                            ax.set_ylabel("Summarized Answers")
                            st.pyplot(fig)
                    else:
                        st.error("Failed to validate Q&A.")
        else:
            st.warning("Please complete summarization and question answering first.")

if __name__ == "__main__":
    main()