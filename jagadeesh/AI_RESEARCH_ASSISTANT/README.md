# AI Research Assistant

This project is an AI-powered research assistant that integrates with **Zotero** to fetch academic papers, extract text, summarize content, generate questions, answer them, and validate results using **LangChain Agents** and **Google Gemini AI**.

## 🚀 Features
- **Fetch PDFs** from Zotero and extract text.
- **Summarization Agent**: Generates concise summaries of research papers.
- **Q&A Agent**: Extracts topics, generates questions, and provides answers.
- **Validation Agent**: Ensures accuracy by comparing summarized vs. original content.
- **Streamlit UI** for interactive usage.

---

--python 10,11,12

##First step:
pip install -r requirements.txt

##Technologies Used:

Python 🐍
Streamlit (UI)
Zotero API (Fetching PDFs)
LangChain (Agents & LLM Integration)
Google Gemini AI (Summarization, Q&A)
NLTK & Gensim (Topic modeling)
Scikit-learn & NumPy (Validation)
Matplotlib & Seaborn (Visualization)


##Running the Application
--To launch the Streamlit UI, run:

streamlit run main.py
