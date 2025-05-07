import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.agents import initialize_agent, Tool
import gensim
from gensim import corpora
from gensim.models import LdaModel
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

GOOGLE_API_KEY = ""
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=GOOGLE_API_KEY)
genai_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

def embed_content(input_text):
    try:
        embedding = genai_embeddings.embed_query(input_text)
        return embedding
    except Exception as e:
        st.error(f"Error embedding content: {e}")
        return None
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]

def extract_topics(text, num_words=5):
    tokens = preprocess_text(text)
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    topics = [" ".join([word for word, _ in topic]) for _, topic in lda_model.show_topics(num_words=num_words, formatted=False)]
    return topics

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    return list(vectorizer.get_feature_names_out())

def preprocess_original_content(original_text):
    return {
        "keywords": extract_keywords(original_text),
        "topics": extract_topics(original_text)
    }

def validate_summary(original_data, summary):
    original_keywords = original_data["keywords"]
    keyword_overlap = len(set(original_keywords) & set(summary.split())) / len(original_keywords) if original_keywords else 0
    original_topics = original_data["topics"]
    topic_overlap = sum(len(set(orig.split()) & set(summary.split())) / len(set(orig.split())) for orig in original_topics) / len(original_topics) if original_topics else 0
    result = {
        "keyword_overlap": round(keyword_overlap, 2),
        "topic_overlap": round(topic_overlap, 2),
        "valid": keyword_overlap > 0.5 and topic_overlap > 0.5
    }
    return json.dumps(result)

def validate_answers(summarized_answers, original_answers):
    summarized_embeddings = [embed_content(ans) for ans in summarized_answers if ans.strip()]
    original_embeddings = [embed_content(ans) for ans in original_answers if ans.strip()]
    summarized_embeddings = [e for e in summarized_embeddings if e is not None]
    original_embeddings = [e for e in original_embeddings if e is not None]
    if not summarized_embeddings or not original_embeddings:
        return json.dumps({"similarity": 0.0})
    similarity_matrix = cosine_similarity(np.array(summarized_embeddings), np.array(original_embeddings))
    avg_similarity = np.mean(similarity_matrix)
    return json.dumps({"similarity": round(avg_similarity, 2)})

summary_validation_tool = Tool(
    name="Summary Validation",
    func=lambda inputs: validate_summary(
        json.loads(inputs)["original_data"] if isinstance(inputs, str) else inputs["original_data"],
        json.loads(inputs)["summary"] if isinstance(inputs, str) else inputs["summary"]
    ),
    description="Validates if the summary retains key topics and keywords from the original text."
)

qa_validation_tool = Tool(
    name="Q&A Validation",
    func=lambda inputs: validate_answers(
        json.loads(inputs)["summarized_answers"] if isinstance(inputs, str) else inputs["summarized_answers"],
        json.loads(inputs)["original_answers"] if isinstance(inputs, str) else inputs["original_answers"]
    ),
    description="Validates Q&A responses by comparing summarized and original answers using embeddings."
)

validation_agent = initialize_agent(
    tools=[summary_validation_tool, qa_validation_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False,
    handle_parsing_errors=True
)
