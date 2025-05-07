import re
import nltk
import gensim
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import GoogleGenerativeAI
import json
from agent2 import preprocess_text, extract_topics

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

GOOGLE_API_KEY = ""  # Replace with your key
llm = GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=GOOGLE_API_KEY)


def answer_with_summary_content(question, summary_content):
    prompt = f"""
    Content:
    {summary_content}

    Question:
    {question}

    Answer based on this content alone (if the content lacks information, state: "The content does not provide information on this topic."):
    """
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"Error answering with summary content: {e}"

def answer_with_original_content(question, original_content):
    prompt = f"""
    Content:
    {original_content}

    Question:
    {question}

    Answer based on this content alone (if the content lacks information, state: "The content does not provide information on this topic."):
    """
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"Error answering with original content: {e}"


def generate_questions_directly(topics):
    prompt = (
        f"Based on the following topics: {', '.join(topics)}, "
        "generate 5 relevant questions about the content. "
        "Format each question as a numbered item (e.g., '1. What...')."
    )
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"Error generating questions: {e}"

def summary_content_answer_tool_func(input_json):
    try:
        data = json.loads(input_json)
        question = data["question"]
        summary_content = data["summary_content"]
        return answer_with_summary_content(question, summary_content)
    except Exception as e:
        return f"Error in summary_content_answer_tool: {e}"

summary_content_answer_tool = Tool(
    name="Summary Content Answer",
    func=summary_content_answer_tool_func,
    description="Answers a question using the provided summarized content. Provide a JSON with keys 'question' and 'summary_content'."
)
def original_content_answer_tool_func(input_text):
    # Expect input_text to be a string in the format "question: <question> | content: <original_content>"
    try:
        parts = input_text.split(" | ")
        question = parts[0].replace("question: ", "").strip()
        original_content = parts[1].replace("content: ", "").strip()
        return answer_with_original_content(question, original_content)
    except Exception as e:
        return f"Error in original_content_answer_tool: {e}"

original_content_answer_tool = Tool(
    name="Original Content Answer",
    func=original_content_answer_tool_func,
    description="Answers a question using the provided original content. Provide input as 'question: <question> | content: <original_content>'."
)

agent2 = initialize_agent(
    tools=[summary_content_answer_tool, original_content_answer_tool],
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=False,
    max_iterations=1000,
    max_execution_time=120,
    handle_parsing_errors=True
)
