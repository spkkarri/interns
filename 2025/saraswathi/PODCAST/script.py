import streamlit as st
import os
from datetime import datetime
import google.generativeai as genai
import pdfplumber
import tiktoken
import fitz  # PyMuPDF
from gtts import gTTS
from pydub import AudioSegment
import io
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from gensim import corpora
from gensim.models import LdaModel

# Download NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Backend functions remain unchanged up to process_pdf_into_episodes
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    model_name = "gemini-2.0-flash-exp"
    return genai.GenerativeModel(model_name)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        raise

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    os.makedirs(output_folder, exist_ok=True)
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(output_folder, f"page{page_number+1}_img{img_index+1}.{image_ext}")
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                images.append(image_filename)
        return images
    except Exception as e:
        print(f"Error extracting images from {pdf_path}: {e}")
        return []

def tokenize_and_chunk_document(document, max_chunk_size=1000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(document)
    chunks = [tokenizer.decode(tokens[i:i + max_chunk_size]) for i in range(0, len(tokens), max_chunk_size)]
    return chunks

def generate_podcast_script(model, text_chunks, image_descriptions, episode_number, total_episodes, previous_summary=""):
    chat = model.start_chat(history=[])
    system_prompt = f"""You are a world-class podcast writer, having ghostwritten for famous podcasters like Joe Rogan, Lex Fridman, Ben Shapiro, and Tim Ferris.

Your job is to craft an engaging **audio** podcast dialogue for Episode {episode_number} of a {total_episodes}-episode series based on the provided PDF content chunks. The podcast follows a structured format with three mandatory sections: **Introduction, Main Content, and Conclusion.** The conversation flows naturally between Speaker 1 (Host) and Speaker 2 (Guest), incorporating all textual content from the chunks and image descriptions seamlessly into a single cohesive script. Ensure continuity with previous episodes and a clear transition to the next.

**Speaker Roles**:
- **Speaker 1 (Host):** Engages the Guest with insightful, engaging, and occasionally humorous questions. They are unfamiliar with the document and ask for clarifications, examples, or follow-ups to ensure clarity.
- **Speaker 2 (Guest):** An expert on the document, providing structured, detailed, and engaging explanations. They use examples, metaphors, and anecdotes to enhance understanding.

---
 **Podcast Structure (MANDATORY - STRICTLY ENFORCED)**:
1️ Introduction
   - Speaker 1 (Host) introduces Episode {episode_number} with an engaging summary of this portion of the document’s scope, covering all chunks provided.
   - If this isn’t Episode 1, briefly recap the previous episode using this summary: "{previous_summary}".
   - Clearly state this is part of a {total_episodes}-episode series and set up the topic for this episode.
   - Capture the listener’s attention with a hook relevant to this episode’s content.

2️ Main Content
   - A lively, continuous conversation unfolds between Speaker 1 and Speaker 2, integrating all provided chunks into a single narrative.
   - Speaker 1 asks thoughtful and natural questions, while Speaker 2 provides clear, engaging explanations tied to this episode’s content across all chunks.
   - **Weave image descriptions naturally into the dialogue** without explicit references (e.g., no "as you can see here"), ensuring they fit the flow of the conversation.
   - Ensure the content feels distinct from other episodes while building on the overall narrative.

3️ Conclusion
   - Speaker 1 summarizes the key points from this episode, reflecting all chunks.
   - If this isn’t the final episode, tease the next episode (e.g., "Next time in Episode {episode_number + 1}, we’ll explore...").
   - End with a **single, natural closing remark** that reinforces the series’ continuity (e.g., "Stay tuned for more!").
---

Every line MUST start with **"Speaker 1:"** or **"Speaker 2:"**. Combine all provided chunks into one script with a single Introduction, Main Content, and Conclusion for this episode.
"""
    combined_text = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(text_chunks)])
    combined_images = "\n\n".join([f"Image {i+1}: {desc}" for i, desc in enumerate(image_descriptions)]) if image_descriptions else "No additional images in this section."
    input_message = f"{system_prompt}\n\nText Content for Episode {episode_number}:\n{combined_text}\n\nImage Insights for Episode {episode_number}:\n{combined_images}"

    try:
        response = chat.send_message(input_message, stream=False)
        return response.text
    except Exception as e:
        print(f"Error generating script for Episode {episode_number}: {e}")
        return (
            f"Speaker 1: Welcome to Episode {episode_number} of our {total_episodes}-episode series. "
            f"{'Last time, ' + previous_summary if previous_summary else 'Today, we’re starting fresh.'}\n"
            f"Speaker 2: Unfortunately, we hit a snag with the content, but let’s talk about what we can.\n"
            f"Speaker 1: Fair enough! What’s one takeaway we can share?\n"
            f"Speaker 2: Even when tech fails, resilience keeps us going.\n"
            f"Speaker 1: That’s Episode {episode_number}! "
            f"{'Next time in Episode ' + str(episode_number + 1) + ', we’ll dive deeper.' if episode_number < total_episodes else 'Thanks for joining us!'} "
            f"Stay tuned for more!"
        )

def refine_script_against_document(model, document, generated_script, episode_number, total_episodes):
    chat = model.start_chat(history=[])
    system_prompt = f"""
    You are an expert podcast script editor refining the script for Episode {episode_number} of a {total_episodes}-episode series. Ensure:
    - It accurately represents the key content from the provided document chunk, covering all provided chunks in a single cohesive narrative.
    - Corrects any deviations or inaccuracies compared to the document.
    - Maintains the structure: **Introduction, Main Content, and Conclusion**.
    - Improves clarity, grammar, and flow while keeping it engaging and episode-specific.
    - Integrates image descriptions naturally without breaking continuity.
    - Ensures the script feels like part of a cohesive series with clear narrative progression.
    **ONLY RETURN the refined script** without commentary.

    Document content for this episode:
    {document}

    Generated script:
    {generated_script}
    """
    try:
        response = chat.send_message(system_prompt, stream=False)
        return response.text if hasattr(response, "text") else generated_script
    except Exception as e:
        print(f"Error refining Episode {episode_number}: {e}")
        return generated_script

def analyze_images(image_paths):
    return [f"Description for image {os.path.basename(image)}" for image in image_paths]

def text_to_audio_gtts(text, output_file, lang='en', tld='com', slow=False):
    try:
        tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
        tts.save(output_file)
        return output_file if os.path.exists(output_file) else None
    except Exception as e:
        print(f"Error in gTTS for {output_file}: {e}")
        return None

def generate_episode_audio(script, episode_number, output_folder="podcast_audio"):
    os.makedirs(output_folder, exist_ok=True)
    if not script or not script.strip():
        print(f"Error: Script for Episode {episode_number} is empty or whitespace.")
        return None

    script = "\n".join(line.strip() for line in script.splitlines() if line.strip())
    lines = script.split("\n")
    audio_segments = []
    styles = {"Host": ('com.au', False), "Guest": ('co.in', False)}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("**Speaker 1:**") or line.startswith("Speaker 1:"):
            text = line.replace("**Speaker 1:**", "").replace("Speaker 1:", "").strip()
            if not text:
                continue
            tts = gTTS(text=text, lang='en', tld=styles["Host"][0], slow=styles["Host"][1])
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_segments.append(AudioSegment.from_file(mp3_fp, format="mp3"))
        elif line.startswith("**Speaker 2:**") or line.startswith("Speaker 2:"):
            text = line.replace("**Speaker 2:**", "").replace("Speaker 2:", "").strip()
            if not text:
                continue
            tts = gTTS(text=text, lang='en', tld=styles["Guest"][0], slow=styles["Guest"][1])
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_segments.append(AudioSegment.from_file(mp3_fp, format="mp3"))

    if audio_segments:
        combined = AudioSegment.empty()
        for segment in audio_segments:
            combined += segment + AudioSegment.silent(duration=500)
        final_file = os.path.join(output_folder, f"podcast_episode_{episode_number}.mp3")
        combined.export(final_file, format="mp3")
        return final_file
    else:
        print(f"Warning: No audio segments generated for Episode {episode_number}.")
        return None

def process_pdf_into_episodes(pdf_path, api_key, chunks_per_episode=3, max_chunk_size=1000):
    try:
        document_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(f"Failed to process PDF: {e}")
        return {"episodes": [], "topics": [], "questions": []}

    extracted_images = extract_images_from_pdf(pdf_path)
    image_descriptions = analyze_images(extracted_images)
    text_chunks = tokenize_and_chunk_document(document_text, max_chunk_size=max_chunk_size)

    model = configure_gemini(api_key)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    episodes = []
    total_episodes = (len(text_chunks) + chunks_per_episode - 1) // chunks_per_episode
    previous_summary = ""

    for episode_num in range(1, total_episodes + 1):
        start_idx = (episode_num - 1) * chunks_per_episode
        end_idx = min(start_idx + chunks_per_episode, len(text_chunks))
        episode_chunks = text_chunks[start_idx:end_idx]
        episode_images = image_descriptions[start_idx:end_idx] if start_idx < len(image_descriptions) else []

        generated_script = generate_podcast_script(model, episode_chunks, episode_images, episode_num, total_episodes, previous_summary)
        refined_script = refine_script_against_document(model, "\n".join(episode_chunks), generated_script, episode_num, total_episodes)

        script_filename = f"podcast_script_episode_{episode_num}_{timestamp}.txt"
        with open(script_filename, "w", encoding="utf-8", errors="replace") as f:
            f.write(refined_script)

        audio_file = generate_episode_audio(refined_script, episode_num)

        summary_prompt = f"Summarize the following podcast script in 1-2 sentences for a recap in the next episode:\n\n{refined_script}"
        chat = model.start_chat(history=[])
        try:
            response = chat.send_message(summary_prompt, stream=False)
            previous_summary = response.text.strip()
        except Exception as e:
            print(f"Error generating summary for Episode {episode_num}: {e}")
            previous_summary = f"Episode {episode_num} covered key insights from the document."

        episodes.append({
            "episode_number": episode_num,
            "refined_script_file": script_filename,
            "audio_file": audio_file,
        })

    topics = extract_topics(document_text)  # Still computed for questions but not displayed
    questions = generate_questions_from_text(model, document_text)
    return {"episodes": episodes, "topics": topics, "questions": questions}

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', ' ', text)
    text = re.sub(r'\d{1,2}:\d{2}', ' ', text)
    text = re.sub(r'\w*_\d+_\w+_\w+\.(indd|pdf)', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 3]

def extract_topics(text, num_words=5, num_topics=3):
    chunks = tokenize_and_chunk_document(text, max_chunk_size=1000)
    all_tokens = [preprocess_text(chunk) for chunk in chunks if chunk.strip()]
    if not any(all_tokens):
        return ["No significant topics found."]
    
    dictionary = corpora.Dictionary(all_tokens)
    if len(dictionary) < num_words:
        return ["Insufficient unique terms for topic modeling."]
    
    corpus = [dictionary.doc2bow(tokens) for tokens in all_tokens if tokens]
    try:
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, iterations=100, random_state=42)
        topics = [" ".join([word for word, _ in topic]) for _, topic in lda_model.show_topics(num_words=num_words, formatted=False)]
        return topics
    except Exception as e:
        print(f"Error in LDA topic modeling: {e}")
        return ["Topic extraction failed."]

def generate_questions_from_text(model, text):
    chat = model.start_chat(history=[])
    topics = extract_topics(text)
    if "No significant topics found." in topics:
        return ["Unable to generate meaningful questions due to insufficient content."]
    prompt = (
        f"Generate 5 to 7 thoughtful and simple questions based on the following topics extracted from a document:\n\n"
        f"{topics}\n\n"
        f"### Output Format:\n"
        f"1. [First Question]\n"
        f"2. [Second Question]\n"
      
    )
    try:
        response = chat.send_message(prompt, stream=False)
        questions_text = response.text if hasattr(response, "text") else "No questions generated."
        questions = [line.strip() for line in questions_text.split("\n") if line.strip() and re.match(r"^[1-7]\.", line.strip())]
        return [q.split(". ", 1)[1].strip("[]") for q in questions][:7]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return ["Unable to generate questions due to an error."]

# Streamlit App (Modified to remove Topics tab)
def main():
    st.set_page_config(page_title="Podcast Generator", layout="wide")
    st.title("🎙️ Podcast Generator")
    st.subheader("Turn your PDF into a multi-episode podcast series!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter Google Gemini API Key", type="password", value="")
        chunks_per_episode = st.slider("Chunks per Episode", min_value=1, max_value=10, value=3)
        max_chunk_size = st.slider("Max Chunk Size (tokens)", min_value=500, max_value=2000, value=1000)
        st.info("More chunks or larger chunk sizes result in longer episodes.")

    # Main content
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if not api_key:
            st.warning("Please enter a valid Google Gemini API Key in the sidebar.")
        else:
            pdf_path = f"temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("Generate Podcast Series"):
                with st.spinner("Processing PDF and generating episodes..."):
                    try:
                        result = process_pdf_into_episodes(pdf_path, api_key, chunks_per_episode, max_chunk_size)
                        st.session_state["result"] = result
                        st.success("Podcast series generated successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                    finally:
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)

   
    if "result" in st.session_state:
        result = st.session_state["result"]
        episodes = result["episodes"]
        questions = result["questions"]

        tab1, tab2 = st.tabs(["Episodes", "Questions"])  # Episodes and Questions

        with tab1:
            st.header("Generated Episodes")
            for episode in episodes:
                episode_num = episode["episode_number"]
                script_file = episode["refined_script_file"]
                audio_file = episode["audio_file"]

                st.subheader(f"Episode {episode_num}")
                with st.expander("View Script"):
                    with open(script_file, "r", encoding="utf-8") as f:
                        st.text_area(f"Script {episode_num}", f.read(), height=300)
                    st.download_button(
                        label="Download Script",
                        data=open(script_file, "rb").read(),
                        file_name=os.path.basename(script_file),
                        mime="text/plain"
                    )
                if audio_file and os.path.exists(audio_file):
                    st.audio(audio_file)
                    st.download_button(
                        label="Download Audio",
                        data=open(audio_file, "rb").read(),
                        file_name=os.path.basename(audio_file),
                        mime="audio/mp3"
                    )
                else:
                    st.warning(f"Audio for Episode {episode_num} not generated.")

        with tab2:
            st.header("Generated Questions")
            if questions:
                for idx, question in enumerate(questions, 1):
                    st.write(f"{idx}. {question}")
            else:
                st.write("No questions generated.")

    st.markdown("---")
    st.write("Built with Streamlit | Powered by Google Gemini")

if __name__ == "__main__":
    main()