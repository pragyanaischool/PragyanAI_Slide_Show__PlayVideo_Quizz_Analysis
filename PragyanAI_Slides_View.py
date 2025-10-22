import streamlit as st
import pandas as pd
import sqlite3
import requests
from google.oauth2 import service_account
import gspread
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Load credentials with correct OAuth scopes ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

creds_dict = dict(st.secrets["google_service_account"])
creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
credentials = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
gc = gspread.authorize(credentials)

# --- SQLite DB ---
conn = sqlite3.connect('student_perf.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS performance (
    student TEXT,
    quiz_name TEXT,
    correct INTEGER,
    wrong INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')

# --- Default sample URLs ---
DEFAULT_SLIDES_URL = "https://docs.google.com/presentation/d/1fAkBtQJQgPFfUXK22rFzLuGhv05MViLg/edit"
DEFAULT_QUIZ_SHEET_URL = "https://docs.google.com/spreadsheets/d/184z7lY3kkMBZ2GA8R_Q7WniZ4EHWC2umcse4yzQHMjw/edit"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=-nQRj_MOKa8&t=6s"

# --- Configuration Form ---
st.title("AI Seminar Configuration")

with st.form("config_form"):
    slides_url = st.text_input("Google Slides URL", value=st.session_state.get('slides_url', DEFAULT_SLIDES_URL))
    quiz_sheet_url = st.text_input("Google Quiz Sheet URL", value=st.session_state.get('quiz_sheet_url', DEFAULT_QUIZ_SHEET_URL))
    video_url = st.text_input("Video URL (YouTube or Google Drive)", value=st.session_state.get('video_url', DEFAULT_VIDEO_URL))
    submit_config = st.form_submit_button("Submit All URLs")

    if submit_config:
        st.session_state['slides_url'] = slides_url
        st.session_state['quiz_sheet_url'] = quiz_sheet_url
        st.session_state['video_url'] = video_url
        st.success("URLs saved successfully! Navigate to other tabs.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 PPT & Video", "📝 Quiz", "🤖 RAG Ask", "🌐 Web & YouTube Search"
])

# --- Tab 1: Slides & Video ---
with tab1:
    st.header("Presentation & Video")
    slides_url = st.session_state.get('slides_url')
    if slides_url:
        embed_url = slides_url.replace("/edit", "/embed?start=false&loop=false&delayms=3000")
        st.components.v1.iframe(embed_url, height=480)
    else:
        st.info("Please submit the Google Slides URL in the Configuration tab.")

    video_url = st.session_state.get('video_url')
    if video_url:
        try:
            from streamlit_player import st_player
            st_player(video_url)
        except:
            st.video(video_url)
    else:
        st.info("Please submit the video URL in the Configuration tab.")

# --- Tab 2: Quiz ---
def safe_rerun():
    import streamlit as st
    try:
        st.experimental_rerun()
    except AttributeError:
        from streamlit.runtime.scriptrunner import RerunException
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        raise RerunException(add_script_run_ctx())

with tab2:
    st.header("Quiz from Google Sheets")
    quiz_sheet_url = st.session_state.get('quiz_sheet_url')

    if not quiz_sheet_url:
        st.info("Please submit a Google Quiz Sheet URL in the Configuration tab.")
    else:
        try:
            sh = gc.open_by_url(quiz_sheet_url)
            sheet_names = [ws.title for ws in sh.worksheets()]
            selected_sheet = st.selectbox("Select Quiz Sheet", sheet_names)

            ws = sh.worksheet(selected_sheet)
            quiz_df = pd.DataFrame(ws.get_all_records())

            total_questions = len(quiz_df)

            if total_questions == 0:
                st.warning("Selected quiz sheet is empty.")
            else:
                if 'question_index' not in st.session_state:
                    st.session_state['question_index'] = 0
                if 'answer_submitted' not in st.session_state:
                    st.session_state['answer_submitted'] = False
                    st.session_state['user_answer'] = None

                q_idx = st.session_state['question_index']
                question = quiz_df.iloc[q_idx]

                st.write(f"Question {q_idx + 1} of {total_questions}:")
                st.write(question['Questions'])

                options = [question.get(f'Option {c}') for c in ['A','B','C','D','E'] if question.get(f'Option {c}')]
                
                if not st.session_state['answer_submitted']:
                    user_answer = st.radio("Choose your answer:", options, key=f"q_{q_idx}")
                    submit = st.button("Submit Answer")
                    if submit:
                        st.session_state['user_answer'] = user_answer
                        st.session_state['answer_submitted'] = True
                        safe_rerun()
                else:
                    correct_answer = question['Answer']
                    if st.session_state['user_answer'] == correct_answer:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. Correct answer is: {correct_answer}")

                    st.markdown(f"**Explanation:** {question.get('Explanation', 'No explanation provided.')}")

                    if q_idx + 1 < total_questions:
                        if st.button("Next Question"):
                            st.session_state['question_index'] += 1
                            st.session_state['answer_submitted'] = False
                            st.session_state['user_answer'] = None
                            safe_rerun()
                    else:
                        st.success("You have completed the quiz!")
                        if st.button("Restart Quiz"):
                            st.session_state['question_index'] = 0
                            st.session_state['answer_submitted'] = False
                            st.session_state['user_answer'] = None
                            safe_rerun()

                # Optional: Save performance
                student_name = st.text_input("Student Name")
                if st.button("Save Quiz Performance") and student_name:
                    correct_count = sum(quiz_df['Answer'] == st.session_state.get('user_answer'))
                    wrong_count = total_questions - correct_count
                    c.execute(
                        'INSERT INTO performance (student, quiz_name, correct, wrong) VALUES (?, ?, ?, ?)',
                        (student_name, selected_sheet, correct_count, wrong_count)
                    )
                    conn.commit()
                    st.success("Quiz performance saved!")

        except Exception as e:
            st.error(f"Failed to load quiz: {e}")

# --- Tab 3: RAG Ask ---
import os

def load_faiss_index():
    return FAISS.load_local(
        "faiss_index",
        FastEmbedEmbeddings(),
        allow_dangerous_deserialization=True
    )

def extract_slide_texts(slides_url):
    # Placeholder for real extraction using Slides API
    # Demo slide texts for example
    return [
        "Slide 1: Introduction to Deep Learning. Neural networks are inspired by the human brain.",
        "Slide 2: Convolutional Neural Networks (CNNs) are used in image recognition.",
        "Slide 3: Recurrent Neural Networks handle sequential data like language.",
    ]

with tab3:
    st.header("RAG Question Answering based on PPT Slides")

    # 1. Provide PPT URL and embed presentation
    slide_url_input = st.text_input("Enter Google Slides URL", value=st.session_state.get('slide_url', ''))
    if slide_url_input:
        st.session_state['slide_url'] = slide_url_input
        embed_url = slide_url_input.replace("/edit", "/embed?start=false&loop=false&delayms=3000")
        st.components.v1.iframe(embed_url, height=480)
    else:
        st.info("Please enter Google Slides URL to embed presentation.")

    # 2. Load / Build Vector DB button
    build_db = st.button("Build or Load Vector DB from Presentation Content")
    if build_db and slide_url_input:
        st.info("Extracting slide texts and building vector DB (this may take a moment)...")
        all_slides_texts = extract_slide_texts(slide_url_input)
        vectorstore = FAISS.from_texts(all_slides_texts, FastEmbedEmbeddings())
        vectorstore.save_local("faiss_index")
        st.success("Vector DB built and saved successfully.")
        st.session_state['faiss_ready'] = True
        st.session_state['all_slides_texts'] = all_slides_texts

    # Load vector DB if already built
    faiss_ready = st.session_state.get('faiss_ready', False)
    if not faiss_ready:
        if os.path.exists("faiss_index/index.faiss"):
            st.session_state['faiss_ready'] = True
            st.session_state['all_slides_texts'] = extract_slide_texts(slide_url_input)
            st.success("FAISS vectorstore loaded from disk.")
        else:
            st.warning("Vector DB is not built yet. Please build it using the button above.")

    # 3. Select Slide Number
    all_slides_texts = st.session_state.get('all_slides_texts', [])
    total_slides = len(all_slides_texts)
    if faiss_ready and total_slides > 0:
        slide_num = st.number_input("Select slide number to refer to", min_value=1, max_value=total_slides)

        slide_idx = slide_num - 1
        selected_slide_text = all_slides_texts[slide_idx]

        # 4. Show slide text box
        st.subheader(f"Content of Slide {slide_num}")
        st.text_area("Slide Text Content", value=selected_slide_text, height=180)

        # 5. Ask question and get answer
        question = st.text_area("Ask a question about this slide")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    vectorstore = load_faiss_index()
                    retriever = vectorstore.as_retriever()

                    prompt_template = """
You are an expert explaining concepts based on the following slide content.
Use the context to answer the question clearly and simply.

Context:
{context}

Question:
{question}

Explain in an understandable way suitable for students.
"""
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = chain.invoke({
                        "context": selected_slide_text,
                        "question": question
                    })

                    # Save Q&A to chat history
                    if 'chat_logs' not in st.session_state:
                        st.session_state['chat_logs'] = []
                    st.session_state['chat_logs'].append({"question": question, "answer": answer})

                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")

                except Exception as e:
                    st.error(f"Error in RAG processing: {e}")

        # 6. Display chat history
        if 'chat_logs' in st.session_state and st.session_state['chat_logs']:
            st.subheader("Previous Questions and Answers")
            for entry in reversed(st.session_state['chat_logs']):
                st.markdown(f"**Q:** {entry['question']}")
                st.markdown(f"**A:** {entry['answer']}")
    else:
        st.info("Build the vector DB and ensure slides are loaded first.")

# --- Tab 4: Web & YouTube Search ---
with tab4:
    st.header("Web and YouTube Search")
    query = st.text_input("Enter search topic")

    if st.button("Search Web"):
        search_endpoint = f"https://api.serpapi.com/search.json?q={query}&api_key=YOUR_SERPAPI_KEY"
        res = requests.get(search_endpoint)
        if res.ok:
            results = res.json().get('organic_results', [])
            for r in results:
                st.markdown(f"**{r['title']}**")
                st.write(r.get('snippet', ''))
                st.markdown(f"[Read more]({r.get('link', '#')})")
        else:
            st.error("Failed to perform web search.")

    if st.button("Search YouTube"):
        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key=YOUR_YOUTUBE_API_KEY&maxResults=3"
        resp = requests.get(yt_url)
        if resp.ok:
            from streamlit_player import st_player
            items = resp.json().get("items", [])
            for item in items:
                vid_id = item['id'].get('videoId')
                if vid_id:
                    st_player(f"https://www.youtube.com/watch?v={vid_id}")
        else:
            st.error("Failed to perform YouTube search.")
