import streamlit as st
import pandas as pd
import sqlite3
import requests
from google.oauth2.service_account import Credentials
import gspread
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Fix multiline private_key from secrets ---
creds_dict = dict(st.secrets["google_service_account"])
creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
credentials = Credentials.from_service_account_info(creds_dict)
gc = gspread.authorize(credentials)

# --- Setup SQLite DB for quiz results ---
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

# --- Default URLs from user input ---
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
        st.success("URLs saved! Navigate across tabs below.")

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
        st.info("Please submit a valid Google Slides URL in the Configuration tab.")

    video_url = st.session_state.get('video_url')
    if video_url:
        try:
            from streamlit_player import st_player
            st_player(video_url)
        except Exception:
            st.video(video_url)
    else:
        st.info("Please submit a video URL in the Configuration tab.")

# --- Tab 2: Quiz ---
with tab2:
    st.header("Quiz from Google Sheets")
    quiz_sheet_url = st.session_state.get('quiz_sheet_url')

    if not quiz_sheet_url:
        st.info("Please submit a Google Quiz Sheet URL in the Configuration tab.")
    else:
        try:
            sh = gc.open_by_url(quiz_sheet_url)
            quiz_df = pd.DataFrame(sh.sheet1.get_all_records())

            quiz_names = quiz_df['quiz_name'].unique()
            selected_quiz = st.selectbox("Select Quiz", quiz_names)
            q_df = quiz_df[quiz_df['quiz_name'] == selected_quiz]

            student_name = st.text_input("Enter Student Name")

            correct, wrong = 0, 0
            for idx, row in q_df.iterrows():
                st.write(f"Q{idx+1}: {row['Questions']}")
                options = [row.get(f'Option {opt}', '') for opt in "ABCDE" if row.get(f'Option {opt}', '')]
                answer = st.radio("Choose your answer:", options, key=f"q{idx}")
                if st.button(f"Submit Question {idx+1}", key=f"submit_{idx}"):
                    if answer == row['Answer']:
                        st.success("Correct!")
                        correct += 1
                    else:
                        st.error(f"Incorrect. Correct answer: {row['Answer']}")
                        wrong += 1

            if st.button("Save Results") and student_name:
                c.execute('INSERT INTO performance (student, quiz_name, correct, wrong) VALUES (?, ?, ?, ?)',
                          (student_name, selected_quiz, correct, wrong))
                conn.commit()
                st.success("Quiz results saved!")

            st.write(f"Your score: {correct} / {len(q_df)}")

        except Exception as e:
            st.error(f"Failed to load quiz sheet: {e}")

# --- Tab 3: RAG Ask ---
with tab3:
    st.header("RAG Question Answering (GROQ + FastEmbed)")

    question = st.text_area("Enter your question")
    slide_number = st.number_input("Relevant Slide Number (optional)", min_value=1, value=1)

    if st.button("Get Answer"):
        try:
            vectorstore = FAISS.load_local("faiss_index", FastEmbedEmbeddings())
            retriever = vectorstore.as_retriever()

            template = """
                Use the following context to answer the question at the end.
                If you don't know the answer, just say that you don't know. Don't make up an answer.
                {context}
                Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)

            llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke({"context": question, "question": question})
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error in RAG processing: {e}")

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

