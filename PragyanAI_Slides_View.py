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
st.write(st.secrets)
creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
credentials = Credentials.from_service_account_info(creds_dict)
gc = gspread.authorize(credentials)

# ----- Config -----
slides_url = st.sidebar.text_input("Google Slides URL")
video_url = st.sidebar.text_input("YouTube/MP4 URL")
quiz_sheet_url = st.sidebar.text_input("Google Sheet URL for Quizzes")

# --- SQLite DB setup ---
conn = sqlite3.connect('student_perf.db', check_same_thread=False)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS performance (
    student TEXT, quiz_name TEXT, correct INTEGER, wrong INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
''')

# --- Streamlit Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 PPT & Video", "📝 Quiz", "🤖 RAG Ask", "🌐 Web & YouTube"
])

# --- Tab 1: Slide & Video ---
with tab1:
    st.header("Slides & Video")
    if slides_url:
        embed_url = slides_url.replace("/edit", "/embed?start=false&loop=false&delayms=3000")
        st.components.v1.iframe(embed_url, height=480)
    if video_url:
        try:
            from streamlit_player import st_player
            st_player(video_url)
        except Exception:
            st.video(video_url)

# --- Tab 2: Quiz ---
with tab2:
    st.header("Quiz from Google Sheets")

    try:
        sh = gc.open_by_url(quiz_sheet_url)
        quiz_df = pd.DataFrame(sh.sheet1.get_all_records())
        quiz_names = quiz_df['quiz_name'].unique()
        selected_quiz = st.selectbox("Select Quiz", quiz_names)
        q_df = quiz_df[quiz_df['quiz_name'] == selected_quiz]

        correct, wrong = 0, 0
        student_name = st.text_input("Student Name", key='student')
        for idx, q in q_df.iterrows():
            st.write(f"Q{idx+1}: {q['Question']}")
            options = [q[f'Option {c}'] for c in 'ABCDE' if q.get(f'Option {c}', None)]
            answer = st.radio("Your answer:", options, key=f"q{idx}")
            if st.button(f"Submit Question {idx+1}", key=f"sub{idx}"):
                selected_opt = answer
                correct_answer = q['Answer']
                if selected_opt == correct_answer:
                    st.success("Correct!")
                    correct += 1
                else:
                    st.error(f"Wrong! Correct answer: {correct_answer}")
                    wrong += 1
        if st.button("Save Results") and student_name:
            c.execute('INSERT INTO performance (student, quiz_name, correct, wrong) VALUES (?, ?, ?, ?)',
                      (student_name, selected_quiz, correct, wrong))
            conn.commit()
            st.success("Results saved!")

        st.write(f"Score: {correct}/{len(q_df)}")
    except Exception as e:
        st.error(f"Error loading quiz: {e}")

# --- Tab 3: RAG with GROQ + FastEmbed ---
with tab3:
    st.header("Ask Question (RAG with GROQ + FastEmbed)")

    slide_number = st.number_input("Slide number for context", min_value=1, step=1)
    question_text = st.text_area("Your Question")

    if st.button("Get Answer"):
        try:
            vectorstore = FAISS.load_local("faiss_index", FastEmbedEmbeddings())
            retriever = vectorstore.as_retriever()

            template = """
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
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

            answer = rag_chain.invoke({"context": question_text, "question": question_text})
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error in RAG answer: {e}")

# --- Tab 4: Web & YouTube Search ---
with tab4:
    st.header("Web & YouTube Search")
    query = st.text_input("Search Topic")
    if st.button("Search Web"):
        search_url = f"https://api.serpapi.com/search.json?q={query}&api_key=YOUR_SERPAPI_KEY"
        response = requests.get(search_url)
        if response.ok:
            results = response.json().get('organic_results', [])
            for res in results:
                st.markdown(f"**{res['title']}**\n{res['snippet']}\n[{res['link']}]")
        else:
            st.error("Web search failed")
    if st.button("Search YouTube"):
        yt_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key=YOUR_YOUTUBE_API_KEY&maxResults=2"
        resp = requests.get(yt_url)
        if resp.ok:
            items = resp.json().get('items', [])
            from streamlit_player import st_player
            for video in items:
                vid_id = video['id'].get('videoId')
                if vid_id:
                    st_player(f"https://www.youtube.com/watch?v={vid_id}")
        else:
            st.error("YouTube search failed")
