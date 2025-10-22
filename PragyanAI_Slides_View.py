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
with tab2:
    st.header("Quiz from Google Sheets")
    quiz_sheet_url = st.session_state.get('quiz_sheet_url')

    if not quiz_sheet_url:
        st.info("Please submit a Google Quiz Sheet URL in the Configuration tab.")
    else:
        try:
            # Open the spreadsheet
            sh = gc.open_by_url(quiz_sheet_url)

            # Get the list of worksheet (tab) names
            sheet_names = [ws.title for ws in sh.worksheets()]
            selected_sheet = st.selectbox("Select Quiz Sheet", sheet_names)

            # Load the selected worksheet
            ws = sh.worksheet(selected_sheet)
            quiz_df = pd.DataFrame(ws.get_all_records())

            # Initialize session state for question index
            if 'question_index' not in st.session_state:
                st.session_state['question_index'] = 0

            total_questions = len(quiz_df)
            q_idx = st.session_state['question_index']

            if total_questions == 0:
                st.warning("Selected quiz sheet is empty.")
            else:
                # Show current question
                question = quiz_df.iloc[q_idx]
                st.write(f"Question {q_idx + 1} of {total_questions}:")
                st.write(question['Questions'])

                # Display options
                options = [question.get(f'Option {c}') for c in ['A','B','C','D','E'] if question.get(f'Option {c}')]
                user_answer = st.radio("Choose your answer:", options, key=f"q_{q_idx}")

                submitted = st.button("Submit Answer")

                if submitted:
                    correct_answer = question['Answer']
                    if user_answer == correct_answer:
                        st.success("Correct!")
                    else:
                        st.error(f"Incorrect. Correct answer is: {correct_answer}")

                    st.markdown(f"**Explanation:** {question.get('Explanation', 'No explanation provided.')}")

                    # Next question button
                    if q_idx + 1 < total_questions:
                        if st.button("Next Question"):
                            st.session_state['question_index'] += 1
                            st.experimental_rerun()
                    else:
                        st.success("You have completed the quiz!")
                        # Reset quiz index for a new run
                        if st.button("Restart Quiz"):
                            st.session_state['question_index'] = 0
                            st.experimental_rerun()

        except Exception as e:
            st.error(f"Failed to load quiz: {e}")

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
            Use the following pieces of context to answer the question at the end.
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
