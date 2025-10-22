import streamlit as st
import pandas as pd
import sqlite3
from llama_index import VectorStoreIndex, SimpleDirectoryReader  # Or your vector DB
from groq import Groq  # GROQ Llama integration
import requests

# ---- Configurable URLs ----
slides_url = st.sidebar.text_input("Google Slides URL")
video_url = st.sidebar.text_input("YouTube/MP4 Video URL")
quiz_sheet_url = st.sidebar.text_input("Google Sheet URL for Quiz")

# ---- SQL DB Connection ----
conn = sqlite3.connect('student_performance.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS performance
             (student TEXT, quiz_name TEXT, correct INTEGER, wrong INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

# ---- Tab Structure ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 PPT & Video", "📝 Quiz", "🤖 RAG Ask", "🌐 Web/YouTube Search"
])

# ---- Tab1: PPT Slide + Video ----
with tab1:
    st.header("Presentation & Video")
    if slides_url and "docs.google.com/presentation" in slides_url:
        embed_url = slides_url.replace("/edit", "/embed?start=false&loop=false&delayms=3000")
        st.components.v1.iframe(embed_url, height=480)
    else:
        st.info("Paste a valid Google Slides URL.")
    # Video playback support via streamlit-player package for YouTube/MP4
    if video_url:
        try:
            from streamlit_player import st_player
            st_player(video_url)
        except Exception:
            st.video(video_url)

# ---- Tab2: List Quizzes, Conduct & Evaluate ----
with tab2:
    st.header("Quiz from Google Sheet")
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    # Sheets API setup: Use st.secrets for real deployment
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('your_google_service_account.json', scope)
    client = gspread.authorize(creds)
    try:
        sh = client.open_by_url(quiz_sheet_url)
        worksheet = sh.sheet1
        df_quiz = pd.DataFrame(worksheet.get_all_records())
        quiz_names = df_quiz['quiz_name'].unique()
        selected_quiz = st.selectbox("Select Quiz", quiz_names)
        quiz_questions = df_quiz[df_quiz['quiz_name'] == selected_quiz]
        correct, wrong = 0, 0
        st.write("Answer the following questions:")
        for idx, row in quiz_questions.iterrows():
            st.write(f"Q: {row['Question']}")
            options = [row[f'Option {c}'] for c in 'ABCDE' if row.get(f'Option {c}', None)]
            answer = st.radio("Choose your answer:", options, key=f"quiz_{idx}")
            submit = st.button("Submit Answer", key=f"submit_{idx}")
            if submit:
                if answer == row['Answer']:
                    st.success(f"Correct! {row['Explanation']}")
                    correct += 1
                else:
                    st.error(f"Incorrect. {row['Explanation']}")
                    wrong += 1
        # Save performance to SQL DB
        student = st.text_input("Student Name", key="student_name")
        if st.button("Submit Results"):
            c.execute("INSERT INTO performance (student, quiz_name, correct, wrong) VALUES (?, ?, ?, ?)",
                      (student, selected_quiz, correct, wrong))
            conn.commit()
            st.success("Results submitted!")
        st.write("Performance Log:")
        df_perf = pd.read_sql_query("SELECT * FROM performance", conn)
        st.dataframe(df_perf)
    except Exception as e:
        st.error(f"Cannot load Google Sheet: {e}")

# ---- Tab3: RAG Ask (GROQ + LlamaIndex) ----
with tab3:
    st.header("Ask LLM (RAG with Slide Number)")
    slide_number = st.number_input("Slide Number for Context", min_value=1, step=1)
    user_query = st.text_area("Ask your question")
    if st.button("Ask with RAG"):
        # Embed slide context using LlamaIndex & answer via GROQ API
        index = VectorStoreIndex(SimpleDirectoryReader('./slides_dir').load_data())
        context = index.query(f"slide {slide_number}")[0]['context']  # This depends on your embedding/query API
        groq_client = Groq()
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Context: {context}\nQuestion: {user_query}"
            }]
        )
        st.write("LLM Answer:", completion.choices[0].message.content)

# ---- Tab4: Web/YouTube Search ----
with tab4:
    st.header("Web & YouTube Search")
    topic = st.text_input("Enter search topic")
    if st.button("Search Web"):
        # Use requests/serpapi for Google search, display summary
        params = {
            "q": topic,
            "num": 3,
            "hl": "en"
        }
        resp = requests.get("https://serpapi.com/search", params=params)
        if resp.ok:
            results = resp.json().get('organic_results', [])
            for res in results:
                st.markdown(f"**{res['title']}**\n{res['snippet']}\n({res['link']})")
        else:
            st.error("Web search failed.")
    if st.button("Search YouTube"):
        yt_params = {
            "part": "snippet",
            "q": topic,
            "maxResults": 2,
            "type": "video",
            "key": "YOUR_YOUTUBE_API_KEY"
        }
        yt_resp = requests.get("https://www.googleapis.com/youtube/v3/search", params=yt_params)
        if yt_resp.ok:
            for item in yt_resp.json().get('items', []):
                vid_url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                from streamlit_player import st_player
                st_player(vid_url)
        else:
            st.error("YouTube search failed.")
