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
with tab3:
    import os
    from urllib.parse import urlparse
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    from googleapiclient.errors import HttpError
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate

    # Helper: extract Google Slides ID from URL
    def get_presentation_id(slides_url):
        path = urlparse(slides_url).path
        parts = path.split('/')
        if "d" in parts:
            idx = parts.index("d")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        raise ValueError("Cannot extract Presentation ID from URL.")

    # Extract slide text content using Slides API with error handling
    def extract_slide_texts(slides_url):
        if 'slides_credentials' not in globals() or not globals()['slides_credentials']:
            st.error("Slides API credentials not available. Cannot extract text.")
            return []
        try:
            presentation_id = get_presentation_id(slides_url)
            service = build('slides', 'v1', credentials=globals()['slides_credentials'])
            presentation = service.presentations().get(presentationId=presentation_id).execute()
            slides = presentation.get('slides', [])
            slide_texts = []
            for slide in slides:
                slide_text_parts = []
                for element in slide.get('pageElements', []):
                    shape = element.get('shape')
                    if shape and 'text' in shape:
                        text_content = ""
                        for te in shape['text'].get('textElements', []):
                            if 'textRun' in te:
                                content = te['textRun'].get('content')
                                if content:
                                    text_content += content.strip() + " "
                        if text_content:
                            slide_text_parts.append(text_content.strip())
                slide_content = " ".join(slide_text_parts).strip()
                if slide_content:
                    slide_texts.append(slide_content)
            return slide_texts
        except HttpError as e:
            if e.resp.status == 400 and "operation is not supported" in str(e).lower():
                st.error(
                    "Error extracting slide texts (400 - Operation Not Supported):\n"
                    "File is not a native Slides presentation.\n"
                    "Use File → Make a copy and try the new file."
                )
            else:
                st.error(f"Error extracting slide texts (HTTP {e.resp.status}). Check sharing permissions.")
            return []
        except Exception as e:
            st.error(f"Error during slide extraction: {e}")
            return []

    # Load FAISS vectorstore safely
    def load_faiss_index():
        try:
            if os.path.exists("faiss_index/index.faiss"):
                return FAISS.load_local(
                    "faiss_index",
                    FastEmbedEmbeddings(),
                    allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            st.error(f"Error loading FAISS vectorstore: {e}")
            return None

    # Initialize Google Slides API credentials once
    if 'slides_credentials' not in globals() or not globals()['slides_credentials']:
        creds_dict = dict(st.secrets["google_service_account"])
        creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
        scopes = ["https://www.googleapis.com/auth/presentations.readonly"]
        globals()['slides_credentials'] = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)

    st.header("RAG Question Answering based on PPT Slides")

    slide_url_input = st.text_input("Enter Google Slides URL", value=st.session_state.get('slides_url', ''))
    if slide_url_input:
        current_url = st.session_state.get('slides_url')
        if current_url != slide_url_input:
            st.session_state['slides_url'] = slide_url_input
            st.session_state['all_slides_texts'] = []
            st.session_state['faiss_ready'] = False
        embed_url = slide_url_input.replace("/edit", "/embed?start=false&loop=false&delayms=3000")
        st.components.v1.iframe(embed_url, height=480)
    else:
        st.info("Please enter Google Slides URL.")

    # Extract slide texts if not loaded yet
    if slide_url_input and not st.session_state.get('all_slides_texts'):
        with st.spinner("Extracting slide texts..."):
            texts = extract_slide_texts(slide_url_input)
            st.session_state['all_slides_texts'] = texts

    # Vector DB build/load button (before slide content)
    if st.button("Build or Load Vector DB (Required for Q&A)"):
        if not st.session_state.get('all_slides_texts'):
            st.error("Slide texts unavailable. Please enter a valid Google Slides URL.")
        else:
            if not os.path.exists("faiss_index"):
                os.makedirs("faiss_index")
            with st.spinner("Building vector DB..."):
                vectorstore = FAISS.from_texts(st.session_state['all_slides_texts'], FastEmbedEmbeddings())
                vectorstore.save_local("faiss_index")
                st.session_state['faiss_ready'] = True
                st.success("Vector DB built and saved.")

    # Load from disk if not yet loaded
    if not st.session_state.get('faiss_ready', False):
        if os.path.exists("faiss_index/index.faiss"):
            st.session_state['faiss_ready'] = True
            st.success("Loaded Vector DB from disk.")

    all_slides_texts = st.session_state.get('all_slides_texts', [])
    # Slide number selection
    if all_slides_texts:
        slide_num = st.number_input(
            "Select slide number to base context on",
            min_value=1, max_value=len(all_slides_texts), value=1
        )
        slide_idx = slide_num -1
        indices = [slide_idx]
        if slide_idx > 0:
            indices.append(slide_idx -1)
        if slide_idx < len(all_slides_texts) -1:
            indices.append(slide_idx +1)
        indices = sorted(indices)
        context_slides = [all_slides_texts[i] for i in indices]
        combined_context = "\n\n".join(context_slides)
        st.subheader(f"Context from Slide {slide_num} and its neighbors")
        st.write(combined_context)
    else:
        combined_context = ""
        st.info("Slide texts not loaded yet.")

    # Question input and LLM answer generation
    if st.session_state.get('faiss_ready', False):
        question = st.text_area("Ask a question based on the selected slide context")
        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            elif not combined_context.strip():
                st.warning("Selected slide context is empty.")
            else:
                with st.spinner("Generating answer..."):
                    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
                    prompt_template = """
You are an expert tutor. Using the following context (selected slide and neighbors), answer clearly and simply:

Context:
{context}

Question:
{question}

Answer clearly for students:
"""
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    input_text = prompt.format(context=combined_context, question=question)

                    # Invoke llm with text input and extract just answer text
                    try:
                        response = llm.invoke(input_text)
                        if hasattr(response, 'content'):
                            answer = response.content
                        elif isinstance(response, dict) and 'content' in response:
                            answer = response['content']
                        else:
                            answer = str(response)
                    except Exception:
                        # Fallback: send as chat messages list
                        response = llm.invoke([{"role": "user", "content": input_text}])
                        if hasattr(response, 'content'):
                            answer = response.content
                        elif isinstance(response, dict) and 'content' in response:
                            answer = response['content']
                        else:
                            answer = str(response)

                    if 'chat_logs' not in st.session_state:
                        st.session_state['chat_logs'] = []
                    st.session_state['chat_logs'].append({"question": question, "answer": answer})

                    st.markdown(f"**Q:** {question}")
                    st.markdown(f"**A:** {answer}")

    # Show chat history
    if 'chat_logs' in st.session_state and st.session_state['chat_logs']:
        st.subheader("Previous Questions and Answers")
        for chat in reversed(st.session_state['chat_logs']):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")


# --- Tab 4: Web & YouTube Search ---
import gspread
import requests
import streamlit as st
from streamlit_player import st_player
from google.oauth2 import service_account

# Google Sheets authorization
creds_dict = dict(st.secrets["google_service_account"])
creds_dict["private_key"] = creds_dict["private_key"].replace('\\n', '\n')
scopes = ['https://www.googleapis.com/auth/spreadsheets.readonly']
credentials = service_account.Credentials.from_service_account_info(creds_dict, scopes=scopes)
gc = gspread.authorize(credentials)

def get_website_list(sheet_url):
    try:
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet("WebSite")  # Correct tab name
        sites = ws.col_values(1)
        return [s.strip() for s in sites if s.strip() and not s.lower().startswith("website")]
    except gspread.exceptions.WorksheetNotFound:
        st.error("Worksheet 'WebSite' not found in the Google Sheet.")
        return []
    except Exception as e:
        st.error(f"Error reading WebSite sheet: {e}")
        return []

def get_youtube_channels(sheet_url):
    try:
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet("YouTube")  # Correct tab name
        channels = ws.col_values(1)
        return [c.strip() for c in channels if c.strip() and not c.lower().startswith("channel")]
    except gspread.exceptions.WorksheetNotFound:
        st.error("Worksheet 'YouTube' not found in the Google Sheet.")
        return []
    except Exception as e:
        st.error(f"Error reading YouTube sheet: {e}")
        return []

def serpapi_site_search(query, site, api_key):
    search_query = f"site:{site} {query}"
    params = {"q": search_query, "api_key": api_key}
    url = "https://serpapi.com/search.json"
    res = requests.get(url, params=params)
    if res.ok:
        return res.json().get("organic_results", [])
    else:
        st.error(f"SerpAPI search failed for {site}: {res.text}")
        return []

def serpapi_general_search(query, api_key):
    params = {"q": query, "api_key": api_key}
    res = requests.get("https://serpapi.com/search.json", params=params)
    if res.ok:
        return res.json().get("organic_results", [])
    else:
        st.error(f"SerpAPI general search failed: {res.text}")
        return []

def youtube_search_channel(query, channel_id, youtube_api_key):
    params = {
        "part": "snippet",
        "channelId": channel_id,
        "q": query,
        "maxResults": 5,
        "type": "video",
        "key": youtube_api_key
    }
    res = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
    if res.ok:
        return res.json().get("items", [])
    else:
        st.error(f"YouTube search failed for channel {channel_id}: {res.text}")
        return []

def youtube_general_search(query, youtube_api_key):
    params = {
        "part": "snippet",
        "q": query,
        "maxResults": 5,
        "type": "video",
        "key": youtube_api_key
    }
    res = requests.get("https://www.googleapis.com/youtube/v3/search", params=params)
    if res.ok:
        return res.json().get("items", [])
    else:
        st.error(f"YouTube general search failed: {res.text}")
        return []

with tab4:
    st.header("Search Across Trusted Websites & YouTube Channels")

    sheet_url = st.text_input("Enter Google Sheet URL with Website & YouTube Lists")
    if st.button("Submit Sheet URL"):
        st.session_state['sheet_url'] = sheet_url
        st.success("Sheet URL saved.")

    saved_sheet_url = st.session_state.get('sheet_url')
    topic = st.text_input("Enter search topic or keyword")

    if st.button("Search") and saved_sheet_url and topic:
        websites = get_website_list(saved_sheet_url)
        youtube_channels = get_youtube_channels(saved_sheet_url)

        serpapi_key = st.secrets.get("SERPAPI_API_KEY")
        youtube_api_key = st.secrets.get("YOUTUBE_API_KEY")

        if not serpapi_key or not youtube_api_key:
            st.error("Please set SERPAPI_API_KEY and YOUTUBE_API_KEY in Streamlit secrets.")
        else:
            # Site restricted Google searches
            st.subheader("Site-Restricted Google Search Results")
            for site in websites:
                st.markdown(f"### Results from {site}")
                results = serpapi_site_search(topic, site, serpapi_key)
                if results:
                    for r in results[:5]:
                        st.markdown(f"**[{r['title']}]({r['link']})**")
                        st.write(r.get('snippet', ''))
                else:
                    st.write("No results found.")

            # General Google Search
            st.subheader("General Google Search Results")
            general_results = serpapi_general_search(topic, serpapi_key)
            if general_results:
                for r in general_results[:5]:
                    st.markdown(f"**[{r['title']}]({r['link']})**")
                    st.write(r.get('snippet', ''))
            else:
                st.write("No results found.")

            # YouTube search on specified channels with embedded players
            st.subheader("YouTube Videos on Listed Channels")
            for channel_id in youtube_channels:
                st.markdown(f"#### Videos from Channel {channel_id}")
                videos = youtube_search_channel(topic, channel_id, youtube_api_key)
                if videos:
                    for v in videos:
                        vid_id = v['id'].get('videoId')
                        title = v['snippet']['title']
                        desc = v['snippet']['description']
                        url = f"https://www.youtube.com/watch?v={vid_id}"
                        st.markdown(f"[{title}]({url})")
                        st.write(desc)
                        if vid_id:
                            st_player(url)
                else:
                    st.write("No videos found.")

            # General YouTube search and embed
            st.subheader("General YouTube Video Search")
            general_videos = youtube_general_search(topic, youtube_api_key)
            if general_videos:
                for v in general_videos:
                    vid_id = v['id'].get('videoId')
                    title = v['snippet']['title']
                    desc = v['snippet']['description']
                    url = f"https://www.youtube.com/watch?v={vid_id}"
                    st.markdown(f"[{title}]({url})")
                    st.write(desc)
                    if vid_id:
                        st_player(url)
            else:
                st.write("No videos found.")


    '''
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
    '''
