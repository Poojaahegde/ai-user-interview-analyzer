# AI User Interview Analyzer 🎙️ — From Raw Transcripts to Product Insights

> **Stop spending 4 hours synthesizing user interviews. Let AI extract themes, pain points, and opportunity signals in minutes.**
>
> [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org) [![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io) [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
>
> ---
>
> ## 🚀 Product Overview
>
> **The Problem:** User research is one of the highest-leverage activities a PM can do — but synthesis is the bottleneck. After conducting 10 user interviews, a PM faces 5–8 hours of manual work: transcribing notes, reading through transcripts, identifying recurring themes, and writing up insights. By the time synthesis is done, the sprint has moved on.
>
> **The Solution:** AI User Interview Analyzer takes raw interview transcripts (plain text or CSV), automatically extracts recurring themes using NLP clustering, identifies pain points and positive signals, generates a structured insight summary per interview, and outputs a prioritized opportunity map — all in a Streamlit dashboard.
>
> **The Impact:**
> - ⏱ Reduces user research synthesis from **5–8 hours → 15 minutes** per research sprint
> - - 🎯 Surfaces the **top 5 recurring themes** across all interviews automatically
>   - - 💡 Separates **pain points from positive signals** with sentiment-aware extraction
>     - - 📋 Produces a **shareable insight report** ready for roadmap and sprint planning meetings
>      
>       - ---
>
> ## 🎯 Why This Matters (Product Perspective)
>
> User research velocity is a competitive advantage. Teams that can synthesize research faster make better product decisions faster. This tool doesn't replace the PM's judgment — it removes the grunt work so PMs can spend time on the insight layer, not the extraction layer. The output is designed to plug directly into sprint planning: "Here are the 5 themes from this week's interviews, ranked by frequency."
>
> ---
>
> ## 🧠 AI/ML Explanation
>
> | Component | Technique | Why |
> |---|---|---|
> | **Sentence Segmentation** | NLTK sent_tokenize | Splits transcript into analyzable units |
> | **Theme Extraction** | TF-IDF + KMeans Clustering | Groups semantically similar statements into recurring themes without supervision |
> | **Pain Point Detection** | Sentiment analysis (TextBlob) + keyword heuristics | Flags sentences with negative sentiment + pain-related language |
> | **Key Quote Extraction** | Sentence scoring (frequency + sentiment intensity) | Surfaces the most representative quote per theme |
> | **Summary Generation** | Extractive summarization (sentence ranking) | Produces concise, accurate summaries without hallucination risk |
>
> **Architecture:**
> ```
> Raw Transcript Text
>         ↓
> Sentence Segmentation (NLTK)
>         ↓
> TF-IDF Vectorization → KMeans Clustering (Themes)
>         ↓
> Sentiment Analysis per sentence (TextBlob)
>         ↓
> Pain Point Detection (negative sentiment + keywords)
>         ↓
> Key Quote Extraction (top sentence per theme)
>         ↓
> Structured Insight Report Output
> ```
>
> ---
>
> ## 🛠 Tech Stack
>
> | Layer | Technology |
> |---|---|
> | UI | Streamlit |
> | NLP | NLTK, TextBlob, scikit-learn (TF-IDF, KMeans) |
> | Data Processing | Pandas |
> | Visualization | Matplotlib, WordCloud |
> | Language | Python 3.8+ |
>
> ---
>
> ## 📊 Sample Output
>
> **Input:** 3 user interview transcripts (avg. 800 words each) about a project management SaaS tool
>
> **Themes Detected:**
>
> | Theme | Frequency | Avg Sentiment | Top Quote |
> |---|---|---|---|
> | Onboarding Complexity | 14 mentions | -0.52 (Negative) | *"I spent 2 hours just trying to figure out how to invite my team"* |
> | Notification Overload | 11 mentions | -0.41 (Negative) | *"I turned off all notifications because they were just too much"* |
> | Search & Discovery | 9 mentions | -0.38 (Negative) | *"I can never find old tasks — search is basically useless for me"* |
> | Mobile Experience | 7 mentions | -0.29 (Negative) | *"I use it on mobile sometimes but it's really painful"* |
> | Reporting Loved | 6 mentions | +0.61 (Positive) | *"The reporting charts are honestly the best feature — I use them every week"* |
>
> **PM Recommendation Output:**
> ```
> HIGH PRIORITY: Onboarding (14x, -0.52 sentiment) — Largest pain cluster.
> ACTION: Simplify team invitation flow; add guided setup wizard.
>
> HIGH PRIORITY: Notifications (11x, -0.41 sentiment) — Users turning off all alerts.
> ACTION: Implement smart notification preferences; digest mode.
>
> OPPORTUNITY: Reporting (6x, +0.61 sentiment) — Strongest positive signal.
> ACTION: Double down; expand reporting templates; promote as differentiator.
> ```
>
> ---
>
> ## 📸 Demo Instructions
>
> ```bash
> # 1. Clone the repo
> git clone https://github.com/Poojaahegde/ai-user-interview-analyzer.git
> cd ai-user-interview-analyzer
>
> # 2. Install dependencies
> pip install -r requirements.txt
>
> # 3. Launch the app
> streamlit run app.py
> ```
>
> Open **http://localhost:8501** in your browser.
>
> **Upload format:** Plain text file (.txt) or CSV with a `transcript` column.
>
> **Sample input:**
> ```
> I: How do you currently manage your projects?
> P: Mostly in spreadsheets, honestly. We tried [Product] but the onboarding was
>    really confusing. I spent like two hours just figuring out how to add team members.
> I: What features do you use most?
> P: The reports are great actually. I send them to my manager every Friday.
>    But search is really bad — I can never find old stuff.
> ```
>
> ---
>
> ## 🎯 Product Thinking Layer
>
> ### 👥 Target Users
> - **Product Managers** synthesizing user research after discovery sprints
> - - **UX Researchers** turning interview notes into actionable themes for design teams
>   - - **Startup founders** doing early customer discovery without a dedicated research team
>    
>     - ### 😣 Pain Points Solved
>     - 1. **Research synthesis bottleneck** — 5–8 hours per sprint spent reading and coding transcripts manually
>       2. 2. **Inconsistent thematic coding** — different PMs categorize themes differently; AI provides consistent classification
>          3. 3. **Key insights get buried** — in 10 long transcripts, the most important quote is easy to miss
>             4. 4. **No shareable artifact** — raw notes can't be shared; this tool produces a structured summary report
>               
>                5. ### 🧩 Key Product Decisions Made
>                6. - **Unsupervised clustering (KMeans) over predefined categories:** Product-specific themes can't be predicted in advance — let the data define the categories
>                   - - **Extract key quotes per theme:** The most powerful artifact in user research is a verbatim user quote — the tool surfaces the best one per theme automatically
>                     - - **Pain/positive split in output:** Different action types (fix vs. amplify) require different signals to be separated
>                       - - **Streamlit dashboard over CLI:** Research synthesis is a visual, interactive workflow — a UI helps PMs explore themes, not just read a text output
>                        
>                         - ### 🗺 Future Roadmap
>                         - | Priority | Feature | Expected Impact |
>                         - |---|---|---|
>                         - | P0 | LLM-powered theme naming (GPT/Claude) | Auto-generate descriptive theme labels instead of "Theme 1" |
>                         - | P0 | Multi-interview comparison dashboard | Spot patterns across 10+ interviews visually |
>                         - | P1 | Export to Notion/Confluence formatted report | Direct integration into PM workflow |
>                         - | P1 | Speaker separation (Interviewer vs. Participant) | Filter to analyze only participant statements |
>                         - | P2 | Opportunity scoring (frequency × sentiment intensity) | Automatic prioritization of themes by business impact |
>                         - | P2 | Interview guide generator from themes | Close the loop — insights feed back into next interview design |
>                         - | P3 | Integration with Dovetail / UserTesting | Enterprise research workflow automation |
>                        
>                         - ---
>
> ## 📁 Project Structure
>
> ```
> ai-user-interview-analyzer/
> ├── app.py               # Main Streamlit dashboard
> ├── analyzer.py          # Core NLP pipeline: segmentation, clustering, sentiment
> ├── requirements.txt     # Python dependencies
> ├── sample_transcript.txt # Example interview transcript for testing
> └── README.md            # This file
> ```
>
> ---
>
> ## 🔗 Related Projects in This Portfolio
> - [**FeedbackSense**](https://github.com/Poojaahegde/FeedbackSense-AI-Product-Feedback-Analyzer) — AI feedback clustering for product teams
> - - [**EmotionLoop**](https://github.com/Poojaahegde/EmotionLoop) — Emotion detection in user text
>   - - [**ScopeCreep**](https://github.com/Poojaahegde/scopecreep) — Real-time scope drift detection
>    
>     - ---
>
> *Built as part of an AI PM portfolio — demonstrating how AI can accelerate user research synthesis and help PMs ship user-centered products faster.*
