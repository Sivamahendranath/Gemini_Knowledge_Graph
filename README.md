# Gemini_Knowledge_Graph
Gemini_AI_Knowledge_Graph
# Knowledge_Graph_with_LLM
Knowledge graph exploration Using the Gemini AI LLM
🔎 Knowledge Graph Builder & Visualizer
An intelligent entity and relationship extraction tool built with Streamlit, powered by Google Gemini AI, that visualizes data as an interactive knowledge graph.

🚀 Features
🧠 Extracts entities (people, orgs, locations, products, etc.) and relationships from raw text using Gemini LLM.

📂 Accepts input from:

Text

URLs (webpages)

🧱 Stores and updates entities in a SQLite3 database.

📊 Computes entity importance scores (weights) based on:

Connections

Attributes

Frequency

🌐 Renders an interactive network graph with expansion, filtering, and customization options.

🗂 Search and filter entities dynamically.

⬇️ Download extracted data as CSV.

🧰 Tech Stack

Component	Tech Used
Frontend	Streamlit
LLM	Google Gemini 1.5 Flash
Database	SQLite3
Visualization	Pyvis + HTML + CSS
NLP / Parsing	Regex, Rule-Based Heuristics
File Handling	PDFs, URLs, .txt
📦 Requirements
Install dependencies with:
pip install -r requirements.txt
requirements.txt (sample)
txt
Copy
Edit
streamlit
google-generativeai
python-dotenv
pandas
numpy
requests
beautifulsoup4
pyvis
Pillow
🔑 Setup
Clone the repository

bash
Copy
Edit
git clone https://github.com/Sivamahendranath/Gemini_Knowledge_Graph/blob/main/backup.py
Set up your .env file with the Gemini API key:

ini
Copy
Edit
GEMINI_API=your_google_gemini_api_key
Run the app

bash
Copy
Edit
streamlit run app.py
🧠 How It Works
User uploads a file or provides a URL.

The system uses Gemini to extract:

Entities (with types & attributes)

Relationships between entities

The extracted info is parsed and stored in a SQLite database.

Graph is rendered based on current entity state and connections.

Users can interactively expand nodes, filter by types, and explore relationships.

📊 Example Use Cases
Academic research: analyzing article content

Business intelligence: extracting org-people relations

News entity linking

Legal/contract document understanding

📸 Screenshots

Home Page	Graph Viewer	Entity List
(add screenshots here)		
🧩 Future Improvements
Integration with local LLMs (e.g., Ollama)

RAG-based refinement of entity context

Multi-language support

Real-time updates and collaboration

📄 License
This project is open-source and available under the MIT License.
