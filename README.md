# 🤖 HR Chatbot with FAISS + GPT-4o

An interactive chatbot that answers HR policy questions using FAISS vector search and GPT-4o via GitHub-provided OpenAI token.

---

## 🛠 Features

- 💬 Chat interface using Streamlit
- 📁 HR documents embedded using `sentence-transformers` + FAISS
- 🔍 Context-aware question rephrasing
- 🤖 GPT-4o responses via GitHub Token (`https://models.github.ai/inference`)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hr-chatbot.git
cd hr-chatbot
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create a `.env` file

```env
GITHUB_TOKEN=your_github_provided_token_here
```

---

## 📁 Folder Structure

```
├── faiss_index/              # Vector store (auto-generated)
├── hr-policies/              # Your HTML files (input)
├── hr-chatbot.py             # Main Streamlit chatbot app
├── hr-upload.py              # Script to build FAISS index from HTML
├── requirements.txt
└── README.md
```

---

## 🚀 Usage

### 1. Build FAISS Vector Index (run once)

Make sure your `hr-policies/` folder contains all the HTML documents.

```bash
python hr-upload.py
```

### 2. Start the Chatbot UI

```bash
streamlit run hr-chatbot.py
```

You’ll be able to chat with the bot about HR policies in a Streamlit web UI.

---

## 🧠 Notes

* GPT-4o is accessed via GitHub’s [models.github.ai](https://models.github.ai/inference) inference endpoint.
* Your `GITHUB_TOKEN` must be valid and have access to that endpoint.
* Ensure `faiss_index/` exists before running the chatbot.

---

## 📝 License

MIT – free to use, modify, and share.

