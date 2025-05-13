# 🤖 Perplexity 2.0

A modern, responsive AI chat interface with integrated web search functionality. **Perplexity 2.0** provides a clean UI similar to [Perplexity.ai](https://www.perplexity.ai), combining conversational AI with real-time search capabilities.

---

## ✨ Features

- **Real-time AI Responses** – Stream AI responses as they're generated
- **Integrated Web Search** – AI can search the web for up-to-date information
- **Conversation Memory** – Maintains context throughout your conversation
- **Search Process Transparency** – Visual indicators show searching, reading, and writing stages
- **Responsive Design** – Clean, modern UI that works across devices

---

## 🏗️ Architecture

Perplexity 2.0 follows a **client-server architecture**:

### Client (Next.js + React)
- Modern React application built with Next.js
- Real-time streaming using Server-Sent Events (SSE)
- Modular components for message display, search status, and input

### Server (FastAPI + LangGraph)
- Python backend using FastAPI
- LangGraph to manage conversation flow between LLM and tools
- Tavily Search API for live web search results
- Streaming responses using Server-Sent Events

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** v18+
- **Python** 3.10+
- Gemini API key
- Tavily API key

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Irshad-Ahmaed/Perplexity-2.0
cd Perplexity-2.0
```

#### 2. Set up the server

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Configure environment variables

Create a `.env` file inside the `server` directory:

```env
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

#### 4. Set up the client

```bash
cd ../client
npm install
```

---

### Running the Application

#### 1. Start the server

```bash
cd server
uvicorn app:app --reload
```

#### 2. Start the client

```bash
cd ../client
npm run dev
```

#### 3. Open in browser

Visit: [http://localhost:3000](http://localhost:3000)

---

## 🔍 How It Works

1. User sends a message through the chat interface
2. Server processes the message using GPT-4o
3. AI decides whether to use search or respond directly
4. If search is needed:
   - Query is sent to Tavily API
   - Results are processed and passed to the AI
   - AI formulates a response based on findings
5. Response is streamed back to the client in real-time
6. Search stages (`searching`, `reading`, `writing`) are shown to the user

---

## 🙏 Acknowledgments

- Inspired by [Perplexity.ai](https://www.perplexity.ai)
- Built with [Next.js](https://nextjs.org), [React](https://reactjs.org), [FastAPI](https://fastapi.tiangolo.com), and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [OpenAI GPT-4o](https://openai.com) and [Tavily Search API](https://tavily.com/)