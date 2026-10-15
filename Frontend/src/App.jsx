
import { useState } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function handleAsk() {
    if (!question.trim() || loading) return;

    const userQuestion = question.trim();

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userQuestion },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: userQuestion }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Could not connect to the backend. Check that FastAPI is running.",
          sources: [],
        },
      ]);
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Technical Manual QA</h1>
        <p>Ask questions about the NASA Systems Engineering Handbook</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>How can I help?</h2>
            <p>Ask Question</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div className={`message ${message.role}`} key={index}>
            <strong>
              {message.role === "user" ? "You" : "Assistant"}
            </strong>

            <div className="answer-content">
  <ReactMarkdown>{message.content}</ReactMarkdown>
</div>

{message.sources?.length > 0 && (
  <div className="sources">
    <strong>Sources</strong>

    {message.sources.map((source, i) => (
      <div key={i}>
        {source.source || "NASA Handbook"}
        {" — PDF page: "}
        {source.page ?? "Unknown"}
      </div>
    ))}
  </div>
)}
          </div>
        ))}

        {loading && <p className="loading">Searching the handbook...</p>}
      </main>

      <div className="input-area">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleAsk();
          }}
          placeholder="Ask a question about the handbook..."
          disabled={loading}
        />

        <button onClick={handleAsk} disabled={loading || !question.trim()}>
          {loading ? "Thinking..." : "Ask"}
        </button>
      </div>
    </div>
  );
}

export default App;
