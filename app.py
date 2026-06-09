import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from uploads.rag_engine import build_vector_store, answer_question
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload and build vector store."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    # Save file safely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Build RAG vector store from this PDF
    num_chunks = build_vector_store(filepath)

    return jsonify({
        "message": f"PDF processed successfully! Created {num_chunks} chunks.",
        "filename": filename
    })

@app.route("/ask", methods=["POST"])
def ask():
    """Handle user question and return AI answer."""
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)