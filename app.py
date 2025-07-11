import os
from flask import Flask, request, jsonify, render_template
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.search import Search

# --- CONFIGURATION (for "My First Project") ---
PROJECT_ID = "onyx-authority-453720-p9"
LOCATION = "europe-west2"
DATA_STORE_ID = "final-pdf-store_1752184701608"
# --- END CONFIGURATION ---

# Initialize Flask App
app = Flask(__name__)

# Initialize Vertex AI Client
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure the Search Tool (do this once at startup)
search_tool = Search(
    data_store=Search.DataStore(
        data_store_id=DATA_STORE_ID,
        location="eu",
    )
)

# Configure the Generative Model (do this once at startup)
model = GenerativeModel.from_pretrained("gemini-1.5-pro-preview-0514")

@app.route("/")
def index():
    """Renders the main chat page."""
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query_vertex():
    """Receives a query, sends it to Vertex AI, and returns the response."""
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Perform the search and generate the answer
        response = model.generate_content(
            user_query,
            tools=[search_tool]
        )

        # Extract the text part of the response
        if response.candidates and response.candidates[0].content.parts:
            answer = response.candidates[0].content.parts[0].text
            return jsonify({"answer": answer})
        else:
            return jsonify({"answer": "No content was generated in the response."})

    except Exception as e:
        # Return a detailed error message if anything goes wrong
        error_message = f"An internal error occurred: {str(e)}"
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
