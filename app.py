import os
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from helper import MedicalBotPipeline

# 1. Initialize Flask application and load environment states
app = Flask(__name__)
load_dotenv()

# Ensure standard project configuration strings are globally mapped
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretmedicalbotkey123")

print("--- Starting Medical Chatbot Server System ---")

# 2. Instantiate and load the RAG backend processor object framework globally
print("Initializing pipeline connection nodes...")
bot_pipeline = MedicalBotPipeline()
is_ready = bot_pipeline.initialize_pipeline()

if is_ready:
    print("🚀 --- Backend Engine Ready! Flask Application Mounted Successfully! ---")
else:
    print("[❌ CRITICAL ERROR] Pipeline initialization failed. Check your API keys and cluster status.")

# ==========================================================================
# Core Routing Framework Operations
# ==========================================================================

@app.route("/", methods=["GET"])
def index():
    """
    Renders the primary dashboard user interface console templates.
    """
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Asynchronously processes user input strings, looks up matching text frames 
    from Pinecone, and returns the generated Gemini clinical assistant answers.
    """
    user_message = request.form.get("msg")
    
    # Validation safeguard for empty context values
    if not user_message or user_message.strip() == "":
        return jsonify({"response": "Empty query received. Please ask a valid medical question."}), 400
        
    try:
        # Route query straight into your helper pipeline object structures
        chatbot_response = bot_pipeline.generate_response(user_message)
        return jsonify({"response": chatbot_response})
        
    except Exception as e:
        print(f"[RUNTIME ERROR] Exception caught during thread execution: {str(e)}")
        return jsonify({"response": f"An internal server error occurred while processing: {str(e)}"}), 500


# ==========================================================================
# Application Launcher Engine Gateways
# ==========================================================================
if __name__ == "__main__":
    # Host configurations mapped to standard local system development addresses
    app.run(
        host="0.0.0.0", 
        port=8080, 
        debug=True
    )