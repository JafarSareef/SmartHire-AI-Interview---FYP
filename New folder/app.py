import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from flask import Flask, render_template, request, jsonify
# g4f removed; using Gemini exclusively
import PyPDF2
import os
import json
import re
import requests
from typing import List, Dict, Optional
from text import call_gemini

# Set the event loop policy for Windows (if applicable)
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

app = Flask(__name__)

# Store resume content and interview state
resume_content = ""
interview_state = {
    "stage": "initial",
    "skills": [],
    "questions_per_skill": {},  # Tracks questions asked per skill
    "total_questions_asked": 0,  # Total questions asked across all skills
    "responses": []
}

# Gemini config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAKTUnsZzrIVDR7G0VHRlR4863hNOCs600")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash-lite")


def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

def analyze_resume(document_content):
    global resume_content, interview_state
    resume_content = document_content
    interview_state["stage"] = "analysis"

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Job Interview Simulator. Analyze the resume and return your answer strictly in JSON format. "
                    "Your output must be a valid JSON object with the following keys: "
                    "`acknowledgment` (a string message acknowledging the resume upload), "
                    "`key_skills` (an array of the top 5 skills), and "
                    "`prompt` (a string message to prompt the interview). "
                    "Do not include any extra text or markdown formatting."
                )
            },
            {"role": "user", "content": f"Analyze this resume:\n\n{document_content}"}
        ]
        # Use Gemini exclusively
        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        if not response or response.startswith("ERROR:"):
            return f"AI service unavailable: {response}"
        
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            return "Error: The AI response did not contain valid JSON. Please try again or adjust your prompt."
        
        json_str = json_match.group()
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as json_err:
            return f"Error parsing JSON: {json_err}"
        
        skills = data.get("key_skills", [])
        interview_state["skills"] = skills[:5] if skills else []
        # Initialize questions_per_skill for each skill
        interview_state["questions_per_skill"] = {skill: 0 for skill in interview_state["skills"]}
        
        formatted_skills = "\n".join([f"- {skill}" for skill in interview_state["skills"]])
        final_response = (
            f"{data.get('acknowledgment', 'Thank you for uploading your resume.')}\n\n"
            f"**Key Skills**:\n{formatted_skills}\n\n"
            f"{data.get('prompt', 'Please type \"start\" to begin the interview.')}"
        )
        return final_response.strip() if final_response else "Sorry, I couldn't process the resume."
    except Exception as e:
        return f"Error: {e}"

def generate_interview_question():
    """Generate an interview question based on extracted skills."""
    global interview_state
    if not interview_state["skills"]:
        return "No skills were identified from your resume. Please upload a more detailed resume to continue the interview."

    # Check if all skills have been asked 5 questions
    total_questions_possible = len(interview_state["skills"]) * 1
    if interview_state["total_questions_asked"] >= total_questions_possible:
        return generate_feedback()

    # Find the next skill to ask a question about
    for skill in interview_state["skills"]:
        if interview_state["questions_per_skill"][skill] < 1:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI Job Interview Simulator conducting a real-time interview. "
                            "Generate one thoughtful, job-relevant question based on the skill provided. "
                            "Keep it concise, professional, and conversational."
                        )
                    },
                    {"role": "user", "content": f"Generate a question for the skill: {skill}"}
                ]
                # Use Gemini exclusively
                response = call_gemini(messages, temperature=0.7, top_p=0.9)
                if not response or response.startswith("ERROR:"):
                    return f"AI service unavailable: {response}"
                interview_state["questions_per_skill"][skill] += 1
                interview_state["total_questions_asked"] += 1
                return response.strip() if response else f"Tell me about your experience with {skill}."
            except Exception as e:
                return f"Error generating question: {e}"
    return "Unexpected error in question generation."

def generate_feedback():
    """Generate feedback based on user responses."""
    global interview_state
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Job Interview Simulator. Based on the user's responses to interview questions, "
                    "provide feedback in a structured format: "
                    "- Start with a positive acknowledgment of their participation. "
                    "- **Strengths**: Highlight what they did well (e.g., clarity, confidence). "
                    "- **Weaknesses**: Note areas for improvement (e.g., detail, specificity). "
                    "- **Areas for Improvement**: Suggest specific ways to enhance their responses. "
                    "Use the responses provided and keep the tone constructive and encouraging."
                )
            },
            {"role": "user", "content": f"Responses:\n\n{chr(10).join(interview_state['responses'])}"}
        ]
        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        if not response or response.startswith("ERROR:"):
            interview_state["stage"] = "completed"
            strengths = []
            improvements = []
            if interview_state["responses"]:
                strengths.append("You answered multiple questions; that shows practice and readiness.")
            vm = interview_state.get("video_metrics", {})
            if vm.get("eye_contact", 0) > 60:
                strengths.append("Good eye contact.")
            else:
                improvements.append("Try to maintain steady eye contact with the camera.")
            if vm.get("confidence_level") == "high":
                strengths.append("Confidence is high; that's great.")
            else:
                improvements.append("Work on projecting confidence in your tone.")

            feedback = "".join([
                "(Local fallback) Thank you for completing the interview.\n\n",
                "Strengths:\n- ", "\n- ".join(strengths) if strengths else "No clear strengths identified.",
                "\n\nAreas for improvement:\n- ", "\n- ".join(improvements) if improvements else "No specific improvements identified.",
                "\n\nNote: AI feedback unavailable (Gemini did not respond)."
            ])
            return feedback
        interview_state["stage"] = "completed"
        return response.strip() if response else "Feedback could not be generated."
    except Exception as e:
        return f"Error: {e}"

def handle_user_response(user_input):
    """Handle user input during the interview."""
    global interview_state
    if interview_state["stage"] == "initial":
        return "Please upload your resume to begin the interview simulation!"
    elif interview_state["stage"] == "analysis":
        if user_input.lower().strip() in ["start", "begin", "yes"]:
            interview_state["stage"] = "interview"
            return generate_interview_question()
        return "Please confirm to start the interview (e.g., 'start' or 'yes')."
    elif interview_state["stage"] == "interview":
        interview_state["responses"].append(user_input)
        return generate_interview_question()
    elif interview_state["stage"] == "completed":
        return "The interview is complete! You can upload a new resume to start again."
    return "Something went wrong. Please try again."

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/interview")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = handle_user_response(user_input)
    return jsonify({"response": response})

@app.route("/upload", methods=["POST"])
def upload():
    global interview_state
    if "file" not in request.files:
        return jsonify({"response": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"response": "No file selected."}), 400

    if file.filename.endswith(".pdf"):
        document_content = extract_text_from_pdf(file)
    elif file.filename.endswith(".txt"):
        document_content = file.read().decode("utf-8")
    else:
        return jsonify({"response": "Unsupported file format. Please upload a PDF or text file."}), 400

    if "Error" in document_content:
        return jsonify({"response": document_content}), 500

    # Reset interview state
    interview_state = {
        "stage": "initial",
        "skills": [],
        "questions_per_skill": {},
        "total_questions_asked": 0,
        "responses": []
    }
    response = analyze_resume(document_content)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)