import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from text import call_gemini
import PyPDF2
import os
import json
import re
import random
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta

# Set the event loop policy for Windows (if applicable)
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this to a secure random key in production
app.permanent_session_lifetime = timedelta(days=1)  # Session lasts 1 day

# Path to JSON database
USERS_DB = "users.json"

# Initialize users.json if it doesn't exist
if not os.path.exists(USERS_DB):
    with open(USERS_DB, "w") as f:
        json.dump({}, f)

# Store resume content and interview state
resume_content = ""
interview_state = {
    "stage": "initial",
    "skills": [],
    "questions_per_skill": {},
    "total_questions_asked": 0,
    "responses": [],
    "video_metrics": {
        "eye_contact": 0,
        "sentiment": "neutral",
        "facial_expression": "neutral",
        "speech_clarity": "moderate",
        "confidence_level": "moderate"
    }
}

def load_users():
    """Load users from JSON file."""
    with open(USERS_DB, "r") as f:
        return json.load(f)

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=4)

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

def extract_json_from_response(response):
    """Attempt to extract JSON from AI response with multiple fallback methods."""
    try:
        # First try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            # Try to find JSON within markdown code blocks
            json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find plain JSON within the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If all else fails, try to clean the response and parse
            cleaned = response.replace("'", '"').replace("None", "null")
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from response: {e}\nResponse was: {response}")

def analyze_resume(document_content):
    global resume_content, interview_state
    resume_content = document_content
    interview_state["stage"] = "analysis"
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Job Interview Simulator. Analyze the resume and return ONLY a JSON response with the following structure:\n"
                    "{\n"
                    "  \"acknowledgment\": \"<acknowledgment message>\",\n"
                    "  \"key_skills\": [\"skill1\", \"skill2\", \"skill3\"],\n"
                    "  \"prompt\": \"<message prompting user to start interview>\"\n"
                    "}\n"
                    "Do not include any additional text, explanations, or markdown formatting outside the JSON structure. "
                    "The JSON must be valid and parsable."
                )
            },
            {"role": "user", "content": "Analyze this resume:\n\n" + document_content}  # Use concatenation instead of f-string
        ]
        
        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        # Defensive checks: handle empty or auth-related responses from the AI provider
        if not response or not isinstance(response, str) or response.strip() == "":
            # AI returned nothing; fall back to local analysis
            data = local_resume_analysis(document_content)
        elif re.search(r"login|sign in|unauthori|unauth|please authenticate|subscription", response, re.I):
            # The AI provider likely requires authentication or the session is blocked.
            # Fall back to a local heuristic analysis and inform the user.
            data = local_resume_analysis(document_content)
            data.setdefault("acknowledgment", "(Partial) Analysis performed locally because the AI service was unavailable.")
        else:
            # Parse the response with improved JSON extraction
            try:
                data = extract_json_from_response(response)
            except ValueError:
                # Could not parse JSON from the AI response; fall back to local analysis
                data = local_resume_analysis(document_content)
        
        skills = data.get("key_skills", [])
        interview_state["skills"] = skills[:5] if skills else []
        interview_state["questions_per_skill"] = {skill: 0 for skill in interview_state["skills"]}
        
        formatted_skills = "\n".join([f"- {skill}" for skill in interview_state["skills"]])
        final_response = (
            f"{data.get('acknowledgment', 'Thank you for uploading your resume.')}\n\n"
            f"**Key Skills**:\n{formatted_skills}\n\n" +
            data.get('prompt', 'Please type "start" to begin the interview.')
        )
        return final_response.strip()
    except Exception as e:
        return f"Error processing resume: {str(e)}"

def generate_interview_question():
    global interview_state
    if not interview_state["skills"]:
        return "No skills were identified from your resume. Please upload a more detailed resume to continue the interview."

    total_questions_possible = len(interview_state["skills"]) * 2  # 2 questions per skill
    if interview_state["total_questions_asked"] >= total_questions_possible:
        return generate_feedback()

    for skill in interview_state["skills"]:
        if interview_state["questions_per_skill"][skill] < 2:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI Job Interview Simulator. Generate one concise, professional "
                            f"question about {skill} that would be relevant in a job interview. "
                            "The question should be directly related to the skill and appropriate for "
                            "the candidate's experience level. Return ONLY the question with no additional text."
                        )
                    }
                ]
                response = call_gemini(messages, temperature=0.7, top_p=0.9)
                # Defensive: if the AI returns an auth/login prompt or empty text, provide a local fallback question
                if not response or re.search(r"login|sign in|unauthori|please authenticate|subscription", str(response), re.I):
                    fallback = f"Tell me about your experience with {skill}."
                    interview_state["questions_per_skill"][skill] += 1
                    interview_state["total_questions_asked"] += 1
                    return fallback

                interview_state["questions_per_skill"][skill] += 1
                interview_state["total_questions_asked"] += 1
                
                # Update video metrics randomly to simulate analysis
                interview_state["video_metrics"] = {
                    "eye_contact": random.randint(30, 90),
                    "sentiment": random.choice(["positive", "neutral", "negative"]),
                    "facial_expression": random.choice(["neutral", "smiling", "confused", "engaged"]),
                    "speech_clarity": random.choice(["clear", "moderate", "muffled"]),
                    "confidence_level": random.choice(["low", "moderate", "high"])
                }
                
                return response.strip() if response else f"Tell me about your experience with {skill}."
            except Exception as e:
                return f"Error generating question: {e}"
    return "Unexpected error in question generation."

def generate_feedback():
    global interview_state
    try:
        responses_joined = "\n".join(interview_state["responses"])  # Compute outside f-string
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Job Interview Simulator. Provide constructive feedback on the interview responses.\n"
                    "Structure your feedback as follows:\n"
                    "1. Start with a positive acknowledgment\n"
                    "2. Highlight 2-3 strengths\n"
                    "3. Mention 2-3 areas for improvement\n"
                    "4. End with encouragement\n"
                    "Keep it professional, concise, and actionable. Base your feedback on these responses:\n\n"
                    f"{responses_joined}"
                )
            }
        ]
        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        # If AI returns a login/auth message or empty, produce a basic local feedback summary
        if not response or re.search(r"login|sign in|unauthori|please authenticate|subscription", str(response), re.I):
            interview_state["stage"] = "completed"
            # Create a simple local feedback based on number of responses and video metrics
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
                "Thank you for completing the interview.\n\n",
                "Strengths:\n- ", "\n- ".join(strengths) if strengths else "No clear strengths identified.",
                "\n\nAreas for improvement:\n- ", "\n- ".join(improvements) if improvements else "No specific improvements identified.",
                "\n\nKeep practicing and consider using a stable AI service session for richer feedback."
            ])
            return feedback

        interview_state["stage"] = "completed"
        return response.strip() if response else "Feedback could not be generated."
    except Exception as e:
        return f"Error generating feedback: {e}"


def local_resume_analysis(document_content):
    """A simple heuristic resume analyzer used as a fallback when the AI service is unavailable.

    Returns a dict with keys: acknowledgment, key_skills, prompt
    """
    # Try to find a Skills section
    skills = []
    # Look for a 'Skills' header
    match = re.search(r"(?mi)^\s*(skills|technical skills)[:\-]?\s*\n([\s\S]{0,400}?)\n\s*\n", document_content)
    if match:
        block = match.group(2)
        # Split on bullets, commas or newlines
        items = re.split(r"[\n•\-\*]+", block)
        for it in items:
            parts = re.split(r"[,;/]+", it)
            for p in parts:
                tok = p.strip()
                if tok:
                    skills.append(tok)
    else:
        # Fallback: find common tech keywords
        common = ["Python", "Java", "C++", "JavaScript", "React", "Node", "SQL", "Django", "Flask", "AWS", "Docker", "Kubernetes"]
        for c in common:
            if re.search(r"\b" + re.escape(c) + r"\b", document_content, re.I):
                skills.append(c)

    # Deduplicate and limit
    seen = []
    for s in skills:
        if s not in seen:
            seen.append(s)
    key_skills = seen[:5]

    return {
        "acknowledgment": "Resume analyzed locally (AI service unavailable).",
        "key_skills": key_skills,
        "prompt": "Please type \"start\" to begin the interview."
    }

def generate_tips():
    """Generate interview tips based on current performance metrics."""
    tips = []
    metrics = interview_state["video_metrics"]
    
    # Eye contact tips
    if metrics["eye_contact"] < 50:
        tips.append("Try to maintain eye contact with the camera for better engagement.")
    elif metrics["eye_contact"] > 70:
        tips.append("Great job maintaining eye contact! Keep it up.")
    
    # Sentiment tips
    if metrics["sentiment"] == "negative":
        tips.append("Try to maintain a more positive tone in your responses.")
    
    # Facial expression tips
    if metrics["facial_expression"] == "neutral":
        tips.append("Consider smiling more naturally to appear approachable.")
    elif metrics["facial_expression"] == "confused":
        tips.append("Try to relax your facial expressions to appear more confident.")
    
    # Speech clarity tips
    if metrics["speech_clarity"] == "muffled":
        tips.append("Speak a bit more clearly and at a moderate pace.")
    
    # Confidence tips
    if metrics["confidence_level"] == "low":
        tips.append("Practice power poses before interviews to boost confidence.")
    
    # Add some general tips if we don't have enough
    general_tips = [
        "Structure your answers using the STAR method (Situation, Task, Action, Result).",
        "Pause briefly before answering to collect your thoughts.",
        "Prepare stories from your experience that highlight your skills.",
        "Avoid filler words like 'um' and 'ah' for more polished responses."
    ]
    
    while len(tips) < 2 and general_tips:
        tips.append(general_tips.pop(random.randint(0, len(general_tips)-1)))
    
    return tips

def handle_user_response(user_input):
    global interview_state
    if interview_state["stage"] == "initial":
        return {
            "response": "Please upload your resume to begin the interview simulation!",
            "metrics": None,
            "tips": None
        }
    elif interview_state["stage"] == "analysis":
        if user_input.lower().strip() in ["start", "begin", "yes"]:
            interview_state["stage"] = "interview"
            question = generate_interview_question()
            return {
                "response": question,
                "metrics": interview_state["video_metrics"],
                "tips": generate_tips()
            }
        return {
            "response": "Please confirm to start the interview (e.g., 'start' or 'yes').",
            "metrics": None,
            "tips": None
        }
    elif interview_state["stage"] == "interview":
        interview_state["responses"].append(user_input)
        question = generate_interview_question()
        return {
            "response": question,
            "metrics": interview_state["video_metrics"],
            "tips": generate_tips()
        }
    elif interview_state["stage"] == "completed":
        return {
            "response": "The interview is complete! You can upload a new resume to start again.",
            "metrics": None,
            "tips": None
        }
    return {
        "response": "Something went wrong. Please try again.",
        "metrics": None,
        "tips": None
    }

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/check-auth")
def check_auth():
    if 'username' in session:
        return jsonify({
            'authenticated': True,
            'username': session['username']
        })
    return jsonify({
        'authenticated': False
    })

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            flash("Username and password are required!", "error")
            return redirect(url_for("register"))
        
        users = load_users()
        if username in users:
            flash("Username already exists!", "error")
            return redirect(url_for("register"))
        
        users[username] = {"password": generate_password_hash(password)}
        save_users(users)
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        users = load_users()
        
        if username in users and check_password_hash(users[username]["password"], password):
            session.permanent = True
            session["username"] = username
            flash("Logged in successfully!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password!", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Logged out successfully!", "success")
    return redirect(url_for("landing"))

@app.route("/interview")
def index():
    if "username" not in session:
        flash("Please log in to access the interview simulator.", "error")
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "username" not in session:
        return jsonify({
            "response": "Please log in to continue.",
            "metrics": None,
            "tips": None
        }), 401
    
    data = request.get_json()
    user_input = data.get("message", "")
    result = handle_user_response(user_input)
    return jsonify(result)

@app.route("/upload", methods=["POST"])
def upload():
    if "username" not in session:
        return jsonify({
            "response": "Please log in to upload a resume.",
            "metrics": None,
            "tips": None
        }), 401
    
    global interview_state
    if "file" not in request.files:
        return jsonify({
            "response": "No file uploaded.",
            "metrics": None,
            "tips": None
        }), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "response": "No file selected.",
            "metrics": None,
            "tips": None
        }), 400

    if file.filename.endswith(".pdf"):
        document_content = extract_text_from_pdf(file)
    elif file.filename.endswith(".txt"):
        document_content = file.read().decode("utf-8")
    else:
        return jsonify({
            "response": "Unsupported file format. Please upload a PDF or text file.",
            "metrics": None,
            "tips": None
        }), 400

    if "Error" in document_content:
        return jsonify({
            "response": document_content,
            "metrics": None,
            "tips": None
        }), 500

    # Reset interview state
    interview_state = {
        "stage": "initial",
        "skills": [],
        "questions_per_skill": {},
        "total_questions_asked": 0,
        "responses": [],
        "video_metrics": {
            "eye_contact": 0,
            "sentiment": "neutral",
            "facial_expression": "neutral",
            "speech_clarity": "moderate",
            "confidence_level": "moderate"
        }
    }
    
    response = analyze_resume(document_content)
    return jsonify({
        "response": response,
        "metrics": None,
        "tips": None
    })

if __name__ == "__main__":
    app.run(debug=True)