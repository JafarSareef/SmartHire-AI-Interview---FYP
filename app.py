import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
from text import call_gemini
import PyPDF2
import os
import json
import re
import random
import base64
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from io import BytesIO
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Set the event loop policy for Windows (if applicable)
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this to a secure random key in production
app.permanent_session_lifetime = timedelta(days=1)  # Session lasts 1 day

# Path to JSON database
USERS_DB = "users.json"
FACE_DATA_DIR = "face_data"

# Create face_data directory if it doesn't exist
if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

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
    "interview_questions_count": 0,  # Track actual interview questions (excluding general questions like "start")
    "responses": [],
    "video_metrics": {
        "eye_contact": 0,
        "sentiment": "neutral",
        "facial_expression": "neutral",
        "speech_clarity": "moderate",
        "confidence_level": "moderate",
        "face_count": 1,        # NEW: how many faces are detected (default 1)
        "multiple_faces": False # NEW: flag for multi-person warning
    },
    "current_question": None,  # Store the current unanswered question
    "current_question_skill": None,
    "questions_history": [],  # Store all questions asked to prevent duplicates
    "cached_feedback": None  # Cache final feedback for report generation
}


def load_users():
    """Load users from JSON file."""
    with open(USERS_DB, "r") as f:
        return json.load(f)


def save_users(users):
    """Save users to JSON file."""
    with open(USERS_DB, "w") as f:
        json.dump(users, f, indent=4)


def save_face_image(email, image_data):
    """Save face image(s) for user. Handles both single image and multiple frames."""
    try:
        print(f"\n[Face Registration] Starting face registration for: {email}")
        
        # Check if image_data is a JSON array (multiple frames)
        try:
            frames = json.loads(image_data)
            if isinstance(frames, list) and len(frames) > 0:
                print(f"[Face Registration] Received {len(frames)} frames for registration")
                
                # Save all frames to separate files for verification
                base_filename = f"{email.replace('@', '_').replace('.', '_')}"
                
                # Create directory for user's face frames if not exists
                user_face_dir = os.path.join(FACE_DATA_DIR, base_filename)
                if not os.path.exists(user_face_dir):
                    os.makedirs(user_face_dir)
                
                # Save each frame
                for idx, frame_data in enumerate(frames):
                    frame_data = frame_data.split(',')[1] if ',' in frame_data else frame_data
                    image_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    frame_filepath = os.path.join(user_face_dir, f"frame_{idx}.jpg")
                    cv2.imwrite(frame_filepath, img)
                    print(f"[Face Registration] Saved frame {idx+1}/{len(frames)}")
                
                # Also save the middle frame as the main reference image
                middle_idx = len(frames) // 2
                middle_frame_raw = frames[middle_idx]
                middle_frame = middle_frame_raw.split(',')[1] if ',' in middle_frame_raw else middle_frame_raw
                image_bytes = base64.b64decode(middle_frame)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Detect faces in the middle frame to ensure single-person registration
                try:
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if len(faces) == 0:
                        return {"success": False, "message": "No face detected in the registration image. Please provide a clear, front-facing photo.", "multiple_faces": False}
                    if len(faces) > 1:
                        return {"success": False, "message": "Multiple faces detected in the registration image. Please ensure only your face is visible.", "multiple_faces": True}
                except Exception as e:
                    print(f"[Face Registration] Warning: face detector not available: {e}")

                main_filepath = os.path.join(FACE_DATA_DIR, f"{base_filename}.jpg")
                cv2.imwrite(main_filepath, img)

                print(f"[Face Registration] ✓ Successfully registered with {len(frames)} frames")
                return {"success": True, "message": "Face registered successfully.", "multiple_faces": False}
                
        except (json.JSONDecodeError, ValueError):
            # Single image (legacy support)
            print(f"[Face Registration] Single frame registration")
            pass
        
        # Handle single image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces to ensure single-person registration
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            if len(faces) == 0:
                return {"success": False, "message": "No face detected in the registration image. Please provide a clear, front-facing photo.", "multiple_faces": False}
            if len(faces) > 1:
                return {"success": False, "message": "Multiple faces detected in the registration image. Please ensure only your face is visible.", "multiple_faces": True}
        except Exception as e:
            print(f"[Face Registration] Warning: face detector not available: {e}")

        filename = f"{email.replace('@', '_').replace('.', '_')}.jpg"
        filepath = os.path.join(FACE_DATA_DIR, filename)
        cv2.imwrite(filepath, img)

        print(f"[Face Registration] ✓ Successfully registered single frame")
        return {"success": True, "message": "Face registered successfully.", "multiple_faces": False}
        
    except Exception as e:
        print(f"[Face Registration] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_face(email, image_data):
    """Verify face against stored image(s) using advanced face recognition with multi-frame support."""
    try:
        base_filename = f"{email.replace('@', '_').replace('.', '_')}"
        filepath = os.path.join(FACE_DATA_DIR, f"{base_filename}.jpg")
        user_face_dir = os.path.join(FACE_DATA_DIR, base_filename)
        
        print(f"\n{'='*60}")
        print(f"[Face Verification] Starting verification for: {email}")
        print(f"[Face Verification] Looking for stored face at: {filepath}")
        
        # Check if multi-frame directory exists
        use_multi_frame = os.path.exists(user_face_dir) and os.path.isdir(user_face_dir)
        
        if use_multi_frame:
            stored_frames = sorted([f for f in os.listdir(user_face_dir) if f.endswith('.jpg')])
            print(f"[Face Verification] Found {len(stored_frames)} registered frames for comparison")
        elif not os.path.exists(filepath):
            print(f"[Face Verification] ERROR: No face file found")
            return {"success": False, "message": "No face registered for this email"}
        else:
            print(f"[Face Verification] Using single frame verification")
        
        # Load main stored image for comparison
        stored_img = cv2.imread(filepath)
        if stored_img is None:
            print(f"[Face Verification] ERROR: Could not read stored image")
            return {"success": False, "message": "Error reading stored face image"}
        
        print(f"[Face Verification] Main stored image loaded: {stored_img.shape}")
        
        # Decode captured image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        captured_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if captured_img is None:
            print(f"[Face Verification] ERROR: Could not decode captured image")
            return {"success": False, "message": "Error decoding captured image"}
        
        print(f"[Face Verification] Captured image decoded: {captured_img.shape}")
        
        # Initialize face detector (Haar Cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to initialize LBPH Face Recognizer for better accuracy (requires opencv-contrib)
        use_lbph = False
        recognizer = None
        try:
            if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                use_lbph = True
            else:
                # Some OpenCV builds may not expose the contrib module
                use_lbph = False
        except Exception as e:
            print(f"[Face Verification] OpenCV 'face' module not available: {e}")
            use_lbph = False

        # Detect faces in both images
        stored_gray = cv2.cvtColor(stored_img, cv2.COLOR_BGR2GRAY)
        captured_gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better recognition
        stored_gray = cv2.equalizeHist(stored_gray)
        captured_gray = cv2.equalizeHist(captured_gray)
        
        stored_faces = face_cascade.detectMultiScale(stored_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        captured_faces = face_cascade.detectMultiScale(captured_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        print(f"[Face Verification] Faces detected - Stored: {len(stored_faces)}, Captured: {len(captured_faces)}")
        
        if len(stored_faces) == 0:
            return {"success": False, "message": "No face detected in registered image. Please re-register with a clear, front-facing photo."}
        
        # If multiple faces are detected in the captured image, return a clear structured error
        if len(captured_faces) > 1:
            print(f"[Face Verification] Multiple faces detected in captured image ({len(captured_faces)}) - aborting verification")
            return {"success": False, "message": "Multiple faces detected in the camera frame. Please ensure only your face is visible.", "multiple_faces": True}

        if len(captured_faces) == 0:
            return {"success": False, "message": "No face detected. Please ensure: 1) Face is clearly visible, 2) Good lighting, 3) Look directly at camera, 4) Remove glasses/mask if possible."}
        
        # Use the largest face if multiple detected
        if len(stored_faces) > 1:
            stored_faces = sorted(stored_faces, key=lambda x: x[2] * x[3], reverse=True)
        if len(captured_faces) > 1:
            captured_faces = sorted(captured_faces, key=lambda x: x[2] * x[3], reverse=True)
        
        (x1, y1, w1, h1) = stored_faces[0]
        (x2, y2, w2, h2) = captured_faces[0]
        
        # Extract and normalize face regions
        stored_face = stored_gray[y1:y1+h1, x1:x1+w1]
        captured_face = captured_gray[y2:y2+h2, x2:x2+w2]
        
        # Resize to consistent size for comparison
        size = (200, 200)
        stored_face = cv2.resize(stored_face, size)
        captured_face = cv2.resize(captured_face, size)
        
        print(f"[Face Verification] Face regions extracted and normalized to {size}")
        
        # Method 1: LBPH Face Recognition (most accurate for face verification)
        # If multi-frame available, train on ALL frames for better accuracy
        training_faces = [stored_face]
        training_labels = [0]
        
        if use_multi_frame:
            print(f"[Face Verification] Training LBPH on multiple frames for better accuracy...")
            for frame_file in stored_frames[:5]:  # Use all 5 frames
                frame_path = os.path.join(user_face_dir, frame_file)
                frame_img = cv2.imread(frame_path)
                if frame_img is None:
                    continue
                
                frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.equalizeHist(frame_gray)
                frame_faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                
                if len(frame_faces) > 0:
                    if len(frame_faces) > 1:
                        frame_faces = sorted(frame_faces, key=lambda x: x[2] * x[3], reverse=True)
                    (fx, fy, fw, fh) = frame_faces[0]
                    frame_face = frame_gray[fy:fy+fh, fx:fx+fw]
                    frame_face = cv2.resize(frame_face, size)
                    training_faces.append(frame_face)
                    training_labels.append(0)  # Same person, same label
            
            print(f"[Face Verification] Training set: {len(training_faces)} faces")
        
        # Compute LBPH similarity if available, otherwise fallback to pixel-based MSE similarity
        lbph_similarity = 0.0
        confidence = None
        if use_lbph and recognizer is not None:
            try:
                # Train on all available faces
                recognizer.train(training_faces, np.array(training_labels))

                # Predict captured face
                label, confidence = recognizer.predict(captured_face)

                # LBPH confidence: lower is better (0 = perfect match, higher = different)
                # With multi-frame training, we get much better confidence scores
                # Adjusted scale: 0-50 is excellent, 50-80 is good, >80 is poor
                max_confidence = 80.0 if use_multi_frame else 100.0
                lbph_similarity = max(0, 1.0 - (confidence / max_confidence))

                print(f"[Face Verification] LBPH Confidence: {confidence:.2f} (lower is better)")
                print(f"[Face Verification] LBPH Similarity: {lbph_similarity:.4f} ({lbph_similarity*100:.2f}%)")
            except Exception as e:
                print(f"[Face Verification] LBPH recognizer failed during training/predict: {e}")
                use_lbph = False

        if not use_lbph:
            # Fallback approach: compute similarity based on normalized MSE between faces.
            # If multiple stored frames exist, compare against each and take the best match.
            try:
                def mse_similarity(imgA, imgB):
                    err = np.mean((imgA.astype('float') - imgB.astype('float')) ** 2)
                    # Normalize by max possible MSE (255^2) to get a 0..1 similarity
                    norm = err / (255.0 ** 2)
                    sim = max(0.0, 1.0 - norm)
                    return sim

                best_sim = mse_similarity(stored_face, captured_face)
                if use_multi_frame:
                    for frame_file in stored_frames[:5]:
                        frame_path = os.path.join(user_face_dir, frame_file)
                        frame_img = cv2.imread(frame_path)
                        if frame_img is None:
                            continue
                        frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
                        frame_gray = cv2.equalizeHist(frame_gray)
                        frame_faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                        if len(frame_faces) > 0:
                            if len(frame_faces) > 1:
                                frame_faces = sorted(frame_faces, key=lambda x: x[2] * x[3], reverse=True)
                            (fx, fy, fw, fh) = frame_faces[0]
                            frame_face = frame_gray[fy:fy+fh, fx:fx+fw]
                            frame_face = cv2.resize(frame_face, size)
                            sim = mse_similarity(frame_face, captured_face)
                            if sim > best_sim:
                                best_sim = sim

                lbph_similarity = float(best_sim)
                # Map to a confidence-like number for logging (lower is better)
                max_confidence = 80.0 if use_multi_frame else 100.0
                confidence = (1.0 - lbph_similarity) * max_confidence
                print(f"[Face Verification] Fallback MSE Similarity used as LBPH substitute")
                print(f"[Face Verification] Fallback Confidence (approx): {confidence:.2f}")
                print(f"[Face Verification] Fallback Similarity: {lbph_similarity:.4f} ({lbph_similarity*100:.2f}%)")
            except Exception as e:
                print(f"[Face Verification] ERROR in fallback similarity computation: {e}")
                lbph_similarity = 0.0
                confidence = max_confidence if 'max_confidence' in locals() else 100.0
        
        # Method 2: ORB Feature Matching for additional validation
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(stored_face, None)
        kp2, des2 = orb.detectAndCompute(captured_face, None)
        
        orb_similarity = 0.0
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Get good matches (distance < threshold)
            good_matches = [m for m in matches if m.distance < 50]
            orb_similarity = len(good_matches) / max(len(kp1), len(kp2), 1)
            
            print(f"[Face Verification] ORB Keypoints - Stored: {len(kp1)}, Captured: {len(kp2)}")
            print(f"[Face Verification] ORB Good matches: {len(good_matches)}/{len(matches)}")
            print(f"[Face Verification] ORB Similarity: {orb_similarity:.4f} ({orb_similarity*100:.2f}%)")
        
        # Method 3: Histogram Comparison
        stored_hist = cv2.calcHist([stored_face], [0], None, [256], [0, 256])
        captured_hist = cv2.calcHist([captured_face], [0], None, [256], [0, 256])
        
        stored_hist = cv2.normalize(stored_hist, stored_hist).flatten()
        captured_hist = cv2.normalize(captured_hist, captured_hist).flatten()
        
        hist_similarity = cv2.compareHist(stored_hist, captured_hist, cv2.HISTCMP_CORREL)
        
        print(f"[Face Verification] Histogram Similarity: {hist_similarity:.4f} ({hist_similarity*100:.2f}%)")
    
        # Method 4: Structural Similarity using Template Matching
        result = cv2.matchTemplate(captured_face, stored_face, cv2.TM_CCOEFF_NORMED)
        template_similarity = np.max(result)
        
        print(f"[Face Verification] Template Matching: {template_similarity:.4f} ({template_similarity*100:.2f}%)")
        
        # Combined weighted score (emphasizing LBPH as it's most reliable for faces)
        combined_similarity = (
            lbph_similarity * 0.50 +      # 50% weight - most important
            orb_similarity * 0.20 +       # 20% weight - feature matching
            hist_similarity * 0.15 +      # 15% weight - histogram
            template_similarity * 0.15    # 15% weight - structural
        )
        
        # Multi-frame training already improved LBPH score significantly
        # No need for additional frame-by-frame comparison
        if use_multi_frame:
            print(f"[Face Verification] ✓ Multi-frame trained model provides enhanced accuracy")
        
        print(f"\n[Face Verification] COMBINED SIMILARITY: {combined_similarity:.4f} ({combined_similarity*100:.2f}%)")
        
        # Dynamic threshold based on training quality
        # Multi-frame trained models are more reliable, so we can use a more reasonable threshold
        if use_multi_frame:
            threshold = 0.40  # 40% with multi-frame training (more accurate baseline)
        else:
            threshold = 0.45  # 45% with single-frame (less reliable)
        
        print(f"[Face Verification] Required threshold: {threshold*100:.0f}% ({'multi-frame trained' if use_multi_frame else 'single-frame'})")
        print(f"[Face Verification] Result: {'✓ PASS - FACE VERIFIED' if combined_similarity >= threshold else '✗ FAIL - FACE NOT VERIFIED'}")
        print(f"{'='*60}\n")
        
        if combined_similarity >= threshold:
            return {
                "success": True,
                "message": "Face verified successfully",
                "similarity": float(combined_similarity),
                "details": {
                    "lbph": float(lbph_similarity),
                    "orb": float(orb_similarity),
                    "histogram": float(hist_similarity),
                    "template": float(template_similarity)
                }
            }
        else:
            return {
                "success": False,
                "message": f"Face verification failed - Match confidence too low ({combined_similarity*100:.1f}% < {threshold*100:.0f}% required). Please ensure good lighting and face the camera directly.",
                "similarity": float(combined_similarity),
                "details": {
                    "lbph": float(lbph_similarity),
                    "orb": float(orb_similarity),
                    "histogram": float(hist_similarity),
                    "template": float(template_similarity)
                }
            }
            
    except Exception as e:
        print(f"[Face Verification] EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Error during verification: {str(e)}"}


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
    # If it's already a dict-like object, assume it's parsed JSON
    if isinstance(response, dict):
        return response

    response_text = str(response or "")
    try:
        # First try to parse the entire response as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            # Try to find JSON within markdown code blocks
            json_match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find plain JSON within the response
            json_match = re.search(r'\{[\s\S]*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If all else fails, try to clean the response and parse
            cleaned = response_text.replace("'", '"').replace("None", "null")
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from response: {e}\nResponse was: {response_text}")


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
            {"role": "user", "content": "Analyze this resume:\n\n" + document_content}
        ]
        
        # Use centralized Gemini caller and return real-time response or clear errors (no local fallback)
        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        if response is None:
            return "AI service error: no response from Gemini. Check GEMINI_API_KEY and GEMINI_MODEL."

        # If call_gemini returned an error string starting with ERROR: or status info, surface it
        if isinstance(response, str) and response.startswith("ERROR:"):
            return f"AI service error: {response}"

        # Normalize response text for regex checks
        response_text = ""
        if isinstance(response, dict):
            # try typical keys
            response_text = response.get("text") or response.get("content") or json.dumps(response)
        else:
            response_text = str(response)

        if re.search(r"login|sign in|unauthori|unauth|please authenticate|subscription", response_text, re.I):
            return f"AI service authentication error: {response_text}"

        try:
            data = extract_json_from_response(response)
        except ValueError as e:
            return f"Could not parse JSON from AI response: {e}\nRaw response: {response_text}"
        
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

    # Ensure backward compatibility
    if "interview_questions_count" not in interview_state:
        interview_state["interview_questions_count"] = 0
    if "current_question" not in interview_state:
        interview_state["current_question"] = None
    if "questions_history" not in interview_state:
        interview_state["questions_history"] = []

    # If there's already a current question waiting for answer, return it (prevents duplicates)
    if interview_state["current_question"] is not None:
        print(f"[Interview] Returning existing unanswered question (preventing duplicate)")
        return interview_state["current_question"]

    # STRICT LIMIT: Only 5 interview questions allowed
    if interview_state["interview_questions_count"] >= 5:
        print(f"\n[Interview] Reached question limit: {interview_state['interview_questions_count']}/5 questions asked")
        print("[Interview] Generating final feedback...")
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
                if response is None:
                    return "AI service error: no response from Gemini. Check GEMINI_API_KEY and GEMINI_MODEL."
                if isinstance(response, str) and response.startswith("ERROR:"):
                    return f"AI service error: {response}"

                response_text = json.dumps(response) if isinstance(response, dict) else str(response)
                if re.search(r"login|sign in|unauthori|please authenticate|subscription", response_text, re.I):
                    return f"AI service authentication error: {response_text}"

                question = response.strip() if response else f"Tell me about your experience with {skill}."
                
                # Store the question as current (will be marked as answered when user responds)
                interview_state["current_question"] = question
                interview_state["current_question_skill"] = skill
                
                # Don't increment counters yet - wait for user response
                print(f"\n[Interview] Generated question {interview_state['interview_questions_count'] + 1}/5 (Skill: {skill})")
                print(f"[Interview] Waiting for user response before incrementing counter...")
                
                # Update video metrics randomly to simulate analysis
                # NOTE: For now we assume exactly 1 person in frame; when you have
                # real video analysis, update `face_count` and `multiple_faces` here.
                interview_state["video_metrics"] = {
                    "eye_contact": random.randint(30, 90),
                    "sentiment": random.choice(["positive", "neutral", "negative"]),
                    "facial_expression": random.choice(["neutral", "smiling", "confused", "engaged"]),
                    "speech_clarity": random.choice(["clear", "moderate", "muffled"]),
                    "confidence_level": random.choice(["low", "moderate", "high"]),
                    "face_count": 1,        # default: single candidate
                    "multiple_faces": False  # set True when your vision model detects > 1
                }
                
                return question
            except Exception as e:
                return f"Error generating question: {e}"
    
    # If all skills exhausted but haven't reached 5 questions yet, generate feedback
    return generate_feedback()


def generate_feedback():
    global interview_state
    try:
        # Check if feedback already generated and cached
        if interview_state.get("cached_feedback"):
            return interview_state["cached_feedback"]
        
        # Use Q&A history if available for better context
        if "questions_history" in interview_state and interview_state["questions_history"]:
            qa_text = "\n\n".join([
                f"Q{idx+1} ({qa['skill']}): {qa['question']}\nA{idx+1}: {qa['answer']}"
                for idx, qa in enumerate(interview_state["questions_history"])
            ])
            questions_answered = len(interview_state["questions_history"])
        else:
            # Fallback to old format
            qa_text = "\n".join(interview_state["responses"])
            questions_answered = interview_state.get("interview_questions_count", len(interview_state["responses"]))
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI Job Interview Simulator conducting the FINAL EVALUATION after a 5-question interview.\n\n"
                    "**IMPORTANT**: This is the interview RESULT and FEEDBACK, not another question.\n\n"
                    f"The candidate answered {questions_answered} interview questions. Provide a comprehensive evaluation:\n\n"
                    "**Structure your response as:**\n"
                    "# 🎯 Interview Complete - Final Evaluation\n\n"
                    "## Overall Performance\n"
                    "[Provide an overall assessment with a score or rating]\n\n"
                    "## ✅ Strengths\n"
                    "[List 2-3 specific strengths demonstrated]\n\n"
                    "## 📈 Areas for Improvement\n"
                    "[List 2-3 specific areas to work on with actionable advice]\n\n"
                    "## 💡 Recommendations\n"
                    "[Provide 2-3 concrete suggestions for improvement]\n\n"
                    "## Final Remarks\n"
                    "[End with encouragement and next steps]\n\n"
                    "**IMPORTANT**: Review each question-answer pair carefully. Do NOT claim answers are the same "
                    "unless they are truly identical. Focus on the content and quality of each unique response.\n\n"
                    "Interview Questions & Answers:\n\n"
                    f"{qa_text}"
                )
            }
        ]

        response = call_gemini(messages, temperature=0.7, top_p=0.9)
        if response is None:
            return "AI service error: no response from Gemini. Check GEMINI_API_KEY and GEMINI_MODEL."
        if isinstance(response, str) and response.startswith("ERROR:"):
            return f"AI service error: {response}"

        response_text = json.dumps(response) if isinstance(response, dict) else str(response)
        if re.search(r"login|sign in|unauthori|please authenticate|subscription", response_text, re.I):
            return f"AI service authentication error: {response_text}"

        interview_state["stage"] = "completed"
        
        # Cache the feedback
        feedback_result = response_text.strip()
        interview_state["cached_feedback"] = feedback_result
        
        print(f"\n[Interview] ✓ Feedback generated after {questions_answered} questions")
        return feedback_result
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

    # Use .get() so we don't crash if any field is missing
    eye_contact = metrics.get("eye_contact", 0)
    sentiment = metrics.get("sentiment", "neutral")
    facial_expression = metrics.get("facial_expression", "neutral")
    speech_clarity = metrics.get("speech_clarity", "moderate")
    confidence_level = metrics.get("confidence_level", "moderate")
    face_count = metrics.get("face_count", 1)
    
    # Eye contact tips
    if eye_contact < 50:
        tips.append("Try to maintain eye contact with the camera for better engagement.")
    elif eye_contact > 70:
        tips.append("Great job maintaining eye contact! Keep it up.")
    
    # Sentiment tips
    if sentiment == "negative":
        tips.append("Try to maintain a more positive tone in your responses.")
    
    # Facial expression tips
    if facial_expression == "neutral":
        tips.append("Consider smiling more naturally to appear approachable.")
    elif facial_expression == "confused":
        tips.append("Try to relax your facial expressions to appear more confident.")
    
    # Speech clarity tips
    if speech_clarity == "muffled":
        tips.append("Speak a bit more clearly and at a moderate pace.")
    
    # Confidence tips
    if confidence_level == "low":
        tips.append("Practice power poses before interviews to boost confidence.")

    # Multi-person tip
    if face_count > 1 or metrics.get("multiple_faces", False):
        tips.append("Only one person should be visible in the camera for an accurate interview evaluation.")
    
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


def generate_report_data():
    """Generate a comprehensive interview report with all metrics and feedback."""
    global interview_state, resume_content
    
    try:
        report = {
            "candidate_name": session.get("username", "Unknown"),
            "interview_date": timedelta(seconds=0).total_seconds(),  # Can be enhanced with actual timestamp
            "skills_tested": interview_state.get("skills", []),
            "questions_answered": interview_state.get("interview_questions_count", 0),
            "questions_per_skill": interview_state.get("questions_per_skill", {}),
            "questions_history": interview_state.get("questions_history", []),
            "video_metrics": interview_state.get("video_metrics", {}),
            "feedback": "",
            "stage": interview_state.get("stage", "initial")
        }
        
        return report
    except Exception as e:
        print(f"[Report] Error generating report data: {e}")
        return None


def create_pdf_report(report_data, feedback_text):
    """Create a PDF report from interview data and feedback."""
    try:
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.6*inch, bottomMargin=0.6*inch)
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10
        )
        
        # Title
        elements.append(Paragraph("AI Job Interview Simulator", title_style))
        elements.append(Paragraph("Interview Report & Recommendations", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Candidate Info
        elements.append(Paragraph(f"<b>Candidate:</b> {report_data['candidate_name']}", normal_style))
        elements.append(Paragraph(f"<b>Interview Stage:</b> {report_data['stage'].upper()}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Skills Tested
        elements.append(Paragraph("Skills Tested", heading_style))
        skills_text = ", ".join(report_data['skills_tested']) if report_data['skills_tested'] else "None"
        elements.append(Paragraph(f"<b>Key Skills:</b> {skills_text}", normal_style))
        elements.append(Paragraph(f"<b>Questions Answered:</b> {report_data['questions_answered']}/5", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Video Metrics
        if report_data['video_metrics']:
            elements.append(Paragraph("Performance Metrics", heading_style))
            metrics = report_data['video_metrics']
            
            # Create metrics table
            metrics_data = [
                ['Metric', 'Score/Value'],
                ['Eye Contact', f"{metrics.get('eye_contact', 'N/A')}%"],
                ['Sentiment', metrics.get('sentiment', 'N/A').capitalize()],
                ['Facial Expression', metrics.get('facial_expression', 'N/A').capitalize()],
                ['Speech Clarity', metrics.get('speech_clarity', 'N/A').capitalize()],
                ['Confidence Level', metrics.get('confidence_level', 'N/A').capitalize()],
            ]
            
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Questions & Answers
        if report_data['questions_history']:
            elements.append(PageBreak())
            elements.append(Paragraph("Questions & Answers Review", heading_style))
            elements.append(Spacer(1, 0.1*inch))
            
            for idx, qa in enumerate(report_data['questions_history'], 1):
                q_text = f"<b>Q{idx} ({qa.get('skill', 'General')}):</b> {qa['question']}"
                elements.append(Paragraph(q_text, normal_style))
                
                a_text = f"<b>Answer:</b> {qa['answer'][:200]}..." if len(qa['answer']) > 200 else f"<b>Answer:</b> {qa['answer']}"
                elements.append(Paragraph(a_text, normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        # Feedback & Recommendations
        elements.append(PageBreak())
        elements.append(Paragraph("Final Feedback & Recommendations", heading_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Parse and format feedback (remove markdown formatting)
        feedback_clean = feedback_text.replace("**", "").replace("##", "").replace("*", "")
        feedback_paragraphs = feedback_clean.split('\n\n')
        for para in feedback_paragraphs:
            if para.strip():
                elements.append(Paragraph(para.strip(), normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph("Thank you for using AI Job Interview Simulator!", heading_style))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"[PDF] Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


def handle_user_response(user_input):
    global interview_state
    
    # Ensure interview_questions_count exists (for backward compatibility)
    if "interview_questions_count" not in interview_state:
        interview_state["interview_questions_count"] = 0
    
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
        # Ensure backward compatibility
        if "current_question" not in interview_state:
            interview_state["current_question"] = None
        if "questions_history" not in interview_state:
            interview_state["questions_history"] = []
        if "current_question_skill" not in interview_state:
            interview_state["current_question_skill"] = None
        
        # User is responding to the current question
        if interview_state["current_question"] is not None:
            # Store the Q&A pair
            qa_pair = {
                "question": interview_state["current_question"],
                "answer": user_input,
                "skill": interview_state.get("current_question_skill", "Unknown")
            }
            interview_state["questions_history"].append(qa_pair)
            interview_state["responses"].append(user_input)
            
            # NOW increment the counters (only after user responds)
            skill = interview_state.get("current_question_skill")
            if skill and skill in interview_state["questions_per_skill"]:
                interview_state["questions_per_skill"][skill] += 1
            interview_state["total_questions_asked"] += 1
            interview_state["interview_questions_count"] += 1
            
            print(f"[Interview] ✓ Question {interview_state['interview_questions_count']}/5 answered")
            print(f"[Interview] Q: {interview_state['current_question'][:50]}...")
            print(f"[Interview] A: {user_input[:50]}...")
            
            # Clear current question (mark as answered)
            interview_state["current_question"] = None
            interview_state["current_question_skill"] = None
        
        # Generate next question
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
        email = request.form.get("email")
        password = request.form.get("password")
        face_data = request.form.get("faceData")
        
        if not username or not password or not email:
            flash("Username, email and password are required!", "error")
            return redirect(url_for("register"))
        
        if not face_data:
            flash("Face registration is mandatory! Please capture your face.", "error")
            return redirect(url_for("register"))
        
        users = load_users()
        if username in users:
            flash("Username already exists!", "error")
            return redirect(url_for("register"))
        
        # Save face data - now mandatory
        face_registered = save_face_image(email, face_data)

        # Backward compatible: accept boolean True/False or dict with details
        if isinstance(face_registered, dict):
            if not face_registered.get("success", False):
                msg = face_registered.get("message", "Failed to save face data. Please try again.")
                flash(msg, "error")
                return redirect(url_for("register"))
        else:
            if not face_registered:
                flash("Failed to save face data. Please try again.", "error")
                return redirect(url_for("register"))
        
        users[username] = {
            "password": generate_password_hash(password),
            "email": email,
            "face_registered": True
        }
        save_users(users)
        
        flash("Registration successful with face verification! Please log in.", "success")
        
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        face_data = request.form.get("faceData")
        
        users = load_users()
        
        if username not in users:
            flash("Invalid username!", "error")
            return redirect(url_for("login"))
        
        user = users[username]
        
        # Check if user has face registered
        if not user.get("face_registered", False):
            flash("This account doesn't have face verification. Please register a new account.", "error")
            return redirect(url_for("login"))
        
        # Check if user has email (required for face verification)
        email = user.get("email")
        if not email:
            flash("Invalid account data. Please register a new account.", "error")
            return redirect(url_for("login"))
        
        # Step 1: Verify face (mandatory)
        if not face_data:
            flash("Face verification is required! Please capture your face.", "error")
            return redirect(url_for("login"))
        
        verification_result = verify_face(email, face_data)
        
        if not verification_result["success"]:
            flash(f"Face verification failed: {verification_result['message']}", "error")
            return redirect(url_for("login"))
        
        # Step 2: Verify password (mandatory)
        if not password:
            flash("Password is required!", "error")
            return redirect(url_for("login"))
        
        if not check_password_hash(user["password"], password):
            flash("Invalid password!", "error")
            return redirect(url_for("login"))
        
        # Both verifications passed
        session.permanent = True
        session["username"] = username
        similarity = verification_result.get('similarity', 0)
        flash(f"Login successful! Face verified ({similarity:.1%} match) ✓ Password verified ✓", "success")
        return redirect(url_for("index"))
            
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

    if file.filename.lower().endswith(".pdf"):
        document_content = extract_text_from_pdf(file)
    elif file.filename.lower().endswith(".txt"):
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
        "interview_questions_count": 0,
        "responses": [],
        "video_metrics": {
            "eye_contact": 0,
            "sentiment": "neutral",
            "facial_expression": "neutral",
            "speech_clarity": "moderate",
            "confidence_level": "moderate",
            "face_count": 1,        # default single person in frame
            "multiple_faces": False
        },
        "current_question": None,
        "current_question_skill": None,
        "questions_history": [],
        "cached_feedback": None  # Store feedback to avoid regenerating
    }
    
    response = analyze_resume(document_content)
    return jsonify({
        "response": response,
        "metrics": None,
        "tips": None
    })


@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze a single webcam frame for face count and basic metrics.

    Expects JSON: { "image": "data:image/jpeg;base64,..." }
    Returns JSON: { "success": True, "face_count": N, "multiple_faces": bool, "metrics": {...} }
    """
    if "username" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    try:
        data = request.get_json() or {}
        image_data = data.get('image')
        if not image_data:
            return jsonify({"success": False, "message": "No image provided"}), 400

        # Support data URLs
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "message": "Could not decode image"}), 400

        # Detect faces using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        face_count = len(faces)

        # Build some simple pseudo-metrics when a face is present
        metrics = {
            "face_count": face_count,
            "multiple_faces": face_count > 1,
            "eye_contact": random.randint(30, 90) if face_count >= 1 else 0,
            "sentiment": random.choice(["positive", "neutral", "negative"]) if face_count >= 1 else "neutral",
            "facial_expression": random.choice(["neutral", "smiling", "confused", "engaged"]) if face_count >= 1 else "neutral",
            "speech_clarity": random.choice(["clear", "moderate", "muffled"]) if face_count >= 1 else "moderate",
            "confidence_level": random.choice(["low", "moderate", "high"]) if face_count >= 1 else "moderate"
        }

        return jsonify({"success": True, "face_count": face_count, "multiple_faces": face_count > 1, "metrics": metrics})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/get-report', methods=['GET'])
def get_report():
    """Get the interview report data and feedback."""
    if "username" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    global interview_state
    
    try:
        report_data = generate_report_data()
        
        if not report_data:
            return jsonify({"success": False, "message": "Could not generate report"}), 500
        
        # Generate feedback if completed
        if interview_state.get("stage") == "completed":
            feedback = generate_feedback()
            report_data["feedback"] = feedback
        else:
            report_data["feedback"] = "Interview not yet completed."
        
        return jsonify({
            "success": True,
            "report": report_data
        })
    except Exception as e:
        print(f"[Report] Error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/download-report', methods=['GET'])
def download_report():
    """Download the interview report as PDF."""
    if "username" not in session:
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    global interview_state
    
    try:
        report_data = generate_report_data()
        
        if not report_data:
            return jsonify({"success": False, "message": "Could not generate report"}), 500
        
        # Generate or get cached feedback
        if interview_state.get("stage") == "completed":
            feedback = interview_state.get("cached_feedback") or generate_feedback()
            report_data["feedback"] = feedback
        else:
            feedback = "Interview not yet completed."
            report_data["feedback"] = feedback
        
        # Create PDF
        pdf_buffer = create_pdf_report(report_data, feedback)
        
        if not pdf_buffer:
            return jsonify({"success": False, "message": "Could not create PDF"}), 500
        
        # Send file with universal compatibility
        filename = f"interview_report_{session.get('username', 'candidate')}.pdf"
        try:
            # Try newer Flask API first
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        except TypeError:
            # Fallback for older Flask versions
            return send_file(
                pdf_buffer,
                mimetype='application/pdf',
                as_attachment=True,
                attachment_filename=filename
            )
    except Exception as e:
        print(f"[PDF Download] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
