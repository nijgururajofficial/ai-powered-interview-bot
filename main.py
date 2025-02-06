import streamlit as st
import openai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json
from utils import extract_text_from_pdf, extract_score_from_response

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatbotInterviewAgent:
    THRESHOLDS = {
        'technical': 0.7,
        'dsa': 0.6,
        'behavioral': 0.65
    }

    def __init__(self, resume_text: str, job_description: str):
        self.resume_text = resume_text
        self.job_description = job_description
        
        # Initialize session state for interview progress
        if 'interview_stage' not in st.session_state:
            st.session_state.interview_stage = 'resume_screening'
        if 'current_question_index' not in st.session_state:
            st.session_state.current_question_index = 0
        if 'stage_questions' not in st.session_state:
            st.session_state.stage_questions = []
        if 'stage_scores' not in st.session_state:
            st.session_state.stage_scores = []
        if 'interview_complete' not in st.session_state:
            st.session_state.interview_complete = False

    def _parse_questions(self, raw_response: str, key: str) -> List[Dict]:
        """Parse questions from OpenAI response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL | re.IGNORECASE)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                return parsed_data.get(key, [])
            return json.loads(raw_response).get(key, [])
        except Exception as e:
            st.error(f"Question parsing error: {e}")
            return []

    def evaluate_response(self, question: Dict, response: str) -> float:
        """AI-powered response evaluation"""
        prompt = f"""Evaluate this interview response:
        Question: {question['question']}
        Response: {response}

        Scoring Criteria:
        {json.dumps(question.get('scoring_criteria', {}))}

        Provide a score out of 10 with brief justification.
        Return JSON: {{"score": 0-10, "feedback": "Brief evaluation"}}
        """

        try:
            eval_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide objective, fair scoring for interview responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            
            result = json.loads(eval_response.choices[0].message.content)
            return result['score'] / 10.0
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            return 0.0

    def generate_technical_questions(self) -> List[Dict]:
        """Generate technical questions with scoring criteria"""
        prompt = f"""Generate 4-5 technical interview questions with explicit scoring criteria:
        Resume Highlights: {self.resume_text}
        Job Description: {self.job_description}

        For each question, provide:
        - Question
        - Scoring rubric (0-10 points)
        - Key evaluation areas

        Return as JSON:
        {{
            "technical_questions": [
                {{
                    "question": "Technical question",
                    "scoring_criteria": {{
                        "technical_depth": "Points for depth of technical knowledge",
                        "problem_solving": "Points for approach and solution",
                        "communication": "Points for clear explanation"
                    }},
                    "max_score": 10
                }}
            ]
        }}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate technical interview questions with precise scoring mechanisms."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return self._parse_questions(response.choices[0].message.content, 'technical_questions')

    def generate_dsa_questions(self) -> List[Dict]:
        """Generate DSA questions with scoring mechanism"""
        prompt = """Generate 2 DSA problems with clear scoring criteria:
        - Include problem statement
        - Provide evaluation points
        - Define scoring for implementation, optimization, and explanation

        Return as JSON:
        {
            "dsa_questions": [
                {
                    "question": "DSA problem",
                    "scoring_criteria": {
                        "correct_solution": "Points for solving the problem",
                        "time_complexity": "Points for efficient implementation",
                        "code_quality": "Points for clean, readable code"
                    },
                    "max_score": 10
                }
            ]
        }
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate DSA interview questions with comprehensive scoring mechanism."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return self._parse_questions(response.choices[0].message.content, 'dsa_questions')

    def generate_behavioral_questions(self) -> List[Dict]:
        """Generate behavioral questions with scoring mechanism"""
        prompt = f"""Generate behavioral questions with scoring criteria:
        Resume Context: {self.resume_text}

        Return as JSON:
        {{
            "behavioral_questions": [
                {{
                    "question": "Behavioral interview question",
                    "scoring_criteria": {{
                        "situation_description": "Points for clear context",
                        "problem_solving": "Points for approach and resolution",
                        "leadership_qualities": "Points for demonstrating team skills"
                    }},
                    "max_score": 10
                }}
            ]
        }}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Generate behavioral questions with precise scoring mechanism."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return self._parse_questions(response.choices[0].message.content, 'behavioral_questions')
            
    def compare_resume_with_job_description(self):
        """Resume screening logic - kept from your original code"""
        prompt = f"""
            You are an expert career advisor. Evaluate how well a resume matches a job description.
            - The resume is provided below.
            - The job description is provided afterwards.
            Please provide a score on a scale of 1 to 10 (where 1 means a poor match and 10 means an excellent match)
            and include a brief explanation for the score.

            Resume:
            {self.resume_text}

            Job Description:
            {self.job_description}

            Please respond in the following format:
            Score: <score>
            Explanation: <explanation>
            """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert career advisor who helps candidates improve their resume alignment with job descriptions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return f"An error occurred while contacting OpenAI API: {e}"

    def conduct_chatbot_interview(self):
        """Conduct interview in a chatbot style"""
        if st.session_state.interview_complete:
            return
            
        # Display current stage header
        stage_headers = {
            'technical': "Technical Interview Stage",
            'dsa': "Problem Solving Stage",
            'behavioral': "Behavioral Interview Stage"
        }
        
        if st.session_state.interview_stage in stage_headers:
            st.header(stage_headers[st.session_state.interview_stage])

        # Handle different stages
        if st.session_state.interview_stage == 'resume_screening':
            self._handle_resume_screening()
        elif st.session_state.interview_stage == 'technical':
            self._handle_interview_stage('technical', self.generate_technical_questions)
        elif st.session_state.interview_stage == 'dsa':
            self._handle_interview_stage('dsa', self.generate_dsa_questions)
        elif st.session_state.interview_stage == 'behavioral':
            self._handle_interview_stage('behavioral', self.generate_behavioral_questions)

    def _handle_resume_screening(self):
        """Handle the resume screening stage"""
        result = self.compare_resume_with_job_description()
        st.markdown(result)
        
        score = extract_score_from_response(result)
        if score is not None:
            normalized_score = score / 10.0
            if normalized_score >= 0.7:
                st.info("Your resume has a strong match! Let's begin the technical interview.")
                st.session_state.interview_stage = 'technical'
                # Add a button to start the interview
                if st.button("Begin Technical Interview"):
                    st.experimental_rerun()
            else:
                st.warning("Your resume did not meet our requirements. Thank you for your interest.")
                st.session_state.interview_complete = True

    def _handle_interview_stage(self, stage: str, question_generator):
        """Handle each interview stage in a chatbot style"""
        # Generate questions if not already generated
        if not st.session_state.stage_questions:
            st.session_state.stage_questions = question_generator()
            st.session_state.current_question_index = 0
            st.session_state.stage_scores = []
            
        # Get current question
        if st.session_state.current_question_index < len(st.session_state.stage_questions):
            current_question = st.session_state.stage_questions[st.session_state.current_question_index]
            
            # Display current question
            st.write(f"Question {st.session_state.current_question_index + 1}:")
            st.write(current_question['question'])
            
            # Get user response
            user_response = st.text_area("Your answer:", key=f"response_{stage}_{st.session_state.current_question_index}")
            
            # Handle response submission
            if st.button("Submit Answer"):
                if user_response.strip():
                    # Evaluate response
                    score = self.evaluate_response(current_question, user_response)
                    st.session_state.stage_scores.append(score)
                    
                    # Show feedback
                    st.write(f"Score: {score:.2f}")
                    
                    # Move to next question
                    st.session_state.current_question_index += 1
                    
                    # Check if stage is complete
                    if st.session_state.current_question_index >= len(st.session_state.stage_questions):
                        self._handle_stage_completion(stage)
                    
                    st.experimental_rerun()
                else:
                    st.error("Please provide an answer before proceeding.")

    def _handle_stage_completion(self, stage: str):
        """Handle the completion of an interview stage"""
        avg_score = sum(st.session_state.stage_scores) / len(st.session_state.stage_scores)
        
        if avg_score >= self.THRESHOLDS[stage]:
            next_stages = {'technical': 'dsa', 'dsa': 'behavioral', 'behavioral': 'complete'}
            if stage in next_stages:
                if next_stages[stage] == 'complete':
                    st.balloons()
                    st.success("Congratulations! You've successfully completed all interview stages!")
                    st.session_state.interview_complete = True
                else:
                    st.success(f"Excellent! You've passed the {stage} stage.")
                    st.session_state.interview_stage = next_stages[stage]
                    st.session_state.stage_questions = []
                    st.session_state.current_question_index = 0
                    st.session_state.stage_scores = []
        else:
            st.error(f"Thank you for your time, but you did not meet our requirements for the {stage} stage.")
            st.session_state.interview_complete = True

def main():
    st.title("AI-Powered Interview Simulator")
    
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Job Description", height=200)

    if st.button("Start Interview"):
        if resume_pdf and job_description.strip():
            resume_text = extract_text_from_pdf(resume_pdf)
            
            # Create or get interview agent
            if 'interview_agent' not in st.session_state:
                st.session_state.interview_agent = ChatbotInterviewAgent(resume_text, job_description)
            
            # Conduct interview
            st.session_state.interview_agent.conduct_chatbot_interview()
                
        else:
            st.error("Please upload resume and provide job description")

if __name__ == "__main__":
    main()