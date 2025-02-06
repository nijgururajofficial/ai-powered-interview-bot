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
        # Use stage-specific keys for questions and progress
        if f'technical_questions' not in st.session_state:
            st.session_state.technical_questions = []
        if f'dsa_questions' not in st.session_state:
            st.session_state.dsa_questions = []
        if f'behavioral_questions' not in st.session_state:
            st.session_state.behavioral_questions = []
        if 'current_stage_index' not in st.session_state:
            st.session_state.current_stage_index = {}
        if 'stage_scores' not in st.session_state:
            st.session_state.stage_scores = {}
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
        prompt = f"""Generate 4 behavioral questions with scoring criteria:
        Resume Context: {self.resume_text}
        Job Description: {self.job_description}

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
        """Resume screening logic"""
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

    def _handle_resume_screening(self):
        """Handle the resume screening stage"""
        result = self.compare_resume_with_job_description()
        st.markdown(result)
        
        score = extract_score_from_response(result)
        if score is not None:
            normalized_score = score / 10.0
            if normalized_score >= 0.4:
                st.success("Your resume has a strong match! Let's begin the interview.")
                st.session_state.interview_stage = 'technical'
                st.rerun()
            else:
                st.error("Your resume did not meet our requirements. Thank you for your interest.")
                st.session_state.interview_complete = True

    def conduct_chatbot_interview(self):
        """Conduct interview in a chatbot style"""
        if st.session_state.interview_complete:
            st.info("Interview complete. Thank you for participating!")
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
        else:
            self._handle_interview_stage(stage=st.session_state.interview_stage)

    def _handle_interview_stage(self, stage: str):
        """Handle each interview stage in a chatbot style"""
        # Stage-specific session state keys
        questions_key = f"{stage}_questions"
        index_key = f"{stage}_index"
        scores_key = f"{stage}_scores"

        # Initialize stage-specific session state
        if not st.session_state.get(questions_key):
            question_generator = getattr(self, f"generate_{stage}_questions")
            st.session_state[questions_key] = question_generator()
            st.session_state[index_key] = 0
            st.session_state[scores_key] = []

        questions = st.session_state[questions_key]
        current_index = st.session_state[index_key]

        if current_index < len(questions):
            current_question = questions[current_index]
            
            # Display current question
            st.subheader(f"Question {current_index + 1} of {len(questions)}")
            st.markdown(f"**{current_question['question']}**")
            
            # Get user response
            response_key = f"{stage}_response_{current_index}"
            user_response = st.text_area("Your answer:", key=response_key)
            
            # Handle response submission
            if st.button("Submit Answer", key=f"{stage}_submit_{current_index}"):
                if user_response.strip():
                    # Evaluate response
                    score = self.evaluate_response(current_question, user_response)
                    st.session_state[scores_key].append(score)
                    
                    # Show feedback
                    st.write(f"**Score:** {score:.2f}/1.0")
                    if current_index < len(questions) - 1:
                        st.write("---")
                    
                    # Move to next question
                    st.session_state[index_key] += 1
                    st.rerun()
                else:
                    st.error("Please provide an answer before proceeding.")
        else:
            self._handle_stage_completion(stage)

    def _handle_stage_completion(self, stage: str):
        """Handle the completion of an interview stage"""
        scores = st.session_state[f"{stage}_scores"]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score >= self.THRESHOLDS[stage]:
            next_stages = {
                'technical': 'dsa',
                'dsa': 'behavioral',
                'behavioral': 'complete'
            }
            next_stage = next_stages.get(stage, 'complete')
            
            if next_stage == 'complete':
                st.balloons()
                st.success("Congratulations! You've successfully completed all interview stages!")
                st.session_state.interview_complete = True
            else:
                st.session_state.interview_stage = next_stage
                # Reset previous stage's progress
                st.session_state[f"{next_stage}_questions"] = []
                st.session_state[f"{next_stage}_index"] = 0
                st.session_state[f"{next_stage}_scores"] = []
                st.rerun()
        else:
            st.error(f"Thank you for your time, but you did not meet our requirements for the {stage} stage. You scored: {avg_score:.2f}")
            st.session_state.interview_complete = True

def main():
    st.title("AI-Powered Interview Simulator")
    
    resume_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Job Description", height=200)

    # Check if interview is already initialized
    if 'interview_agent' in st.session_state:
        st.session_state.interview_agent.conduct_chatbot_interview()
    else:
        if st.button("Start Interview"):
            if resume_pdf and job_description.strip():
                resume_text = extract_text_from_pdf(resume_pdf)
                st.session_state.interview_agent = ChatbotInterviewAgent(resume_text, job_description)
                st.rerun()
            else:
                st.error("Please upload resume and provide job description")

if __name__ == "__main__":
    main()