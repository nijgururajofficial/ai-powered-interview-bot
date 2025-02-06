# AI-Powered Interview Simulator

An intelligent interview simulation system that uses OpenAI's GPT-3.5 to conduct multi-stage technical interviews. The system evaluates resumes, conducts technical assessments, poses DSA problems, and assesses behavioral competencies in a chat-like interface.

## Features

- Resume screening against job descriptions
- Multi-stage interview process:
  - Technical interview (4-5 questions)
  - Data Structures & Algorithms (2 problems)
  - Behavioral assessment
- Real-time response evaluation
- Automatic progression based on performance
- Chat-style interface for natural interaction
- Score-based stage advancement
- Final job offer for successful candidates

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- PDF reader capabilities for resumes

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nijgururajofficial/ai-powered-interview-bot
cd ai-powered-initerview-bot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Required Packages

Create a `requirements.txt` file with these dependencies:
```
streamlit==1.24.0
openai==0.28
python-dotenv==1.0.0
PyPDF2==3.0.1  # For PDF processing
```

## Project Structure

```
interview-simulator/
│
├── main.py              # Main application file
├── utils.py            # Utility functions
├── requirements.txt    # Package dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
```

## Running the Application

Start the Streamlit application:
```bash
streamlit run main.py
```

The application will be available at `http://localhost:8501` by default.

## Usage Instructions

1. Upload your resume (PDF format)
2. Paste the job description
3. Click "Start Interview"
4. Progress through stages:
   - Resume screening (must score ≥ 7/10)
   - Technical questions
   - DSA problems
   - Behavioral questions
5. Receive immediate feedback and scores
6. Get final results and potential job offer

## Interview Stage Thresholds

- Resume Screening: 0.7 (7/10)
- Technical Interview: 0.7 (7/10)
- DSA Problems: 0.6 (6/10)
- Behavioral Assessment: 0.65 (6.5/10)

## Notes

- Ensure your OpenAI API key has sufficient credits
- Resume must be in PDF format
- Each stage must be passed to proceed to the next
- Responses are evaluated in real-time
- The system provides immediate feedback and scoring
