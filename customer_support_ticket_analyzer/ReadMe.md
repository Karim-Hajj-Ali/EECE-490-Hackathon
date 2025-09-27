# Customer Support Ticket Analyzer

A small Flask web application to help support teams triage customer tickets and understand common issues.

# Features

Priority Ranking:
Analyze ticket subjects and descriptions to assign a priority score and label (High / Medium / Low).
Helps support teams focus on the most urgent tickets first.

Top Terms Analysis:
Identify the most frequent phrases (bigrams) across tickets to discover recurring issues and pain points.
Priority-weighted, so more urgent tickets have more influence on the top terms.

Web Interface:
View all tickets in order of priority.
Explore a separate tab with the most prevalent issues.

Installation

Clone the repository:

git clone https://github.com/Karim-Hajj-Ali/EECE-490-Hackathon
cd customer_support_ticket_analyzer


Create and activate a virtual environment (optional but recommended):

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Make sure the dataset is available:

data/customer_support_tickets.csv

Usage

Start the Flask app:

python app.py


Open your browser and go to:

http://127.0.0.1:5000/


Tickets tab: View all tickets sorted by priority.

Top Terms tab: See the most common two-word phrases across tickets.

Customization

Adjust keyword weights in app.py (KEYWORD_WEIGHTS) to better match your company's SLA or ticket semantics.

Change the number of top terms displayed by modifying the top_n parameter in the top_issues() function.

Requirements

Python 3.10+

Flask

pandas

scikit-learn
