import os
import re
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------
# Utility functions
# ---------------------------

KEYWORD_WEIGHTS = {
    'urgent': 3, 'asap': 3, 'immediately': 3,
    'critical': 4, 'critical.': 4, 'down': 4, 'outage': 4,
    'payment': 2, 'billing': 2, 'error': 2, 'fail': 3, 'failed': 3,
    'unable': 2, 'lost': 4, 'data loss': 5, 'lost data': 5,
    'security': 4, 'breach': 5
}

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {
        "Ticket ID": "ticket_id",
        "Ticket Subject": "subject",
        "Ticket Description": "description",
        "Ticket Priority": "priority",
    }
    for k, v in colmap.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    if "ticket_id" not in df.columns:
        df["ticket_id"] = df.index.astype(str)
    return df

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r'[\r\n]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def compute_priority(df):
    """
    Compute a more fine-grained priority score (0-10) using heuristics.
    - Uses keywords, text length, and all-caps emphasis.
    - Maps numeric score to labels:
        high: 7-10
        medium: 4-6
        low: 0-3
    """
    out = df.copy()

    def heur_score(row):
        text = ' '.join(str(row.get(c, '') or '') for c in ('subject', 'description'))
        clean = _clean_text(text)
        score = 0.0
        for kw, w in KEYWORD_WEIGHTS.items():
            if kw in clean:
                score += w
        l = len(clean.split())
        if l > 50:
            score += 2.0
        elif l > 20:
            score += 1.0
        raw = (row.get('subject') or '') + ' ' + (row.get('description') or '')
        if re.search(r'\b[A-Z]{3,}\b', str(raw)):
            score += 1.5
        return score

    raw_scores = out.apply(heur_score, axis=1)
    max_score = raw_scores.max() if raw_scores.max() > 0 else 1
    out['priority_score'] = raw_scores / max_score * 10

    def label_from_score(s):
        if s >= 7:
            return 'high'
        elif s >= 4:
            return 'medium'
        else:
            return 'low'

    out['priority_label'] = out['priority_score'].apply(label_from_score)
    out = out.sort_values('priority_score', ascending=False).reset_index(drop=True)
    return out

def top_issues(df, top_n=20):
    """
    Return top_n bigram terms across subject+description,
    weighted by ticket priority score so urgent tickets count more.
    """
    texts = (df.get('subject', pd.Series('')) + ' ' + df.get('description', pd.Series(''))).fillna('').astype(str).tolist()
    priorities = df.get('priority_score', pd.Series(1)).tolist()  # fallback to 1 if missing

    vect = CountVectorizer(stop_words='english', ngram_range=(2,2), min_df=2)
    X = vect.fit_transform(texts)  # shape (n_tickets, n_terms)
    terms = vect.get_feature_names_out()

    weighted_counts = (X.multiply(priorities)).sum(axis=0).A1
    counts_sorted = sorted(zip(terms, weighted_counts), key=lambda x: x[1], reverse=True)

    return [(t, int(c)) for t,c in counts_sorted[:top_n]]

# ---------------------------
# Flask App
# ---------------------------

DATA_PATH = os.path.join("data", "customer_support_tickets.csv")
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ticket Triage & Issue Overview</title>
<style>
body { font-family: Inter, Arial, sans-serif; max-width: 900px; margin: 30px auto; color:#111; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
th, td { border: 1px solid #ddd; padding: 8px; text-align:left; }
tr:nth-child(even){ background-color:#f9f9f9; }
th { background:#222; color:#fff; }
.high { color: #9b1c1c; font-weight: bold; }
.medium { color: #b27b00; }
.low { color: #2b6b2b; }
.term { display:inline-block; background:#eef; padding:6px 8px; margin:4px; border-radius:6px; }
.tab { display: none; }
.tab-buttons button { padding:8px 16px; margin-right:4px; cursor:pointer; }
.tab-buttons button.active { background:#222; color:#fff; }
</style>
</head>
<body>

<h1>Ticket Triage & Issue Overview</h1>
<p>Data loaded from: <code>{{ data_path }}</code></p>

<div class="tab-buttons">
  <button onclick="showTab('tickets')" class="active">Tickets</button>
  <button onclick="showTab('terms')">Top Terms</button>
</div>

<div id="tickets" class="tab" style="display:block;">
  <h2>All Tickets (by Priority)</h2>
  <table>
    <thead><tr><th>#</th><th>ticket_id</th><th>priority_score</th><th>priority_label</th><th>subject</th><th>description</th></tr></thead>
    <tbody>
      {% for i,row in top_tickets.iterrows() %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ row.ticket_id }}</td>
        <td>{{ "%.2f"|format(row.priority_score) }}</td>
        <td class="{{ row.priority_label }}">{{ row.priority_label }}</td>
        <td>{{ row.subject or '' }}</td>
        <td>{{ row.description[:200] }}{% if row.description|length > 200 %}...{% endif %}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<div id="terms" class="tab">
  <h2>Most Prevalent Terms (Bigrams, Priority-Weighted)</h2>
  <div>
    {% for term,count in issues %}
      <span class="term">{{ term }} ({{ count }})</span>
    {% endfor %}
  </div>
</div>

<script>
function showTab(id){
    document.querySelectorAll('.tab').forEach(t=>t.style.display='none');
    document.getElementById(id).style.display='block';
    document.querySelectorAll('.tab-buttons button').forEach(b=>b.classList.remove('active'));
    event.currentTarget.classList.add('active');
}
</script>

<p style="margin-top:30px; color:#666;">Tip: adapt KEYWORD_WEIGHTS to match your company's SLA/semantics.</p>

</body>
</html>
"""

@app.route('/')
def index():
    if not os.path.exists(DATA_PATH):
        return f"<p>Data file not found at <code>{DATA_PATH}</code>.</p>", 404
    df = load_data(DATA_PATH)
    dfp = compute_priority(df)
    issues = top_issues(df, top_n=20)
    return render_template_string(HTML_TEMPLATE, data_path=DATA_PATH, top_tickets=dfp, issues=issues)

@app.route('/api/summary')
def api_summary():
    n_top = int(request.args.get('n', 10))
    n_terms = int(request.args.get('terms', 12))
    if not os.path.exists(DATA_PATH):
        return jsonify({'error': f'data file not found at {DATA_PATH}'}), 404
    df = load_data(DATA_PATH)
    dfp = compute_priority(df)
    top = dfp.head(n_top).to_dict(orient='records')
    issues = top_issues(df, top_n=n_terms)
    return jsonify({
        'top_tickets': top,
        'issue_overview': [{'term': t, 'count': int(c)} for t,c in issues]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
