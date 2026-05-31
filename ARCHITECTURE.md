# North India College Predictor – System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI LAYER                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐   │
│  │ Stream Select│ │ Input Scores │ │ Preference Config    │   │
│  └──────────────┘ └──────────────┘ └──────────────────────┘   │
└──────────────────────────────┬────────────────────────────────┘
                               │
                    User Input Submission
                               │
┌──────────────────────────────▼────────────────────────────────┐
│              DATA PREPROCESSING MODULE                        │
│  ├─ Input Validation & Sanitization                           │
│  ├─ Normalization (Score scaling to 0-1 range)               │
│  ├─ Categorical Encoding (Stream, Category)                   │
│  └─ Feature Engineering (Percentile calculations)             │
└──────────────────────────────┬────────────────────────────────┘
                               │
                    Preprocessed Feature Vector
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│ ML MODEL LAYER       │                  │ RULES ENGINE LAYER   │
│                      │                  │                      │
│ ├─ Joblib Loader     │                  │ ├─ Cutoff Rules      │
│ ├─ Predict Proba     │                  │ ├─ Location Rules    │
│ └─ Confidence Score  │                  │ └─ Category Rules    │
└──────────────────────┘                  └──────────────────────┘
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
                    Hybrid Score Computation
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│ DATABASE QUERY       │                  │ RANKING ENGINE       │
│                      │                  │                      │
│ ├─ Engineering DB    │                  │ ├─ Sort by Score     │
│ ├─ Medical DB        │                  │ ├─ Apply Filters     │
│ ├─ Commerce DB       │                  │ └─ Format Output     │
│ └─ Humanities DB     │                  │                      │
└──────────────────────┘                  └──────────────────────┘
        │                                             │
        └──────────────────────┬──────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                    OUTPUT FORMATTER                           │
│  ├─ College Rankings (Top 10)                                │
│  ├─ Match Confidence & Probability                           │
│  ├─ Rationale Generation                                     │
│  └─ Explainability Metrics                                   │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│              STREAMLIT UI OUTPUT DISPLAY                      │
│  ├─ Ranked College Cards with Metadata                       │
│  ├─ Confidence Gauge & Match Percentage                      │
│  └─ Prediction Rationale Explanation                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Streamlit Frontend (`app.py`)

**Responsibilities:**
- User session management
- Input form rendering (stream, scores, preferences)
- Real-time prediction triggering
- Result visualization and formatting

**Key Functions:**
```python
@st.cache_resource
def load_models(stream):
    # Load pre-trained .pkl models per stream
    
def render_input_form():
    # Stream selection, score input, preferences
    
def trigger_prediction(user_input):
    # Orchestrate end-to-end inference pipeline
```

---

### 2. Data Preprocessing Module (`utils/preprocessing.py`)

**Input:** Raw user submission (stream, scores, preferences)

**Processing:**
- Validation against schema
- Normalization (min-max scaling for scores)
- Categorical encoding (stream → numeric)
- Feature vector construction

**Output:** Normalized NumPy array (shape: [1, n_features])

```python
def preprocess_input(user_data, stream):
    """
    Args:
        user_data: dict with keys {score_12, entrance_score, category, location}
        stream: str in {'Engineering', 'Medical', 'Commerce', 'Humanities'}
    Returns:
        np.ndarray of shape (1, n_features)
    """
```

---

### 3. ML Model Layer (`models/`)

**Structure:**
```
models/
├── engineering_model.pkl      # Scikit-Learn classifier/regressor
├── medical_model.pkl          # [Future integration]
├── commerce_model.pkl         # [Future integration]
└── humanities_model.pkl       # [Future integration]
```

**Inference Pipeline:**
```python
def predict_with_model(features, stream):
    """
    Args:
        features: np.ndarray of preprocessed features
        stream: str
    Returns:
        predictions: array of class labels or probabilities
        confidence: float in [0, 1]
    """
    model = joblib.load(f'models/{stream.lower()}_model.pkl')
    return model.predict_proba(features)
```

**Per-Stream Model Details:**

| Stream | Input Features | Output Type | Model Type |
|--------|---|---|---|
| Engineering | [Score_12, Entrance, Category, etc.] | College Class | Logistic Regression / SVM |
| Medical | [NEET Score, 12th %, Category, etc.] | College Class | [TBD] |
| Commerce | [Score_12, Entrance, Specialization, etc.] | College Class | [TBD] |
| Humanities | [Score_12, Subject Combo, Category, etc.] | College Class | [TBD] |

---

### 4. Rules Engine (`utils/rules_engine.py`)

**Domain-Specific Hardcoded Rules:**

```python
def apply_rules(predictions, user_data, stream):
    """
    Apply business logic constraints to ML predictions
    
    Rules include:
    1. Cutoff thresholds (hard gates based on stream cutoffs)
    2. Location filters (preference vs. availability)
    3. Category-based rules (SC/ST/OBC reservations)
    4. Stream compatibility (validity checks)
    """
```

**Rule Examples:**
- Engineering: If score < stream_cutoff, disqualify candidates
- Medical: If NEET < admission_threshold, exclude institution
- Commerce: Apply specialization-wise preference filters
- Humanities: Consider merit-cum-means criteria

**Output:** Adjusted confidence scores with rule-derived flags

---

### 5. Database Query Module (`utils/database.py`)

**Multi-Stream Database Architecture:**

```python
class CollegeDatabase:
    """
    Modular database interface supporting dynamic stream addition.
    """
    
    def __init__(self, stream):
        self.stream = stream
        self.data = self._load_database(stream)
    
    def _load_database(self, stream):
        """
        Loads from:
        - data/engineering_colleges.csv
        - data/medical_colleges.csv
        - data/commerce_colleges.csv
        - data/humanities_colleges.csv
        """
        return pd.read_csv(f'data/{stream.lower()}_colleges.csv')
    
    def query(self, filters):
        """
        Args:
            filters: dict with keys {category, location, specialization}
        Returns:
            pd.DataFrame of matching colleges
        """
```

**Database Schema (per stream):**

| Column | Type | Purpose |
|--------|------|---------|
| college_id | int | Unique identifier |
| college_name | str | Institution name |
| cutoff_gen | float | General category cutoff |
| cutoff_obc | float | OBC category cutoff |
| location | str | State/City |
| specializations | list | Available streams/programs |
| placement_rate | float | Historical metric |
| ranking | int | National/State ranking |

---

### 6. Ranking & Output Formatter (`utils/formatter.py`)

**Input:** 
- ML predictions (probability scores)
- Rule-adjusted confidence scores
- Database query results

**Processing:**
1. **Scoring Algorithm:**
   ```
   Final_Score = (0.6 * ML_Probability) + (0.4 * Rules_Score)
   ```

2. **Ranking:** Sort colleges by Final_Score (descending)

3. **Filtering:** Apply user preferences (location, specialization)

4. **Top-K Selection:** Return top 10 recommendations

**Output:**
```python
{
    "rankings": [
        {
            "rank": 1,
            "college_name": "NIT Delhi",
            "match_score": 0.92,
            "cutoff_expected": 95.2,
            "placement_rate": 0.94,
            "specializations": ["CSE", "ECE"],
            "rationale": "High ML confidence + exceeds cutoff rules"
        },
        ...
    ],
    "overall_confidence": 0.87,
    "prediction_timestamp": "2026-04-21T14:32:00Z"
}
```

---

## Data Flow Sequence Diagram

```
User Input
    │
    ▼
┌────────────────────────┐
│ Streamlit Form Submit  │
└────────────────────────┘
    │ (stream, scores, prefs)
    ▼
┌────────────────────────┐
│ Preprocess & Validate  │◄───── Error handling & retry
└────────────────────────┘
    │ (feature_vector)
    ▼
┌────────────────────────┐
│ Load ML Model (Joblib) │
│ [stream-specific]      │
└────────────────────────┘
    │ (model object)
    ▼
┌────────────────────────┐
│ Predict Probability    │
└────────────────────────┘
    │ (prediction_array, confidence)
    ▼
┌────────────────────────┐
│ Apply Rules Engine     │
│ (cutoffs, filters)     │
└────────────────────────┘
    │ (adjusted_scores)
    ▼
┌────────────────────────┐
│ Query Stream DB        │
│ [Engineering/Med/etc]  │
└────────────────────────┘
    │ (college_dataframe)
    ▼
┌────────────────────────┐
│ Rank & Format Output   │
│ (Top 10 + rationale)   │
└────────────────────────┘
    │
    ▼
┌────────────────────────┐
│ Render in Streamlit UI │
│ (Cards, scores, expl.) │
└────────────────────────┘
```

---

## Modular Extensibility for Future Databases

### Adding New Stream (e.g., Humanities)

**Step 1:** Train ML model on Humanities dataset
```bash
python scripts/train_model.py --stream humanities --output models/humanities_model.pkl
```

**Step 2:** Create database CSV
```
data/humanities_colleges.csv
├── Columns: [college_id, college_name, cutoff_gen, location, specializations, ...]
```

**Step 3:** No code changes required
- `_load_database(stream)` auto-discovers the CSV
- `predict_with_model()` loads correct model by stream
- `apply_rules()` applies generic rules (extensible via config)

### Configuration-Driven Rules (Future Enhancement)

```yaml
# config/rules.yaml
rules:
  engineering:
    - name: "minimum_cutoff"
      threshold: 85.0
    - name: "location_match"
      weight: 0.2
  
  medical:
    - name: "neet_threshold"
      threshold: 600
    - name: "category_reservation"
      enabled: true
  
  commerce:
    - name: "specialization_filter"
      options: [BBA, CA, Economics]
```

---

## Error Handling & Resilience

| Scenario | Handling |
|----------|----------|
| Model load failure | Fallback to rule-based predictions |
| Database unavailable | Cache previous results / offline mode |
| Invalid input | User-facing validation + suggestion prompt |
| Prediction latency | Async computation with progress indicator |

---

## Performance & Optimization

- **Model Caching:** Joblib caching (loaded once per session)
- **Feature Vector Caching:** Streamlit `@st.cache_data` for user history
- **Database Indexing:** CSV → Parquet conversion (future) for 10x faster queries
- **Inference SLA:** Target < 100ms per prediction

---

## Deployment Checklist for New Stream

- [ ] ML model trained & validated (>80% accuracy)
- [ ] Database CSV created with required columns
- [ ] Rules configured in `config/rules.yaml`
- [ ] Integration tested end-to-end
- [ ] UI updated to include stream option
- [ ] Documentation & changelog updated
