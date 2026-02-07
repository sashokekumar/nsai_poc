# Level 0: Baseline TF-IDF Intent Classification

## Overview
Level 0 represents the baseline statistical classifier for intent classification using traditional machine learning techniques. This level establishes the foundation for NSAI progression.

## Architecture
**Pure Statistical Approach**
- **Algorithm**: TF-IDF + Logistic Regression
- **Features**: 5000 TF-IDF features with 1-2 grams
- **Confidence Thresholding**: Predictions below 0.7 confidence are marked as "abstain"
- **No Rules**: Direct model predictions without symbolic intervention

## Dataset
- **Source**: `../data/intents_base.csv`
- **Records**: 614 utterances
- **Classes**: 
  - `investigate` (149 samples)
  - `execution` (150 samples)
  - `summarization` (146 samples)
  - `out_of_scope` (169 samples)

## Model Configuration
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english'
)

LogisticRegression(
    solver='lbfgs',
    random_state=42,
    max_iter=1000
)
```

## Key Features
1. **Confidence Threshold**: 0.7 minimum for accepting predictions
2. **Abstain Logic**: Low-confidence predictions are flagged for human review
3. **Token-Level Evidence**: Extracts top contributing tokens for explainability
4. **Performance Metrics**: ~91% accuracy on test set

## Files
- `level0_tfidf_classification.ipynb`: Main notebook (22 cells)
- `models/`: Saved model artifacts (if any)
- `artifacts/`: Output files and visualizations
 - `level0_model.py`: Reusable Level0Classifier module for predict/save/load (used by validation)

## Usage
```bash
# Navigate to level0 directory
cd level0

# Open notebook
jupyter notebook level0_tfidf_classification.ipynb

# Or use VS Code
code level0_tfidf_classification.ipynb
```

## Execution
Run all cells sequentially:
1. Import dependencies
2. Load and validate data
3. Train/test split (80/20, stratified)
4. Train TF-IDF + LR pipeline
5. Evaluate performance
6. Test with sample utterances
7. Inspect confidence thresholding behavior

## Output Structure
```python
{
    'predicted_intent': 'execution',
    'confidence': 0.92,
    'evidence_tokens': [
        {'token': 'restart', 'weight': 1.23},
        {'token': 'nginx', 'weight': 0.87},
        ...
    ]
}
```

## Limitations
- No symbolic reasoning
- No execution safety gates
- No multi-signal aggregation
- Fixed confidence threshold
- No context awareness

## Next Level
→ **Level 1**: Adds neuro-symbolic decision layer with explicit rules, priority-based precedence, and separation of intent from decision state.

---

**Level 0 Status**: ✅ Complete  
**Accuracy**: ~91%  
**Approach**: Pure Statistical Learning
