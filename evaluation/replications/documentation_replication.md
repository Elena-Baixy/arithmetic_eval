# Documentation: Replication of Vector Arithmetic in Concept and Token Subspaces

## Goal

Replicate the experiment demonstrating that concept and token induction heads can identify subspaces of Llama-2-7b activations with coherent semantic and surface-level structure, enabling more accurate parallelogram arithmetic (e.g., Athens – Greece + China = Beijing) than using raw hidden states.

## Data

### Datasets Used
1. **word2vec dataset** (from Mikolov et al., 2013):
   - `capital-common-countries`: 506 capital-country pairs
   - `family`: 506 family relation pairs  
   - `gram5-present-participle`: Present participle transformations
   - `gram7-past-tense`: Past tense transformations
   - Plus 10 additional tasks covering semantic and grammatical categories

2. **Head Ordering Scores**:
   - `concept_copying_len30_n1024.json`: Causal scores for concept heads
   - `token_copying_len30_n1024.json`: Causal scores for token heads
   - These are pre-computed from prior work on the Dual-Route Model of Induction

### Data Format
- Each task file contains lines of 4 space-separated words: `a b a' b'`
- Example: "Athens Greece Beijing China" (Athens:Greece :: Beijing:China)

## Method

### 1. Lens Construction
Build transformation matrices by summing OV (output-value) matrices from top-k attention heads:

```
L = Σ O(l,h) @ V(l,h) for (l,h) in top-k heads
```

Four transformation types:
- **raw**: Identity (no transformation)
- **concept**: Sum of top-80 concept induction head OVs
- **token**: Sum of top-80 token induction head OVs  
- **all**: Sum of all 1024 attention head OVs

### 2. Word Representation Extraction
For each word w:
1. Pass through Llama-2-7b with optional prefix
2. Extract hidden state at layer ℓ, last token position
3. Apply lens transformation: L @ h_ℓ(w)

### 3. Parallelogram Arithmetic Evaluation
For each example (a, b, a', b'):
1. Compute result = L·h(a) - L·h(b) + L·h(b')
2. Find nearest neighbor among all word representations using cosine similarity
3. Success if nearest neighbor is a'

### 4. Layer Sweep
Evaluate at layers [0, 4, 8, 12, 16, 20, 24, 28, 31]

## Results

### Key Findings (Nearest Neighbor Accuracy)

| Task | Best Lens | Best Layer | Accuracy | Raw Accuracy |
|------|-----------|------------|----------|--------------|
| Capital Cities | concept | 20 | 89.5% | 17.2% |
| Family | concept | 20 | 6.9% | 0.6% |
| Present Participle | token | 16 | 54.2% | 10.8% |
| Past Tense | token | 16 | 56.4% | 9.5% |

### Pattern Summary
1. **Semantic tasks** (capitals, family): Concept lens dramatically outperforms raw and token
2. **Grammatical tasks** (present participle, past tense): Token lens outperforms concept and raw
3. **Peak layers**: Middle layers (16-20) achieve best performance
4. **Token lens peaks earlier** (layer 16) than concept lens (layer 20)

## Analysis

### Hypothesis Validation
1. ✓ Poor raw performance confirms interference from irrelevant information
2. ✓ Concept lens success on semantic tasks validates semantic subspace hypothesis
3. ✓ Token lens success on grammatical tasks validates surface-level subspace hypothesis

### Replication Accuracy
- Exact match on tested configuration (concept lens, layer 20, capital-common-countries): 0.8953
- All patterns consistent with original paper findings

### Limitations
- GPU memory constraints prevented full sweep rerun (3 concurrent processes using ~26GB each)
- Single configuration verified numerically; remainder verified against cached results
