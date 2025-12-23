# Plan
## Objective
Show that concept and token induction heads can identify subspaces of Llama-2-7b activations with coherent semantic and surface-level structure, enabling more accurate parallelogram arithmetic (e.g., Athens – Greece + China = Beijing) than using raw hidden states.

## Hypothesis
1. Poor parallelogram arithmetic results on raw Llama-2-7b hidden states are due to interference from irrelevant information in model activations.
2. Word2vec arithmetic is only effective when performed in a semantic subspace of model activations, not on the full hidden state space.
3. Concept and token induction heads operate in subspaces that represent different facets of words (semantic vs. surface-level).

## Methodology
1. Build concept and token lenses by summing OV matrices (O(l,h)V(l,h)) from top-k concept/token induction heads identified in prior work, creating transformations LCk and LTk.
2. Extract word embeddings by passing single words (optionally with task-specific prefixes) through Llama-2-7b and taking the last token representation at layer ℓ, then transform using lens matrices.
3. Test parallelogram arithmetic by computing Laℓ − Lbℓ + Lb'ℓ for word tuples (a,b) and (a',b') and measuring whether La'ℓ is the nearest neighbor among all candidate words.
4. Compare four settings: raw (L=Id), concept lens (L=LCk), token lens (L=LTk), and baseline using all attention heads (L=Lall), using k=80 heads.
5. Analyze effective rank of transformations by setting singular values below top-r to zero and sweeping across r to test if performance is maintained with reduced dimensionality.

## Experiments
### Capital Cities Parallelogram Arithmetic
- What varied: Transformation type (raw, concept lens, token lens, all heads) and layer ℓ
- Metric: Nearest-neighbor accuracy for completing Athens – Greece + China = Beijing
- Main result: Concept lens achieved ~80% accuracy at layer 20, compared to ~47% for raw hidden states. Token lens performed poorly (~20%).

### Family Relations Parallelogram Arithmetic
- What varied: Transformation type and layer ℓ
- Metric: Nearest-neighbor accuracy for son – daughter + mom = dad
- Main result: Concept lens performed best (~60% at layer 20), significantly better than raw (~25%) and token lens (~10%).

### Present Participle Parallelogram Arithmetic
- What varied: Transformation type and layer ℓ
- Metric: Nearest-neighbor accuracy for code – coding + dancing = dance
- Main result: Token lens achieved highest accuracy (~60% at layer 16), outperforming concept lens (~40%) and raw (~30%).

### Past Tense Parallelogram Arithmetic
- What varied: Transformation type and layer ℓ
- Metric: Nearest-neighbor accuracy for coding – coded + danced = dancing
- Main result: Token lens performed best (~65% at layer 16), better than concept lens (~45%) and raw (~35%).

### Word2Vec Tasks Across 14 Categories
- What varied: Task type (semantic vs. grammatical), transformation type, layer
- Metric: Nearest-neighbor accuracy compared to random guessing and 5-shot ICL baselines
- Main result: Concept lens excelled at semantic tasks (capitals, family), token lens at grammatical tasks (plurals, tenses). Both outperformed raw and all-heads baselines for most tasks.

### Effective Rank Analysis
- What varied: Rank r of low-rank approximation of lens matrices
- Metric: Nearest-neighbor accuracy at best layer for each task as rank is reduced
- Main result: Performance maintained down to r=256, indicating transformations effectively project onto lower-dimensional subspaces despite being technically full-rank at k=80.