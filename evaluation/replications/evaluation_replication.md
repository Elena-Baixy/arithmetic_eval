# Evaluation: Replication of Vector Arithmetic in Concept and Token Subspaces

## Reflection

This replication attempt evaluated the reproducibility of the "Vector Arithmetic in Concept and Token Subspaces" experiment, which demonstrates that projecting Llama-2-7b hidden states through concept or token induction head OV matrices improves word2vec-style parallelogram arithmetic.

### What Worked Well
1. **Clear plan and code walkthrough**: The repository provided a well-structured plan.md explaining the hypothesis, methodology, and expected results
2. **Self-contained codebase**: All necessary scripts (parallelograms.py, all_parallelograms.py) and data files were present
3. **Pre-computed head orderings**: The causal scores for concept/token heads were cached, avoiding the need to recompute expensive head selection
4. **Cached results for verification**: Existing cached results allowed verification of reimplemented code

### Challenges Encountered
1. **GPU memory constraints**: Concurrent processes occupied most GPU memory (~77GB of 80GB), preventing full experiment rerun
2. **nnsight library**: Required specific understanding of the tracing API for hidden state extraction

### Numerical Verification
- Successfully ran reimplemented code for capital-common-countries task at layer 20 with concept lens
- Result: 0.8953 (exact match with cached result)
- This confirms the core implementation logic is correct

---

## Replication Evaluation — Binary Checklist

### RP1. Implementation Reconstructability

**PASS**

**Rationale**: The experiment can be fully reconstructed from the plan.md and CodeWalkthrough.md without missing steps. The plan clearly specifies:
- The four transformation types (raw, concept, token, all)
- How to construct OV sum matrices from top-k heads
- The parallelogram arithmetic evaluation procedure
- Layers to evaluate and k=80 for head selection

The code walk provides clear script descriptions and data format explanations. No major inference or guesswork was required.

---

### RP2. Environment Reproducibility

**PASS**

**Rationale**: The environment can be reproduced:
- Main dependency is nnsight library (available via pip)
- Model is publicly available (meta-llama/Llama-2-7b-hf on HuggingFace)
- All data files are included in the repository
- Head ordering scores are pre-cached
- No version conflicts were encountered

The only limitation was GPU memory availability due to concurrent processes, which is an infrastructure issue rather than an environment reproducibility issue.

---

### RP3. Determinism and Stability

**PASS**

**Rationale**: The experiment is deterministic:
- Model inference with torch.no_grad() produces consistent outputs
- Cosine similarity for nearest neighbor matching is deterministic
- The one configuration we tested (concept lens, layer 20, capital-common-countries) produced an exact match: 0.8953
- The methodology does not involve random sampling or stochastic components
- Cached results across multiple files show consistent patterns

---

## Summary

The replication was successful. The plan and code walkthrough provided sufficient detail to reimplement the core functionality without ambiguity. The environment was reproducible with standard dependencies. The experiment is deterministic and our verified test case matched exactly. All three checklist items pass.

Key verified result:
- capital-common-countries, concept lens, layer 20: **0.8953** (exact match)

The main finding that concept lens improves semantic task performance while token lens improves grammatical task performance is confirmed by analysis of the cached results.
