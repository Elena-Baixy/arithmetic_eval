# Vector Arithmetic in Concept and Token Subspaces
*Code and data for short paper at the NeurIPS 2025 Mechanistic Interpretability Workshop. See paper website [here](https://arithmetic.baulab.info).*

In this work, we use the weights of concept and token induction heads discovered in ["The Dual-Route Model of Induction"](https://dualroute.baulab.info/) to analyze word embeddings. We find that using these heads to "focus" on semantic information can make word2vec-style analogies like `Athens - Greece + China = Beijing` work out much more cleanly than they do using raw hidden states. Doing the same with token induction heads can help with more wordform-focused word2vec tasks, like `dance - dancing + coding = code`. 

# Data
We use two datasets in this work, which each have a number of tasks. 
1. `word2vec` - original data from [Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781)
2. `fvs` - function vector tasks from [Todd et al. (2024)](https://functions.baulab.info/)

# Scripts
- Running `all_parallelograms.py` will save results in the `cache` folder for every task in the dataset specified. If you want to run the analysis with prefixes for each word (e.g. "She travelled to Athens" rather than just "Athens"), provide the `--with_prefix` flag. 
- `parallelogram_ranks.py` must be run after `all_parallelograms.py`. It chooses the best-performing layer in the vanilla setting, and evaluates performance for a range of possible low-rank approximations of the token/concept/"all" lenses at that best layer. 
- `parallelograms.py` provides helper functions for the above two scripts.
- `parallelogram_analysis.ipynb` provides plotting code for figures in the paper. 

So far, we provide code only for Llama-2-7b; if you are interested in expanding this codebase to support other tasks/models, please contact `feucht.s[at]northeastern.edu`. 

# Citing this work
Here's how you can cite this work, if you want: 
```
@inproceedings{feucht2025arithmetic,
  title={Vector Arithmetic in Concept and Token Subspaces},
  author={Sheridan Feucht and Byron Wallace and David Bau},
  booktitle={Second Mechanistic Interpretability Workshop at NeurIPS},
  year={2025},
  url={https://arithmetic.baulab.info}
}
```
