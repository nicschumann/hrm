# Hierarchical Reasoning Model

An implementation of [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734) by Sapient Intelligence / Wang et al. The goal here is to reproduce various features of the paper while building a public implementation that's both readible and extensible to domains outside of the paper.

## References

- Schwartzschild et al. Datasets for Studying Generalization from Easy to Hard Examples. [here](https://arxiv.org/pdf/2108.06011). _Used this as a source of training data for maze problems in particular. See `data/` for usage!_

- Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. [here](https://arxiv.org/pdf/2104.09864). _Just a callback to the implementation of RoPE, which is used for positional encoding in the paper. Note that we use the `lucidrains` implementation of rotary embeddings: [here](https://github.com/lucidrains/rotary-embedding-torch)._

- Prieto et al. Grokking at the edge of Numerical Stability. [here](https://arxiv.org/html/2501.04697v1). _Used for the introduction of the StableMax activation. StableMax reference implementation [here](https://github.com/QuixiAI/stablemax-orthogonal)_

## Note on Accuracy

For mazes, the model essentially predicts, for each position in the sequence, whether that position is part of the solution route, or whether the position is ignored. A perfect prediction from the model is one that predices exactly the route, and nothing else. To capture this we measure accuracy as the number of correctly predicted route positions over the number of true route positions plus the number of falsely predicted route segments. You could also think of this as:

```py
true_positives / (true_labels + false_positives)
```
