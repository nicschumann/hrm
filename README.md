# Hierarchical Reasoning Model

An implementation of [Hierarchical Reasoning Model](https://arxiv.org/pdf/2506.21734) by Sapient Intelligence / Wang et al. The goal here is to reproduce various features of the paper while building a public implementation that's both readible and extensible to domains outside of the paper.

## References

- Schwartzschild et al. Datasets for Studying Generalization from Easy to Hard Examples. [here](https://arxiv.org/pdf/2108.06011). _Used this as a source of training data for maze problems in particular. See `data/` for usage!_

- Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. [here](https://arxiv.org/pdf/2104.09864). _Just a callback to the implementation of RoPE, which is used for positional encoding in the paper. Note that we use the `lucidrains` implementation of rotary embeddings: [here](https://github.com/lucidrains/rotary-embedding-torch)._
