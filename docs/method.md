# Method

## Motivation

Naïve ERM fails under subpopulation shift, often relying on spurious or majority-correlated features. DPE counters this by introducing ensemble members that learn different latent substructures.

## Architecture

1. **Stage 1**: Train a feature extractor using standard ERM.
2. **Stage 2**: Replace the final layer with an ensemble of prototype classifiers.

Each prototype learns to classify based on different features or samples. We enforce diversity via:

- **Inter-Prototype Similarity (IPS)** loss
- **Bootstrap sampling** of validation subsets

## Classification Rule

For input \( x \), prediction is made by averaging predictions across \( N \) prototypes per class:

\[
\hat{y} = \arg\max_k \frac{1}{N} \sum_{i=1}^{N} \exp\left(-D(f(x), p_{k}^{(i)})\right)
\]

## IPS Loss

\[
\mathcal{L}_{\text{IPS}} = \sum_{k=1}^{K} \sum_{i \ne j} \frac{|\langle p_k^{(i)}, p_k^{(j)} \rangle|}{n \cdot d}
\]

Where \( p_k^{(i)} \) is the \( i \)-th prototype for class \( k \).


