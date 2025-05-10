# Visualizations

## Ensemble Prototype Diversity

We visualize cosine similarity matrices between prototypes for `Waterbird` and `Landbird` classes. Lower similarity confirms that IPS promotes prototype diversity.

![Prototype Similarity](../figures/prototype_diversification.jpg)

## Ablation Study

### Diversification Strategy Impact

Three strategies:

- Fixed subsets (no diversity)
- Random subsets
- Random + IPS (ours)

Ours consistently improves worst-group accuracy across all datasets.

![Diversification Impact](../figures/cov_loss_ablation.jpg)

## Linear vs. Prototypical Ensemble

DPE outperforms linear ensembles in all settings, especially in attribute generalization (IMAGENETBG, LIVING17).

![Linear vs Prototypical](../figures/linear_vs_prototype.png)

