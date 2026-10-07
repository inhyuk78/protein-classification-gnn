# protein-classification-gnn

**Does incorporating graph connectivity improve protein classification compared to using node features alone?**

## Purpose
A simple graph neural network was built using PyTorch Geometric on the PROTEINS dataset. The purpose of this project was to get a general understanding of how GNNs work and to familiarise myself with the concepts of nodes vs edges.

## Dataset
[PROTEINS](https://chrsmrrs.github.io/datasets/docs/datasets/) (via PyTorch Geometric's 'TUDataset'): 1,113 proteins represented as graphs, where nodes are secondary structure elements and edges connect neighbouring elements. The task is binary classification (enzyme vs non-enzyme). Each node has 3 input features.

The data was split 80/20 (stratified) into 890 training and 223 test proteins.

## Models
Both models are trained with same settings: Adam (lr=0.01), cross-entropy loss, batch size 32, 150 epochs.

- **Node-only NN:** a 3-layer MLP (3 → 16 → 32 → 64) applied to each node independently, followed by sum pooling and a linear classifier. It ignores edges entirely.
- **GCN (nodes + edges):** three `GCNConv` layers (3 → 16 → 32 → 64), followed by sum pooling, and a linear classifier. Message passing lets each node learn from its neighbours.


## Results
| Model | Accuracy | F1 Score |
|---|---|---|
| Node-only NN | 0.735 | 0.684 |
| GCN (nodes + edges) | 0.717 | 0.644 |

*Single run, stratified 80/20 split, 223 test proteins.*

**Findings**
- In this experiment, adding graph connectivity did **not** improve classification. The node-only model performed slightly better (+1.8 points accuracy, +4.0 points F1).
- However, the gap is small. A 1.8-point accuracy difference on 223 test proteins corresponds to only about 4 proteins, and re-running the notebook shifts results by a similar amount. Thus, I can only conclude that the GCN showed no benefit here, rather than that the node-only model is better.

**Possible explanations**
- The PROTEINS node features are only 3 values per node, so summing them may already capture most of the useful signal, leaving little for message passing to add.
- With only 1,113 graphs, the extra capacity of the GCN may not translate into improved generalisation.

**Limitations**
- Only one train/test split and one random seed were used, so the variability of the results is unknown.
- No validation set or hyperparameter tuning was used.
