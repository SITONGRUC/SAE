# SAE

Reading notes on *Exploratory Causal Inference in SAEnce* (ICLR 2026), written in my own
words as I work through the paper.


## Related work

- Sparse autoencoders find highly interpretable features in language models.
  - https://github.com/fabiopapais/sparse-interpretability — a student replication, used as
    a reading reference only, not cloned in.
  - https://github.com/HoagyC/sparse_coding — the authors' own implementation of the SAE
    paper above (not of the ECI paper, whose only released code is the NES reference in
    the ICLR supplementary, kept in [`original paper/`](original%20paper/)). Was vendored
    locally as a reading reference; removed once the replication had its own SAE.


## Part 1: the SAE

1. **T is known.** We have two groups, one gets the treatment and one doesn't. Randomized.
2. **Y is unknown.** We don't know what the outcome is, and the outcome data is something
   like multimedia or other unstructured data.
3. **What we can do** — this is the first part of the paper:
   - Convert the outcome data into dense representations via a foundation model. I do
     textual analysis, so if the data is text the output would be embeddings.
   - The problem is that the vector is dense, and each node is not interpretable.
   - So we use a sparse autoencoder to make the nodes sparse, so that each node carries
     less information load. That way each node can be interpretable to some degree. For
     example, the dense vector h is ~900 dimensional, z is ~9000 dimensional, and ĥ is
     ~900 dimensional again. We try to (1) minimize ĥ − h, and (2) only allow ~20 of the
     9000 nodes to be non-zero. This makes sure those 20 nodes actually carry information,
     given that ĥ has to be reconstructed from them.
     - The width (768 → 9,216) gives enough axes that concepts don't have to share.
     - Top-k (only 20 fire per input, for example) forces each active code to carry real
       structure, pushing toward one concept per node.
   - Then we run a t-test 9,216 times — on all the codes, not just the 20. The 20 is per
     input, and which 20 fire changes every time. Each test compares treated vs. control
     means, and we keep the ones that shifted.

That's the first half. But naive testing fails, and not because of multiple comparisons —
Bonferroni handles that.


## Part 2: NES (Neural Effect Search)

**The problem is node information leakage:** each node can end up loaded with some of the
information or feature that belongs to another concept.

The real issue is that SAE codes leak. A code that mostly encodes some other concept still
has a **genuinely non-zero** τ. Corrections only guard against false positives, and these
aren't false.

Meanwhile the t-stat grows like √n while the Bonferroni cutoff grows like √(2 log m) — it
scales with the number of tests, not with the sample size. So with enough power every leaky
code eventually rejects. **More data makes the result less interpretable.** That is the
paradox of exploratory causal inference, and NES is the answer to it.

### My understanding of the algorithm

1. Run the test Y(node) ~ T on the nodes. Keep the ones that come out significant — the
   cutoff is α/m, so the multiple-testing correction still sits inside each round — and
   among those, pick up the one with the biggest coefficient. Note these are two separate
   criteria applied in order: significance first, effect size second.
2. Divide the dataset into two parts by the value of that node — one part where it is low,
   one where it is high — and run the tests again inside each part. This is like giving the
   condition of node 1. Then take a weighted average over the two groups to get the result.
3. Do this recursively, until no node comes out significant any more. That is when it stops,
   and the nodes picked up along the way are the answer. This is how it solves the node
   information leakage problem.

**The order matters — it has to be one at a time.** After you have tested the first one and
begin the second round holding node 1, you find that a lot of them are insignificant, since
a lot of them were only there because of the info leakage.


## Experiment 1 (what I reproduce)

### Data generating process

Treatment ~ B(0.5), Y1 = eyeglasses, Y2 = wearing hat, W = smiling.

Two paths:

- for Y2: T → Y2
- for Y1: T → Y1, and W → Y1

The setting of the paper is that

```
P(Y2 | T=1) = 0.5 + ATE/2      P(Y2 | T=0) = 0.5 − ATE/2
```

which is the showing up of T increasing P(Y2) at the ATE level, and

```
P(Y1 | T=1, W=1) = 0.5 + ATE/2      P(Y1 | T=0, W=1) = 0.5 − ATE/2
P(Y1 | T=1, W=0) = 0.2 + ATE        P(Y1 | T=0, W=0) = 0.2
```

Based on the picture, the paper simulates a standard RCT, with W as an exogenous cause.

### FM features

Use SigLIP to convert the picture to vectors, taking the last output dim (d=768), 196
patches per image.

### SAE training / val / test

Following the official CelebA split: the SAE is trained on the val set, and the test set is
used for the causal experiment. The train set is left unused.

Aggregate patchwise by mean pooling, giving 9,216 per image — 9,216 being the number of
nodes the paper tests on.

It looks like there are two averaging steps, one on the FM side and one after the SAE. I am
not fully sure, but that is the reading where both sentences in the paper do something.

### Evaluation

1. Use the F-score to **label** the nodes. Since we do not know which nodes are for Y1 or
   Y2, we need to work that out.
2. Treat each node as a binary output — a node counts as predicting the attribute when it is
   active on that image. From this matrix we get which nodes best represent Y1 and Y2, so we
   label the nodes.
3. Once we know what the nodes are for, we run NES on the table, and NES should tell us that
   the nodes for Y1 and Y2 are the significant ones.
