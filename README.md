```mermaid
flowchart LR
  subgraph Inputs
    Xraw[[X_raw (raw)]]:::io
    Xaug[[X_aug (aug)]]:::io
  end

  subgraph Model[Encoder–Classifier (shared encoder, locked head)]
    ENC[Encoder f_θ]:::block
    CLF[Classifier (locked)]:::lock
  end

  Xraw --> ENC
  Xaug --> ENC
  ENC --> CLF

  AVG1((AVG)):::avg
  AVG2((AVG)):::avg

  CLF -- logits --> AVG2

  subgraph Proto[Uncertainty-aware Prototypes]
    SUP[Supports (warmup + low-entropy K/cls)]:::block
    PLOG[Prototype logits]:::block
    SUP --> PLOG
  end

  CLF -- features/probs --> PLOG
  PLOG --> EC[Entropy Comparison]:::gate
  AVG2 --> EC

  EC --> SELP[select_pred (for metrics)]:::io

  subgraph EATA[EATA selection & memory]
    MEM[EATAMemory (feats, probs)]:::block
    GATE[EATA sample gate\n(low-entropy + de-redundancy)]:::gate
  end

  CLF --> MEM
  MEM --> GATE
  GATE -. selected indices .-> LOSS

  subgraph Train[Adaptation losses (only BN affine + Conv1d update)]
    CLUS[Augmented Contrastive Clustering]:::block
    LOSS[Loss = Contrastive + λ·Entropy + Fisher/L2-SP]:::block
  end

  EC --> CLUS
  CLUS --> LOSS
  LOSS -. grads .- CLF
  LOSS -. grads .- ENC

  classDef block fill:#f7f7f7,stroke:#111,stroke-width:1.5;
  classDef lock fill:#f7f7f7,stroke:#111,stroke-width:1.5,stroke-dasharray:3 2;
  classDef gate fill:#fff,stroke:#111,stroke-width:1.5,stroke-dasharray:4 3;
  classDef avg fill:#fff,stroke:#111,stroke-width:1.6;
  classDef io fill:#fff,stroke:#111,stroke-width:1.5;
