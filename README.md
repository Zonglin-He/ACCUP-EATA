```mermaid
flowchart LR
  subgraph Inputs
    Xraw([X_raw raw])
    Xaug([X_aug aug])
  end

  subgraph Model[Encoder-Classifier shared encoder, locked head]
    ENC[Encoder f_theta]
    CLF[Classifier locked]
  end

  Xraw --> ENC
  Xaug --> ENC
  ENC --> CLF

  AVG2((AVG))
  CLF -->|logits| AVG2

  subgraph Proto[Uncertainty-aware Prototypes]
    SUP[Supports warmup + low-entropy K per class]
    PLOG[Prototype logits]
    SUP --> PLOG
  end

  CLF -->|features/probs| PLOG
  PLOG --> EC[Entropy Comparison]
  AVG2 --> EC

  EC --> SELP([select_pred for metrics])

  subgraph EATA[EATA selection and memory]
    MEM[EATAMemory feats probs]
    GATE[EATA sample gate low-entropy and de-redundancy]
  end

  CLF --> MEM
  MEM --> GATE
  GATE -. selected indices .-> LOSS

  subgraph Train[Adaptation losses only BN affine + Conv1d update]
    CLUS[Augmented Contrastive Clustering]
    LOSS[Loss = Contrastive + lambda Entropy + Fisher/L2-SP]
  end

  EC --> CLUS
  CLUS --> LOSS
  LOSS -. grads .- CLF
  LOSS -. grads .- ENC
