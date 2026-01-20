Dataset: CIFAR-10
Model: ResNet-18
Pathological label-skew: 2 classes/client
Topology: paper uses |E|=5 edges, |K|=50 clients 
Constraints: d=5, B=3
Horizon: T=50
Metric threshold: TTR α=0.20 
ICFEC_Research_Paper-17
Also include how to produce:
training_metrics.csv (global/seen/void acc curves)
cp_metrics.csv (ACE, DMR, bytes delivered, etc.)
summary JSON with TTR/VoidAUC/regret
