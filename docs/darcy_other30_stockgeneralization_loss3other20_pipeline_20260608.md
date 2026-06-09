# Darcy other30 + stockgeneralization + stockloss3other20 pipeline

Updated UTC: 2026-06-08T08:56:29Z

Observed evidence and paths:

- other30 training directory: `/workspace/NeuralOperatorRobustness2/2D_Darcy_FNO2d/saved_models/2D/darcy_flow_other30training_20260608`; final exists: `True`; best exists: `True`.
- stockgeneralization pool: `/workspace/NeuralOperatorRobustness2/generalization_datasets_darcy_stockgeneralization_pool_20260608`; manifest exists: `True`.
- stockgeneralization screen: `/workspace/NeuralOperatorRobustness2/forensics/darcy_stockgeneralization_gradient_screen_20260608`; summary exists: `True`.
- selected stockgeneralization root: `/workspace/NeuralOperatorRobustness2/generalization_datasets_darcy_stockgeneralization_20260608`; manifest exists: `True`.
- stockloss3other20 run: `/workspace/NeuralOperatorRobustness2/adversarial_training_runs/darcy_stockloss3other20training_20260608`; summary exists: `True`.

Inference:

- This pipeline treats `other30training` as a 30-epoch Darcy FNO baseline trained on the local Darcy screen train/test tensors.
- It treats `stockgeneralization` as a newly generated Darcy loss-drop-style candidate pool screened with the other30 checkpoint and selected to 50 datasets.
- It treats `stockloss3other20training` as 20 epochs of Darcy solver-label loss3 adversarial self-training initialized from the other30 checkpoint.
