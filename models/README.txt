Model artifacts are saved here after running the notebook.

Files generated:
- cnn_encoder_best.pt       — Best CNN weights (PyTorch)
- cnn_encoder_traced.pt     — TorchScript model for production serving
- ensemble_classifier.pkl   — XGBoost + Random Forest ensemble
- feature_scaler.pkl        — StandardScaler for feature normalization
- label_encoder.pkl         — LabelEncoder for class names
- metadata.json             — Model config and performance metrics
