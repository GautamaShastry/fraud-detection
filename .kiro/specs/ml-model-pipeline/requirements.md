# Requirements Document: ML Model Training and Evaluation Pipeline

## Introduction

This document specifies the requirements for the machine learning model training and evaluation pipeline that supports the Marketplace Fraud Detection Platform. The pipeline consumes the IEEE-CIS transaction dataset, performs feature engineering, trains gradient boosted tree models and logistic regression baselines, evaluates model performance against fraud detection metrics, and produces versioned artifacts for deployment to CPU-based inference containers. The pipeline enforces strict data hygiene (no train/test leakage, chronological splits), supports reproducible training via SageMaker jobs, and includes fallback mechanisms for partial feature availability.

## Glossary

- **Pipeline**: The complete ML model training and evaluation system from data audit through artifact promotion
- **IEEE_Dataset**: The IEEE-CIS Fraud Detection dataset containing train_transaction.csv, train_identity.csv, test_transaction.csv, and test_identity.csv files
- **Feature_Contract**: The versioned schema defining feature names, types, missingness semantics, and offline/online parity requirements
- **Training_Job**: A SageMaker training job instance that consumes a specific data split and hyperparameter configuration
- **Model_Artifact**: A trained model serialization file with associated manifest containing SHA-256 hash, feature schema version, and training metadata
- **Fraud_Scorer**: The inference component that loads Model_Artifact and produces risk scores for transactions
- **Baseline_Model**: A logistic regression model serving as performance floor and fallback
- **GBT_Model**: Gradient boosted tree model (LightGBM or XGBoost)
- **Synthetic_Simulator**: A seeded random data generator producing realistic transaction scenarios for testing without label leakage
- **FPR**: False Positive Rate, the fraction of legitimate transactions incorrectly flagged as fraud
- **Dataset_Card**: A document describing IEEE_Dataset schema, feature availability, label distribution, and temporal characteristics
- **Artifact_Manifest**: A JSON file containing Model_Artifact SHA-256 hash, Feature_Contract version, training timestamp, and hyperparameters
- **Promotion**: The manual approval process for moving Model_Artifact from training to production deployment
- **Fallback_Scorer**: A degraded inference mode that operates with partial feature availability
- **Leakage_Audit**: Verification that features do not contain information from the future relative to prediction time
- **Chronological_Split**: Dataset partition where training precedes validation which precedes test in time order
- **Calibration**: The alignment between predicted fraud probabilities and observed fraud rates
- **Risk_Score**: The final 0-100 integer output from Fraud_Scorer, transformed from model probability via ranking

## Requirements

### Requirement 1: IEEE Dataset Audit and Documentation

**User Story:** As a data scientist, I want comprehensive documentation of the IEEE-CIS dataset structure and characteristics, so that I understand feature availability, label distribution, and temporal ordering before training models.

#### Acceptance Criteria

1. THE Pipeline SHALL produce a Dataset_Card documenting all columns in train_transaction.csv and train_identity.csv
2. THE Dataset_Card SHALL specify the join key (TransactionID) between transaction and identity tables
3. THE Dataset_Card SHALL report the isFraud label distribution (fraud rate percentage)
4. THE Dataset_Card SHALL document the temporal ordering column (TransactionDT) and its semantics
5. THE Dataset_Card SHALL report feature missingness rates for all columns
6. THE Dataset_Card SHALL identify categorical versus numeric feature types
7. WHEN the audit completes, THE Pipeline SHALL verify that TransactionID values in train_identity.csv are a subset of those in train_transaction.csv

### Requirement 2: Temporal Data Leakage Prevention

**User Story:** As a data scientist, I want guarantees that features do not leak future information, so that model performance estimates are realistic and deployment performance is not degraded.

#### Acceptance Criteria

1. THE Leakage_Audit SHALL verify that all features are available at inference time before the transaction outcome is known
2. THE Leakage_Audit SHALL flag any features derived from future transactions
3. THE Leakage_Audit SHALL verify that aggregation features only use data strictly prior to the current transaction timestamp
4. WHEN a feature fails Leakage_Audit, THE Pipeline SHALL exclude the feature from Feature_Contract
5. THE Pipeline SHALL document all excluded features with justification in the Dataset_Card

### Requirement 3: Chronological Train/Validation/Test Split

**User Story:** As a data scientist, I want chronologically ordered dataset splits, so that model evaluation reflects realistic deployment conditions where the model predicts future transactions.

#### Acceptance Criteria

1. THE Pipeline SHALL sort IEEE_Dataset by TransactionDT before splitting
2. THE Pipeline SHALL partition sorted data into training (60%), validation (20%), and test (20%) sets
3. THE Chronological_Split SHALL ensure no training transaction has TransactionDT greater than any validation transaction
4. THE Chronological_Split SHALL ensure no validation transaction has TransactionDT greater than any test transaction
5. THE Pipeline SHALL persist split transaction IDs to enable reproducible experiments
6. THE Pipeline SHALL verify split sizes and temporal ordering before training begins

### Requirement 4: Feature Contract Definition and Versioning

**User Story:** As a data scientist, I want a versioned feature schema, so that offline training and online inference use consistent feature definitions and transformations.

#### Acceptance Criteria

1. THE Pipeline SHALL produce a Feature_Contract specifying feature names, types, and transformations
2. THE Feature_Contract SHALL include a semantic version number (e.g., 1.0.0)
3. THE Feature_Contract SHALL document missingness handling strategy for each feature (drop, impute, indicator)
4. THE Feature_Contract SHALL specify categorical encoding schemes (one-hot, target encoding, hash)
5. THE Feature_Contract SHALL document numeric feature scaling (standardization, min-max, none)
6. WHEN Feature_Contract is modified, THE Pipeline SHALL increment the version number
7. THE Feature_Contract SHALL specify which features are required versus optional for inference

### Requirement 5: Offline/Online Feature Parity Validation

**User Story:** As a data scientist, I want validation that offline feature engineering matches online inference transformations, so that training/serving skew does not degrade production performance.

#### Acceptance Criteria

1. THE Pipeline SHALL implement feature transformations identically for training and inference code paths
2. THE Pipeline SHALL produce a parity test suite comparing offline and online feature outputs
3. WHEN a sample transaction is processed through both paths, THE Pipeline SHALL verify feature values match within 1e-6 tolerance
4. THE parity test SHALL cover all categorical encodings, numeric transformations, and missingness handling
5. WHEN parity test fails, THE Pipeline SHALL report which features diverge and by how much

### Requirement 6: Synthetic Data Generation for Testing

**User Story:** As a data scientist, I want deterministic synthetic transaction data, so that I can test the pipeline without exposing production patterns or leaking labels into inference testing.

#### Acceptance Criteria

1. THE Synthetic_Simulator SHALL accept a random seed for reproducible generation
2. THE Synthetic_Simulator SHALL generate transactions with realistic feature distributions matching IEEE_Dataset statistics
3. THE Synthetic_Simulator SHALL generate transactions without using fraud labels to determine feature values
4. THE Synthetic_Simulator SHALL support configurable fraud scenarios (stolen card, account takeover, friendly fraud)
5. THE Synthetic_Simulator SHALL produce transactions conforming to Feature_Contract
6. WHEN the same seed is used twice, THE Synthetic_Simulator SHALL produce identical transaction sequences

### Requirement 7: Baseline Logistic Regression Model

**User Story:** As a data scientist, I want a logistic regression baseline model, so that I have a performance floor and a simple fallback for deployment.

#### Acceptance Criteria

1. THE Pipeline SHALL train a Baseline_Model using logistic regression on the training split
2. THE Baseline_Model SHALL use L2 regularization with cross-validated hyperparameter tuning
3. THE Baseline_Model SHALL train on the same Feature_Contract as GBT_Model
4. THE Pipeline SHALL serialize Baseline_Model with associated Artifact_Manifest
5. THE Pipeline SHALL evaluate Baseline_Model on validation and test splits
6. THE Pipeline SHALL report Baseline_Model performance metrics for comparison with GBT_Model

### Requirement 8: Gradient Boosted Tree Model Training

**User Story:** As a data scientist, I want to train gradient boosted tree models, so that I can achieve high fraud detection recall while controlling false positive rates.

#### Acceptance Criteria

1. THE Pipeline SHALL train a GBT_Model using LightGBM or XGBoost
2. THE GBT_Model SHALL use binary cross-entropy loss for fraud classification
3. THE Pipeline SHALL tune GBT_Model hyperparameters (learning rate, max depth, num leaves, min child samples, subsample rate) via validation set performance
4. THE Pipeline SHALL use early stopping based on validation AUC to prevent overfitting
5. THE GBT_Model SHALL train using class weights to handle label imbalance
6. THE Pipeline SHALL serialize GBT_Model with associated Artifact_Manifest

### Requirement 9: Minority Class Evaluation Metrics

**User Story:** As a data scientist, I want evaluation metrics focused on minority class performance, so that I understand fraud detection capability despite severe class imbalance.

#### Acceptance Criteria

1. THE Pipeline SHALL report fraud recall (true positive rate on fraud class) for all models
2. THE Pipeline SHALL report precision (positive predictive value) for all models
3. THE Pipeline SHALL compute FPR (false positive rate) for all models
4. THE Pipeline SHALL compute ROC-AUC (area under receiver operating characteristic curve) for all models
5. THE Pipeline SHALL compute Precision-Recall AUC for all models
6. THE Pipeline SHALL report metrics separately for training, validation, and test splits
7. THE Pipeline SHALL stratify metrics by transaction amount quintiles to detect bias

### Requirement 10: Threshold Selection with FPR Constraints

**User Story:** As a data scientist, I want to select decision thresholds that meet business FPR constraints, so that fraud detection does not generate excessive false alarms.

#### Acceptance Criteria

1. THE Pipeline SHALL compute fraud recall at FPR thresholds of 0.1%, 0.5%, 1.0%, and 2.0%
2. THE Pipeline SHALL identify the probability threshold achieving target FPR on validation set
3. THE Pipeline SHALL report fraud recall at the selected threshold on test set
4. THE Pipeline SHALL produce a threshold-recall-FPR curve for stakeholder review
5. WHEN no threshold achieves target FPR, THE Pipeline SHALL report the minimum achievable FPR
6. THE Pipeline SHALL document the selected threshold in Artifact_Manifest

### Requirement 11: Model Calibration Assessment

**User Story:** As a data scientist, I want to assess probability calibration, so that I know whether model confidence scores are reliable for risk-based decisioning.

#### Acceptance Criteria

1. THE Pipeline SHALL bin test set predictions into 10 equal-frequency buckets
2. THE Pipeline SHALL compute observed fraud rate within each bucket
3. THE Pipeline SHALL compute expected calibration error (mean absolute difference between predicted and observed rates)
4. THE Pipeline SHALL produce a calibration plot showing predicted vs observed fraud rates
5. THE Pipeline SHALL flag the model as poorly calibrated if ECE exceeds 0.05

### Requirement 12: Risk Score Transformation

**User Story:** As a data scientist, I want to transform model probabilities into 0-100 risk scores, so that downstream systems consume ranking scores rather than raw probabilities.

#### Acceptance Criteria

1. THE Pipeline SHALL rank all validation transactions by predicted fraud probability
2. THE Pipeline SHALL assign Risk_Score from 0-100 based on percentile rank
3. THE Risk_Score transformation SHALL map lowest probability to 0 and highest to 100
4. THE Pipeline SHALL document the probability-to-Risk_Score mapping in Artifact_Manifest
5. THE Fraud_Scorer SHALL apply the documented transformation at inference time

### Requirement 13: SageMaker Training Job Configuration

**User Story:** As a data scientist, I want reproducible SageMaker training job scripts, so that models can be retrained consistently with version control and audit trails.

#### Acceptance Criteria

1. THE Pipeline SHALL generate a SageMaker training job specification including instance type, container image, hyperparameters, and input data S3 paths
2. THE training job specification SHALL include a unique job name with timestamp and git commit hash
3. THE training job specification SHALL specify output S3 path for Model_Artifact
4. THE training job specification SHALL configure instance type to optimize cost (ml.m5.xlarge or ml.m5.2xlarge)
5. THE training job specification SHALL estimate job cost before execution
6. THE Pipeline SHALL log all training job parameters for reproducibility

### Requirement 14: Training Cost Approval Gate

**User Story:** As a data scientist, I want explicit cost approval before executing SageMaker jobs, so that training expenses are controlled and authorized.

#### Acceptance Criteria

1. WHEN a training job is configured, THE Pipeline SHALL estimate total cost based on instance type and expected duration
2. THE Pipeline SHALL require manual approval before submitting the training job to SageMaker
3. THE Pipeline SHALL display estimated cost and job parameters during approval
4. WHEN approval is denied, THE Pipeline SHALL abort job submission without incurring charges
5. WHEN approval is granted, THE Pipeline SHALL submit the job and log the approval timestamp

### Requirement 15: Model Artifact SHA-256 Verification

**User Story:** As a data scientist, I want cryptographic verification of model artifacts, so that I can detect corruption or tampering during storage and transfer.

#### Acceptance Criteria

1. THE Pipeline SHALL compute SHA-256 hash of Model_Artifact file immediately after serialization
2. THE Pipeline SHALL store the SHA-256 hash in Artifact_Manifest
3. THE Fraud_Scorer SHALL compute SHA-256 hash of Model_Artifact file at load time
4. WHEN the computed hash does not match Artifact_Manifest, THE Fraud_Scorer SHALL refuse to load the model
5. THE Fraud_Scorer SHALL log hash mismatches as critical security events

### Requirement 16: Artifact Manifest Contents

**User Story:** As a data scientist, I want comprehensive metadata in model manifests, so that I can trace model provenance and ensure compatibility with inference systems.

#### Acceptance Criteria

1. THE Artifact_Manifest SHALL include Model_Artifact filename and SHA-256 hash
2. THE Artifact_Manifest SHALL include Feature_Contract version number
3. THE Artifact_Manifest SHALL include training timestamp (ISO 8601 format)
4. THE Artifact_Manifest SHALL include git commit hash of training code
5. THE Artifact_Manifest SHALL include hyperparameter values used for training
6. THE Artifact_Manifest SHALL include validation metrics (recall, FPR, AUC)
7. THE Artifact_Manifest SHALL include the selected decision threshold and corresponding Risk_Score mapping
8. THE Artifact_Manifest SHALL include SageMaker training job name and S3 paths

### Requirement 17: Model Artifact Promotion Process

**User Story:** As a data scientist, I want manual approval for model promotion, so that only validated models are deployed to production.

#### Acceptance Criteria

1. THE Pipeline SHALL require manual approval before promoting Model_Artifact from training to production S3 bucket
2. THE Promotion process SHALL display model metrics, Feature_Contract version, and training metadata
3. WHEN promotion is approved, THE Pipeline SHALL copy Model_Artifact and Artifact_Manifest to production S3 path
4. WHEN promotion is denied, THE Pipeline SHALL archive Model_Artifact to a staging S3 path
5. THE Pipeline SHALL log all promotion decisions with approver identity and timestamp

### Requirement 18: Container Model Loading and Validation

**User Story:** As a data scientist, I want the inference container to validate models at startup, so that misconfigured deployments are detected before serving traffic.

#### Acceptance Criteria

1. WHEN Fraud_Scorer starts, THE container SHALL load Model_Artifact from local filesystem
2. THE Fraud_Scorer SHALL verify Model_Artifact SHA-256 hash against Artifact_Manifest
3. THE Fraud_Scorer SHALL verify Feature_Contract version compatibility
4. THE Fraud_Scorer SHALL perform a warmup inference on a synthetic transaction
5. WHEN validation fails, THE Fraud_Scorer SHALL exit with error code and descriptive message
6. WHEN validation succeeds, THE Fraud_Scorer SHALL log model version, Feature_Contract version, and startup timestamp

### Requirement 19: Feature Contract Version Compatibility

**User Story:** As a data scientist, I want version compatibility checks, so that models are not deployed with mismatched feature schemas.

#### Acceptance Criteria

1. THE Fraud_Scorer SHALL read Feature_Contract version from Artifact_Manifest
2. THE Fraud_Scorer SHALL read Feature_Contract version from deployed feature engineering code
3. WHEN major versions differ, THE Fraud_Scorer SHALL refuse to load Model_Artifact
4. WHEN minor versions differ, THE Fraud_Scorer SHALL log a compatibility warning but proceed
5. WHEN patch versions differ, THE Fraud_Scorer SHALL load Model_Artifact without warning

### Requirement 20: Fallback Model for Partial Features

**User Story:** As a data scientist, I want a fallback scoring mode for transactions with missing optional features, so that the system degrades gracefully rather than rejecting transactions.

#### Acceptance Criteria

1. THE Fallback_Scorer SHALL train on a reduced feature set containing only required features from Feature_Contract
2. THE Fallback_Scorer SHALL achieve at least 50% fraud recall at 1.0% FPR on validation set
3. WHEN a transaction is missing optional features, THE Fraud_Scorer SHALL invoke Fallback_Scorer
4. THE Fallback_Scorer SHALL produce Risk_Score using the same 0-100 transformation
5. THE Fraud_Scorer SHALL log fallback invocations for monitoring
6. THE Fallback_Scorer SHALL have latency budget of ≤110ms matching primary model

### Requirement 21: Reproducible Training Pipeline

**User Story:** As a data scientist, I want deterministic pipeline execution, so that retraining produces identical results for audit and debugging.

#### Acceptance Criteria

1. THE Pipeline SHALL accept a master random seed controlling all stochastic operations
2. THE Pipeline SHALL set random seeds for data splitting, feature sampling, and model training
3. WHEN the same seed and code version are used, THE Pipeline SHALL produce Model_Artifact with identical SHA-256 hash
4. THE Pipeline SHALL log all random seeds in Artifact_Manifest
5. THE Pipeline SHALL pin dependency versions (Python packages, LightGBM/XGBoost) in training environment

### Requirement 22: Fraud Detection Performance Targets

**User Story:** As a product manager, I want models to meet minimum fraud detection performance, so that the service provides value to customers.

#### Acceptance Criteria

1. THE GBT_Model SHALL achieve at least 60% fraud recall at 0.5% FPR on test set (initial target)
2. THE GBT_Model SHALL achieve at least 80% fraud recall at 0.5% FPR on test set (stretch target)
3. WHEN GBT_Model fails to meet initial target, THE Pipeline SHALL report the gap and recommend next steps
4. THE Baseline_Model SHALL achieve at least 40% fraud recall at 0.5% FPR on test set
5. THE Pipeline SHALL verify that GBT_Model outperforms Baseline_Model on all metrics

### Requirement 23: Inference Latency Budget Compliance

**User Story:** As a platform engineer, I want models optimized for CPU inference, so that inference latency meets the ≤110ms budget without expensive GPU infrastructure.

#### Acceptance Criteria

1. THE GBT_Model SHALL have at most 500 trees to control inference latency
2. THE GBT_Model SHALL have at most 31 leaf nodes per tree to control inference latency
3. THE Pipeline SHALL benchmark Model_Artifact inference time on target CPU (AWS m5.2xlarge)
4. THE Pipeline SHALL report p50, p95, and p99 inference latencies
5. WHEN p95 latency exceeds 110ms, THE Pipeline SHALL reject Model_Artifact and recommend hyperparameter adjustments

### Requirement 24: Training Performance Reporting

**User Story:** As a data scientist, I want comprehensive performance reports, so that I can diagnose model behavior and communicate results to stakeholders.

#### Acceptance Criteria

1. THE Pipeline SHALL generate a performance report including all metrics from Requirement 9
2. THE performance report SHALL include confusion matrices for each split at the selected threshold
3. THE performance report SHALL include feature importance rankings for GBT_Model
4. THE performance report SHALL include calibration plots and ECE values
5. THE performance report SHALL compare GBT_Model and Baseline_Model side-by-side
6. THE performance report SHALL include latency benchmarks
7. THE Pipeline SHALL persist the performance report to S3 alongside Model_Artifact

### Requirement 25: Synthetic Performance Disclosure

**User Story:** As a data scientist, I want clear labeling of synthetic evaluation results, so that synthetic performance is never misrepresented as real-world commercial results.

#### Acceptance Criteria

1. WHEN reporting performance on Synthetic_Simulator data, THE Pipeline SHALL prefix all metrics with "Synthetic:"
2. THE performance report SHALL include a disclaimer: "Performance on synthetic data does not represent real-world commercial results"
3. THE Pipeline SHALL log the data source (IEEE_Dataset vs Synthetic_Simulator) for all evaluation runs
4. THE Pipeline SHALL reject any attempt to promote models trained solely on synthetic data

### Requirement 26: Feature Missingness Handling

**User Story:** As a data scientist, I want explicit handling strategies for missing features, so that models are robust to incomplete transaction data.

#### Acceptance Criteria

1. THE Feature_Contract SHALL specify missingness handling for each feature: impute, drop, or indicator
2. WHEN a feature uses imputation, THE Feature_Contract SHALL specify the imputation method (mean, median, mode, constant)
3. WHEN a feature uses indicator strategy, THE Pipeline SHALL create a binary is_missing feature
4. THE Pipeline SHALL apply identical missingness handling in training and inference
5. THE Pipeline SHALL report missingness rates in Dataset_Card and performance report

### Requirement 27: Hyperparameter Tuning Strategy

**User Story:** As a data scientist, I want systematic hyperparameter tuning, so that models are optimized for fraud detection performance.

#### Acceptance Criteria

1. THE Pipeline SHALL define hyperparameter search spaces for GBT_Model (learning_rate: [0.01, 0.1], max_depth: [3, 7], num_leaves: [7, 31])
2. THE Pipeline SHALL use validation set AUC as the optimization objective
3. THE Pipeline SHALL explore at least 20 hyperparameter configurations
4. THE Pipeline SHALL use random search or Bayesian optimization for efficient exploration
5. THE Pipeline SHALL report the best hyperparameter configuration in Artifact_Manifest
6. THE Pipeline SHALL persist all trial results for post-hoc analysis

### Requirement 28: Class Imbalance Mitigation

**User Story:** As a data scientist, I want techniques to handle severe class imbalance, so that models learn to detect rare fraud events.

#### Acceptance Criteria

1. THE Pipeline SHALL compute class weights inversely proportional to label frequencies
2. THE GBT_Model SHALL use computed class weights during training
3. THE Pipeline SHALL support SMOTE or ADASYN oversampling as optional preprocessing
4. WHEN oversampling is used, THE Pipeline SHALL apply it only to training split to avoid validation contamination
5. THE Pipeline SHALL compare model performance with and without class balancing techniques

### Requirement 29: Model Serialization Format

**User Story:** As a data scientist, I want standardized model serialization, so that models are portable across training and inference environments.

#### Acceptance Criteria

1. THE Pipeline SHALL serialize GBT_Model using native format (LightGBM .txt or XGBoost .json)
2. THE Pipeline SHALL serialize Baseline_Model using pickle or joblib format
3. THE Pipeline SHALL include Python version and library versions in Artifact_Manifest
4. THE Fraud_Scorer SHALL verify library version compatibility at load time
5. THE Pipeline SHALL support ONNX export as optional format for interoperability

### Requirement 30: Training Data Provenance

**User Story:** As a data scientist, I want complete provenance tracking, so that I can trace which data was used for each model version.

#### Acceptance Criteria

1. THE Artifact_Manifest SHALL include S3 paths for training, validation, and test data
2. THE Artifact_Manifest SHALL include SHA-256 hashes of training data files
3. THE Artifact_Manifest SHALL include IEEE_Dataset version or snapshot date
4. THE Artifact_Manifest SHALL include Chronological_Split parameters (percentages, random seed)
5. THE Pipeline SHALL verify training data integrity by comparing hashes before training begins

### Requirement 31: Pretty Printer for Feature Contracts

**User Story:** As a data scientist, I want human-readable feature contract formatting, so that I can review and validate feature definitions easily.

#### Acceptance Criteria

1. THE Pipeline SHALL implement a pretty printer for Feature_Contract JSON
2. THE pretty printer SHALL format Feature_Contract with indentation and sorted keys
3. THE pretty printer SHALL include inline comments for feature semantics
4. THE pretty printer SHALL validate Feature_Contract schema before formatting
5. THE Pipeline SHALL use the pretty printer to persist Feature_Contract to version control

### Requirement 32: Feature Contract Round-Trip Validation

**User Story:** As a data scientist, I want validation that feature contracts serialize and deserialize correctly, so that no information is lost in persistence.

#### Acceptance Criteria

1. FOR ALL valid Feature_Contract objects, parsing the JSON representation then serializing SHALL produce an equivalent object
2. THE Pipeline SHALL verify Feature_Contract round-trip property during initialization
3. WHEN round-trip validation fails, THE Pipeline SHALL report which fields were corrupted
4. THE Pipeline SHALL test round-trip property with 100 randomly generated feature configurations
5. THE round-trip test SHALL verify semantic equivalence including type, transformation, missingness strategy, and version
