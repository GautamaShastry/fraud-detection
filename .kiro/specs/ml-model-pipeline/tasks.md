# Implementation Plan: ML Model Training and Evaluation Pipeline

## Overview

This implementation plan translates the requirements and technical design into actionable coding tasks for building the ML Model Training and Evaluation Pipeline. The pipeline processes IEEE-CIS transaction data, trains fraud detection models (baseline logistic regression and LightGBM GBT), evaluates performance, and produces versioned artifacts for CPU-based inference.

The implementation follows a bottom-up approach: foundational components (data audit, feature contracts) → training pipeline → evaluation → artifact management → SageMaker integration → deployment → testing.

## Implementation Language

**Python** will be used for all implementation tasks, with the following key libraries:
- **Data Processing**: pandas, numpy
- **ML Training**: scikit-learn (baseline), LightGBM (GBT)
- **Evaluation**: scikit-learn metrics
- **AWS Integration**: boto3, sagemaker SDK
- **Testing**: pytest, hypothesis (property-based testing)
- **Serialization**: joblib, json, hashlib

## Tasks

- [ ] 1. Setup project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `models/`, `config/`
  - Define `requirements.txt` with pinned versions (pandas, numpy, scikit-learn, lightgbm, boto3, sagemaker, hypothesis, pytest)
  - Create `setup.py` or `pyproject.toml` for package installation
  - Initialize git repository with `.gitignore` for Python, data files, and model artifacts
  - Create README.md with project overview and setup instructions
  - _Requirements: 21.5, 29.3_

- [ ] 2. Implement data management layer
  - [ ] 2.1 Create Dataset Auditor component
    - Implement `DatasetAuditor` class with `audit()` method
    - Load IEEE-CIS train_transaction.csv and train_identity.csv using pandas
    - Verify join integrity: assert all identity TransactionIDs exist in transactions
    - Compute fraud rate: `isFraud` column mean
    - Document temporal column (TransactionDT) and range (min, max)
    - Compute missingness rates for all columns
    - Classify features as CATEGORICAL (object/string, cardinality < 50) or NUMERIC (int/float)
    - Generate `DatasetCard` dataclass with schema, statistics, and temporal info
    - Persist Dataset Card to JSON file
    - _Requirements: 1.1-1.7, 2.1_

  - [ ] 2.2 Write property test for Dataset Auditor
    - **Property 1: Feature Missingness Rate Accuracy**
    - **Validates: Requirements 1.5**
    - Use hypothesis to generate random DataFrames with missing values
    - Verify computed missingness rate equals true fraction within 1e-6 tolerance
    - Test with 100+ random datasets
    - _Requirements: 1.5_

  - [ ] 2.3 Write unit tests for Dataset Auditor
    - Test join integrity verification with mismatched TransactionIDs
    - Test fraud rate calculation with known label distribution
    - Test feature type classification edge cases (empty columns, mixed types)
    - Test temporal range extraction
    - _Requirements: 1.1-1.7_

  - [ ] 2.4 Implement Temporal Leakage Audit
    - Create `audit_for_leakage()` function scanning feature names for leakage keywords
    - Flag features with "future", "daysuntil", "next", "after" in name (case-insensitive)
    - Return tuple: (clean_features, excluded_features)
    - Document excluded features in Dataset Card with justification
    - _Requirements: 2.1-2.5_

  - [ ]* 2.5 Write unit tests for Leakage Audit
    - Test detection of leakage keywords in feature names
    - Test handling of features without leakage indicators
    - Test case-insensitive detection
    - _Requirements: 2.1-2.4_

  - [ ] 2.6 Implement Chronological Splitter
    - Create `ChronologicalSplitter` class with `split()` method
    - Sort data by TransactionDT column
    - Partition into train (60%), validation (20%), test (20%) by index slicing
    - Extract TransactionID lists for each split
    - Verify temporal ordering: train_max_time ≤ val_min_time, val_max_time ≤ test_min_time
    - Return `ChronologicalSplit` dataclass with IDs, percentages, seed, temporal boundaries
    - Persist split IDs to text files: `splits/train_ids.txt`, `splits/val_ids.txt`, `splits/test_ids.txt`
    - _Requirements: 3.1-3.6_

  - [ ]* 2.7 Write property tests for Chronological Splitter
    - **Property 3: Dataset Temporal Sorting**
    - **Validates: Requirements 3.1**
    - **Property 4: Split Size Conformance**
    - **Validates: Requirements 3.2**
    - **Property 5: Chronological Ordering Across Splits**
    - **Validates: Requirements 3.3, 3.4**
    - Use hypothesis to generate random datasets with timestamps
    - Verify temporal sorting: each timestamp >= previous
    - Verify split sizes within ±1% of specified percentages
    - Verify train max ≤ val min, val max ≤ test min
    - Test with 100+ random configurations
    - _Requirements: 3.1-3.4_

  - [ ]* 2.8 Write unit tests for Chronological Splitter
    - Test with known dataset: verify exact split sizes and temporal boundaries
    - Test reproducibility: same seed produces identical split IDs
    - Test edge cases: minimum dataset size, extreme split ratios
    - _Requirements: 3.1-3.6_

- [ ] 3. Checkpoint - Data layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement feature engineering layer
  - [ ] 4.1 Define Feature Contract schema
    - Create `FeatureContract` dataclass with version (str), features (List), metadata
    - Create `FeatureDefinition` dataclass with name, source_column, feature_type, required, missingness_strategy, imputation_method, encoding_scheme, scaling_method, and params
    - Define enums: `FeatureType` (CATEGORICAL, NUMERIC), `MissingnessStrategy` (IMPUTE, DROP, INDICATOR), `ImputationMethod` (MEAN, MEDIAN, MODE, CONSTANT), `EncodingScheme` (TARGET, ONEHOT, HASH, ORDINAL), `ScalingMethod` (STANDARD, MINMAX, NONE)
    - Create `ContractMetadata` dataclass with created_at, git_commit, dataset_version, description
    - Implement JSON serialization/deserialization methods
    - Implement semantic versioning increment methods (major, minor, patch)
    - _Requirements: 4.1-4.7, 31.1-31.5_

  - [ ]* 4.2 Write property test for Feature Contract round-trip
    - **Property 9: Feature Contract Round-Trip Equivalence**
    - **Validates: Requirements 32.1, 32.4, 32.5**
    - Use hypothesis to generate 100+ random feature contracts
    - Serialize to JSON, deserialize, verify semantic equivalence
    - Check version, feature names, types, transformations, missingness, encoding match
    - _Requirements: 32.1-32.5_

  - [ ]* 4.3 Write unit tests for Feature Contract
    - Test version increment logic (major, minor, patch)
    - Test JSON schema validation
    - Test pretty printing with indentation and sorted keys
    - _Requirements: 4.1-4.7, 31.1-31.5_

  - [ ] 4.4 Implement Feature Transformer
    - Create `FeatureTransformer` class with `fit_transform()` and `transform()` methods
    - Implement `_handle_missingness()`: apply IMPUTE (mean/median/mode/constant), DROP, or INDICATOR strategies
    - Implement `_encode_categorical()`: dispatch to target encoding, one-hot, hash, or ordinal
    - Implement `_target_encode()`: compute smoothed target encoding on training data, store global mean and category maps
    - Implement `_onehot_encode()`: use scikit-learn OneHotEncoder with handle_unknown='ignore'
    - Implement `_scale_numeric()`: apply standardization (z-score) or min-max scaling
    - Store fitted parameters (means, stds, encoding maps) in `fitted_params` dict
    - In `transform()`, use fitted params without refitting
    - Handle missing optional features: use default values, log warnings
    - _Requirements: 4.1-4.7, 5.1-5.2, 26.1-26.5_

  - [ ]* 4.5 Write property test for feature transformation parity
    - **Property 6: Offline/Online Feature Transformation Parity**
    - **Validates: Requirements 5.1, 5.3**
    - Use hypothesis to generate random transactions
    - Fit transformer on training data
    - Transform same transaction through offline (fit_transform) and online (transform) paths
    - Verify feature values differ by at most 1e-6 in any dimension
    - Test with 100+ random transactions
    - _Requirements: 5.1, 5.3_

  - [ ]* 4.6 Write unit tests for Feature Transformer
    - Test missingness handling for each strategy (impute, drop, indicator)
    - Test categorical encoding (target, one-hot, hash)
    - Test numeric scaling (standardization, min-max, none)
    - Test handling of missing optional features at inference time
    - Test fitted parameter persistence and loading
    - _Requirements: 5.1-5.5, 26.1-26.5_

  - [ ] 4.7 Implement Parity Validator
    - Create `ParityValidator` class with `validate()` method
    - Accept offline and online transformers, sample transactions, tolerance (1e-6)
    - Transform transactions through both paths
    - Compute max absolute difference for each feature
    - Report divergent features exceeding tolerance
    - Return `ParityReport` dataclass with pass/fail status and divergence details
    - _Requirements: 5.1-5.5_

  - [ ]* 4.8 Write unit tests for Parity Validator
    - Test with identical transformers (should pass)
    - Test with intentionally divergent transformers (should fail and report features)
    - Test tolerance threshold behavior
    - _Requirements: 5.1-5.5_

- [ ] 5. Checkpoint - Feature engineering complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement Synthetic Simulator for testing
  - [ ] 6.1 Create Synthetic Simulator component
    - Implement `SyntheticSimulator` class with `__init__(seed, dataset_stats, feature_contract)`
    - Initialize numpy random generator with seed: `self.rng = np.random.default_rng(seed)`
    - Implement `generate(n_transactions, fraud_scenarios)` method
    - Generate TransactionID: random integers in [1e6, 1e9]
    - Generate TransactionDT: uniform distribution over dataset temporal range
    - Generate TransactionAmt: log-normal distribution (mean=4.0, sigma=1.5)
    - Generate categorical features: sample from dataset empirical distributions
    - Generate numeric features: sample from Gaussian approximations (mean, std from dataset stats)
    - Implement `_apply_fraud_scenario()`: modify features based on scenario (stolen_card, account_takeover, friendly_fraud) WITHOUT using labels
    - Return DataFrame with synthetic transactions (no labels)
    - _Requirements: 6.1-6.6_

  - [ ] 6.2 Implement Synthetic Simulator determinism verification
    - Add `verify_determinism(n_trials)` method
    - Generate same n_transactions with same seed across multiple trials
    - Compare DataFrames with `df.equals()` for exact match
    - Return True if all trials produce identical results
    - _Requirements: 6.6_

  - [ ]* 6.3 Write property tests for Synthetic Simulator
    - **Property 7: Generated Transactions Conform to Contract**
    - **Validates: Requirements 6.5**
    - **Property 8: Synthetic Simulator Determinism**
    - **Validates: Requirements 6.6**
    - Use hypothesis to generate random seeds and feature contracts
    - Verify generated transactions contain exactly contract features with correct types
    - Verify same seed produces identical DataFrames across multiple calls
    - Test with 100+ random configurations
    - _Requirements: 6.5, 6.6_

  - [ ]* 6.4 Write unit tests for Synthetic Simulator
    - Test feature distribution realism (means, stds close to dataset stats)
    - Test fraud scenario modifications (stolen_card increases amount, account_takeover clusters transactions)
    - Test handling of categorical cardinality
    - Test edge cases (very small n_transactions, empty fraud scenarios)
    - _Requirements: 6.1-6.6, 25.1-25.4_

- [ ] 7. Implement baseline logistic regression model
  - [ ] 7.1 Create Baseline Trainer component
    - Implement `BaselineTrainer` class with `train()` method
    - Compute class weights using `compute_class_weight('balanced')`
    - Define hyperparameter grid: C=[0.001, 0.01, 0.1, 1.0, 10.0], max_iter=[1000]
    - Create `LogisticRegression(penalty='l2', solver='lbfgs', class_weight=class_weight_dict)`
    - Use `GridSearchCV` with cv=5, scoring='roc_auc', n_jobs=-1
    - Fit on training data
    - Evaluate best model on validation set: compute `roc_auc_score()`
    - Return `BaselineModel` dataclass with estimator, best_params, val_auc, class_weights
    - _Requirements: 7.1-7.6_

  - [ ]* 7.2 Write unit tests for Baseline Trainer
    - Test training with small synthetic dataset
    - Verify class weight computation for imbalanced labels
    - Test hyperparameter grid search execution
    - Verify validation AUC calculation
    - _Requirements: 7.1-7.6_

- [ ] 8. Implement LightGBM gradient boosted tree model
  - [ ] 8.1 Create GBT Trainer component
    - Implement `GBTTrainer` class with `train()` method
    - Compute class weight: `scale_pos_weight = legit_count / fraud_count`
    - Define hyperparameter search space using Optuna:
      - learning_rate: [0.01, 0.1] log scale
      - max_depth: [3, 7]
      - num_leaves: [7, 31]
      - min_child_samples: [10, 100]
      - subsample: [0.6, 1.0]
      - colsample_bytree: [0.6, 1.0]
      - n_estimators: 500 (fixed for latency constraint)
    - Create Optuna study with direction='maximize', seed for reproducibility
    - Define objective function: train LGBMClassifier, evaluate validation AUC
    - Use early stopping: `lgb.early_stopping(stopping_rounds=50)`
    - Optimize for n_trials (default 20)
    - Train final model with best params on full training set
    - Return `GBTModel` dataclass with estimator, best_params, val_auc, n_trees, optimization_trials
    - _Requirements: 8.1-8.6, 23.1-23.2, 27.1-27.6_

  - [ ]* 8.2 Write unit tests for GBT Trainer
    - Test training with small synthetic dataset
    - Verify class weight computation
    - Test Optuna hyperparameter search execution
    - Verify early stopping behavior
    - Test final model training with best params
    - _Requirements: 8.1-8.6, 27.1-27.6_

- [ ] 9. Implement fallback model for partial features
  - [ ] 9.1 Create Fallback Trainer component
    - Implement `FallbackTrainer` class with `train()` method
    - Accept required_features list from Feature Contract (features where required=True)
    - Filter training and validation data to required features only
    - Reuse `GBTTrainer` with reduced feature set and fewer trials (n_trials=10)
    - Evaluate on validation set: compute recall at 1% FPR
    - Verify minimum performance: recall_at_1pct_fpr >= 0.5, else raise ValueError
    - Return `FallbackModel` dataclass with estimator, required_features, val_auc, recall_at_1pct_fpr
    - _Requirements: 20.1-20.6_

  - [ ]* 9.2 Write unit tests for Fallback Trainer
    - Test training with reduced feature set
    - Verify minimum performance threshold enforcement
    - Test handling of insufficient recall (should raise ValueError)
    - Test latency constraint (fallback should meet ≤110ms budget)
    - _Requirements: 20.1-20.6_

- [ ] 10. Checkpoint - Model training complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. Implement evaluation framework
  - [ ] 11.1 Create Metrics Calculator component
    - Implement `MetricsCalculator` class with `compute()` method
    - Accept model, splits={'train': (X, y), 'val': (X, y), 'test': (X, y)}, threshold
    - For each split: predict probabilities, apply threshold, compute confusion matrix
    - Calculate metrics: fraud_recall (TPR), precision, FPR, ROC-AUC, PR-AUC, F1 score
    - Implement `_stratify_by_amount()`: compute metrics by transaction amount quintiles
    - Return `MetricsReport` dataclass with per-split metrics, threshold, timestamp
    - _Requirements: 9.1-9.7_

  - [ ]* 11.2 Write unit tests for Metrics Calculator
    - Test metric computation with known confusion matrices
    - Verify stratification by transaction amount quintiles
    - Test handling of edge cases (all fraud, all legitimate, empty splits)
    - Test metrics match scikit-learn implementations
    - _Requirements: 9.1-9.7_

  - [ ] 11.3 Implement Threshold Selector
    - Create `ThresholdSelector` class with `select()` method
    - Accept y_true, y_probs, target_fprs=[0.001, 0.005, 0.01, 0.02]
    - Compute ROC curve: `fprs, tprs, thresholds = roc_curve(y_true, y_probs)`
    - For each target FPR: find threshold minimizing |fpr - target_fpr|
    - Record achieved FPR and fraud recall at selected threshold
    - Generate threshold-recall-FPR curve DataFrame
    - Return `ThresholdSelection` dataclass with selections dict, curve, recommended_threshold (default: 0.5% FPR)
    - _Requirements: 10.1-10.6_

  - [ ]* 11.4 Write unit tests for Threshold Selector
    - Test threshold selection with known probability distributions
    - Verify FPR targeting accuracy
    - Test handling of unachievable FPR targets (report minimum achievable)
    - Test curve generation completeness
    - _Requirements: 10.1-10.6_

  - [ ] 11.5 Implement Calibration Assessor
    - Create `CalibrationAssessor` class with `assess()` method
    - Accept y_true, y_probs, n_bins (default 10)
    - Compute equal-frequency bin edges: `np.percentile(y_probs, linspace(0, 100, n_bins+1))`
    - For each bin: compute predicted probability (mean of y_probs), observed probability (mean of y_true)
    - Compute Expected Calibration Error (ECE): weighted mean absolute difference
    - Flag model as poorly calibrated if ECE > 0.05
    - Generate calibration plot data (predicted vs observed by bin)
    - Return `CalibrationReport` dataclass with ECE, is_well_calibrated, calibration_data, n_bins
    - _Requirements: 11.1-11.5_

  - [ ]* 11.6 Write unit tests for Calibration Assessor
    - Test ECE computation with perfectly calibrated probabilities (ECE ≈ 0)
    - Test ECE computation with miscalibrated probabilities (ECE > 0.05)
    - Test binning strategy (equal-frequency)
    - Test calibration plot data generation
    - _Requirements: 11.1-11.5_

  - [ ] 11.7 Implement Risk Score Transformer
    - Create `RiskScoreTransformer` class with `fit()` and `transform()` methods
    - In `fit()`: compute percentile thresholds from validation probabilities: `np.percentile(val_probs, linspace(0, 100, 101))`
    - Store `self.probability_thresholds` array (101 values for 0-100 risk scores)
    - In `transform()`: map each probability to percentile rank using `np.searchsorted()`
    - Clip risk scores to [0, 100] range
    - Return integer risk scores
    - Implement `get_mapping()`: return DataFrame with risk_score and min_probability columns
    - _Requirements: 12.1-12.5_

  - [ ]* 11.8 Write unit tests for Risk Score Transformer
    - Test percentile fitting on validation data
    - Test probability-to-risk-score transformation
    - Verify mapping monotonicity (higher probability → higher risk score)
    - Test edge cases (probability = 0, probability = 1)
    - Test get_mapping() output format
    - _Requirements: 12.1-12.5_

- [ ] 12. Checkpoint - Evaluation framework complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement artifact management
  - [ ] 13.1 Create Model Serializer component
    - Implement `ModelSerializer` class with `serialize()` and `deserialize()` methods
    - In `serialize()`: accept model, output_path, model_type ('lightgbm' | 'sklearn' | 'onnx')
    - For LightGBM: use `model.booster_.save_model(output_path)` (native .txt format)
    - For sklearn: use `joblib.dump(model, output_path, compress=3)`
    - For ONNX (optional): convert model and write serialized bytes
    - Compute SHA-256 hash: read file in 8KB chunks, update hashlib.sha256()
    - Return `SerializedArtifact` dataclass with path, sha256, model_type, file_size_bytes
    - In `deserialize()`: verify SHA-256 hash before loading, raise ValueError on mismatch
    - Load model based on model_type: lgb.Booster(), joblib.load(), or onnxruntime.InferenceSession()
    - _Requirements: 15.1-15.5, 29.1-29.5_

  - [ ]* 13.2 Write unit tests for Model Serializer
    - Test LightGBM serialization and deserialization
    - Test sklearn serialization and deserialization
    - Test SHA-256 hash computation accuracy
    - Test hash verification on load (should raise ValueError on mismatch)
    - Test handling of corrupted artifacts
    - _Requirements: 15.1-15.5, 29.1-29.5_

  - [ ] 13.3 Create Artifact Manifest schema and generator
    - Define `ArtifactManifest` dataclass with all fields: model_name, model_type, artifact_filename, artifact_sha256, feature_contract_version, training_timestamp, git_commit, training_job_name, data provenance (S3 paths + SHA-256 hashes), split_config, hyperparameters, metrics, selected_threshold, threshold_fpr, threshold_recall, risk_score_mapping, python_version, library_versions, approval metadata
    - Implement `create_manifest()` function accepting model artifacts, metrics, training metadata
    - Populate manifest fields: extract git commit with `subprocess.run(['git', 'rev-parse', 'HEAD'])`
    - Capture library versions: lightgbm.__version__, sklearn.__version__, pandas.__version__, numpy.__version__
    - Serialize manifest to JSON with indent=2
    - _Requirements: 16.1-16.8, 30.1-30.5_

  - [ ]* 13.4 Write unit tests for Artifact Manifest
    - Test manifest creation with complete training artifacts
    - Test JSON serialization and deserialization
    - Test handling of missing optional fields
    - Verify git commit extraction
    - Verify library version capture
    - _Requirements: 16.1-16.8, 30.1-30.5_

  - [ ] 13.5 Implement Promotion Gate
    - Create `PromotionGate` class with `request_promotion()`, `approve_promotion()`, `deny_promotion()` methods
    - In `request_promotion()`: display formatted summary with model metrics, Feature Contract version, training metadata
    - Show performance report path for review
    - Return `PromotionRequest` dataclass with status='pending'
    - In `approve_promotion()`: update manifest with approver, approval_timestamp, deployment_environment
    - Copy artifact and manifest to production S3 bucket using boto3
    - Upload performance report to S3
    - Return `PromotionResult` with approved=True, production_s3_paths, approver, timestamp
    - In `deny_promotion()`: update manifest with denial metadata, archive to staging S3
    - Return `PromotionResult` with approved=False, denial_reason
    - _Requirements: 17.1-17.5_

  - [ ]* 13.6 Write unit tests for Promotion Gate
    - Test promotion request formatting
    - Test approval flow (mock S3 upload)
    - Test denial flow with archival
    - Test manifest updates with approval metadata
    - _Requirements: 17.1-17.5_

- [ ] 14. Implement SageMaker integration
  - [ ] 14.1 Create SageMaker Job Configurator
    - Implement `SageMakerJobConfigurator` class with `configure()` method
    - Accept training_script, hyperparameters, input_data_s3_paths, output_s3_path, instance_type, git_commit, estimated_duration_hours
    - Generate unique job name: `fraud-detection-{timestamp}-{git_short_hash}`
    - Define instance costs: ml.m5.xlarge=$0.23/hr, ml.m5.2xlarge=$0.46/hr
    - Estimate total cost: instance_cost * estimated_duration_hours
    - Create `SageMakerJobConfig` dataclass with job_name, training_script, hyperparameters, I/O paths, instance config, max_runtime_seconds (1.5x buffer), git_commit, estimated_cost, framework (pytorch), version, python_version
    - _Requirements: 13.1-13.6_

  - [ ]* 14.2 Write unit tests for SageMaker Job Configurator
    - Test job name generation (timestamp + git hash)
    - Test cost estimation for different instance types
    - Test max runtime calculation with buffer
    - Test configuration dataclass population
    - _Requirements: 13.1-13.6_

  - [ ] 14.3 Implement Cost Approval Gate
    - Create `CostApprovalGate` class with `request_approval()`, `approve()`, `deny()` methods
    - In `request_approval()`: display formatted cost estimate, instance type, job name, hyperparameters
    - Prompt for user approval (can be interactive or log-based)
    - Return `CostApprovalRequest` dataclass with status='pending'
    - In `approve()`: return `CostApprovalResult` with approved=True, approver, approval_timestamp
    - In `deny()`: return `CostApprovalResult` with approved=False, reviewer, denial_reason
    - _Requirements: 14.1-14.5_

  - [ ]* 14.4 Write unit tests for Cost Approval Gate
    - Test approval request formatting
    - Test approval flow
    - Test denial flow with reason
    - _Requirements: 14.1-14.5_

  - [ ] 14.5 Implement SageMaker Executor
    - Create `SageMakerExecutor` class with `execute()` method
    - Initialize with role_arn, region, create sagemaker.Session()
    - In `execute()`: verify approval (raise ValueError if not approved)
    - Create PyTorch estimator with config parameters: entry_point, role, instance_type, instance_count, volume_size, max_run, framework_version, py_version, hyperparameters, output_path
    - Add tags: Project=FraudDetection, GitCommit, ApprovedBy
    - Submit training job: `estimator.fit(inputs=input_data_s3_paths, job_name=job_name, wait=False)`
    - Log job submission with ARN
    - Return `SageMakerJobExecution` dataclass with job_name, job_arn, status, submission_timestamp
    - _Requirements: 13.1-13.6_

  - [ ]* 14.6 Write unit tests for SageMaker Executor
    - Test job submission with mocked SageMaker API
    - Test approval verification (should raise ValueError if denied)
    - Test estimator configuration
    - Test tagging with metadata
    - _Requirements: 13.1-13.6_

- [ ] 15. Checkpoint - Artifact management and SageMaker integration complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. Implement deployment components
  - [ ] 16.1 Create Inference Container Loader
    - Implement `InferenceContainerLoader` class with `load()` method
    - Accept artifact_path, manifest_path, deployed_feature_contract_version
    - Load manifest JSON, parse into `ArtifactManifest` dataclass
    - Deserialize model using `ModelSerializer.deserialize()` with SHA-256 verification
    - Verify feature contract compatibility: compare major versions, log warning for minor differences, raise ValueError for major mismatch
    - Load feature transformer with fitted parameters from training
    - Load risk score transformer with probability-to-score mapping
    - Generate warmup synthetic transaction
    - Run warmup inference: transform features, predict probability, compute risk score, measure latency
    - Log model metadata: model_name, model_type, feature_contract_version, training_timestamp, git_commit
    - Return `LoadedModel` dataclass with model, manifest, transformer, risk_transformer
    - _Requirements: 18.1-18.6, 19.1-19.5_

  - [ ]* 16.2 Write unit tests for Inference Container Loader
    - Test successful model loading with valid SHA-256
    - Test rejection on SHA-256 mismatch
    - Test feature contract compatibility checks (major, minor, patch versions)
    - Test warmup inference execution
    - Test logging of model metadata
    - _Requirements: 18.1-18.6, 19.1-19.5_

  - [ ] 16.3 Implement Fraud Scorer
    - Create `FraudScorer` class with `score()` and `score_batch()` methods
    - Accept loaded_model, optional fallback_model in __init__
    - In `score()`: transform raw transaction through feature transformer
    - Predict probability: `model.predict_proba(features)[:, 1][0]`
    - Transform to risk score: `risk_transformer.transform([prob])[0]`
    - Handle exceptions: if primary model fails and fallback exists, use fallback model
    - Measure latency: time.time() before and after, convert to milliseconds
    - Log warning if latency > 110ms
    - Return `FraudScore` dataclass with risk_score (int 0-100), probability, latency_ms, used_fallback, model_version, timestamp
    - In `score_batch()`: vectorize transformation and prediction for batch_size transactions
    - _Requirements: 20.1-20.6, 23.1-23.5_

  - [ ]* 16.4 Write unit tests for Fraud Scorer
    - Test single transaction scoring
    - Test batch scoring with multiple transactions
    - Test fallback model invocation on primary failure
    - Test latency measurement accuracy
    - Test risk score range (0-100)
    - _Requirements: 20.1-20.6, 23.1-23.5_

  - [ ] 16.5 Implement fallback scorer selection logic
    - Create `select_scorer()` function accepting transaction, feature_contract
    - Check available features in transaction: extract keys
    - Check required features from contract: filter features where required=True
    - If all required present: return 'primary'
    - If required missing: raise ValueError with missing feature list
    - If optional missing but required present: return 'fallback', log missing features
    - _Requirements: 20.1-20.6_

  - [ ]* 16.6 Write unit tests for scorer selection logic
    - Test primary selection with all features present
    - Test fallback selection with optional features missing
    - Test error on required features missing
    - Test logging of missing optional features
    - _Requirements: 20.1-20.6_

- [ ] 17. Implement end-to-end training pipeline orchestration
  - [ ] 17.1 Create main training pipeline script
    - Load IEEE-CIS dataset from data/raw/ieee-cis/
    - Run Dataset Auditor: generate Dataset Card
    - Run Temporal Leakage Audit: exclude flagged features
    - Create initial Feature Contract (v1.0.0) based on clean features
    - Run Chronological Splitter: generate train/val/test splits, persist IDs
    - Fit Feature Transformer on training data: store fitted parameters
    - Transform all splits using fitted transformer
    - Train Baseline Model: logistic regression with CV, evaluate on validation
    - Train GBT Model: LightGBM with Optuna tuning, evaluate on validation
    - Train Fallback Model: LightGBM on required features only
    - Compute metrics for all models on all splits using Metrics Calculator
    - Select threshold using Threshold Selector on validation set (target 0.5% FPR)
    - Assess calibration using Calibration Assessor on test set
    - Fit Risk Score Transformer on validation probabilities
    - Generate performance report: metrics, confusion matrices, feature importance, calibration plots
    - Serialize all models using Model Serializer with SHA-256 hashing
    - Create Artifact Manifests for all models with complete metadata
    - Output artifacts to models/ directory, manifests to manifests/
    - _Requirements: All training requirements 1-12, 21-28_

  - [ ]* 17.2 Write integration test for end-to-end pipeline
    - Test pipeline execution with small synthetic dataset
    - Verify all intermediate artifacts created (Dataset Card, splits, models, manifests)
    - Verify model performance metrics computed
    - Verify serialization and manifest generation
    - Test reproducibility with same seed
    - _Requirements: 21.1-21.5_

- [ ] 18. Checkpoint - Deployment and orchestration complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. Implement performance benchmarking
  - [ ] 19.1 Create latency benchmark suite
    - Load trained GBT model and fallback model
    - Load Feature Transformer and Risk Score Transformer
    - Generate 1000 synthetic test transactions
    - Measure single-transaction inference latency for each
    - Compute p50, p95, p99 latency percentiles
    - Verify p95 latency ≤ 110ms on target hardware (CPU: AWS m5.2xlarge or equivalent)
    - Log warning if p99 > 110ms, recommend optimization
    - Generate latency distribution plot (histogram)
    - _Requirements: 23.1-23.5_

  - [ ]* 19.2 Write performance tests for latency
    - Test latency measurement on known hardware
    - Verify p95 threshold enforcement
    - Test with different transaction complexities (varying missingness)
    - Test fallback model latency (should also meet ≤110ms)
    - _Requirements: 23.1-23.5_

  - [ ] 19.3 Create throughput benchmark suite
    - Test batch inference with batch_size=[10, 50, 100, 500]
    - Measure total time for processing 10,000 transactions
    - Compute throughput: transactions per second
    - Verify throughput ≥ 100 txn/sec for batch_size=100
    - Generate throughput vs batch_size plot
    - _Requirements: 23.1-23.5_

  - [ ]* 19.4 Write performance tests for throughput
    - Test batch inference with varying batch sizes
    - Verify throughput meets minimum target
    - Test memory usage during batch processing
    - _Requirements: 23.1-23.5_

- [ ] 20. Implement property-based test suite
  - [ ]* 20.1 Write property test: Feature Type Classification Correctness
    - **Property 2: Feature Type Classification Correctness**
    - **Validates: Requirements 1.6**
    - Use hypothesis to generate random DataFrames with mixed column types
    - Verify feature type classifier correctly identifies CATEGORICAL (object/string, cardinality < 50) and NUMERIC (int/float) without misclassification
    - Test with 100+ random datasets
    - _Requirements: 1.6_

  - [ ]* 20.2 Consolidate all property tests
    - Ensure all 9 properties implemented: Properties 1-9
    - Organize tests in tests/property/ directory
    - Tag each test with feature name and property number
    - Configure pytest to run property tests with minimum 100 iterations
    - Generate property test report showing pass/fail for each property
    - _Requirements: All correctness properties 1-9_

- [ ] 21. Implement comprehensive unit test suite
  - [ ]* 21.1 Write unit tests for error handling
    - Test DataValidationError on missing columns
    - Test ModelArtifactError on corrupted files
    - Test FeatureContractError on version mismatch
    - Test LatencyBudgetExceededError on slow inference
    - Test ApprovalDeniedError on rejected promotion
    - _Requirements: Error handling scenarios_

  - [ ]* 21.2 Write unit tests for edge cases
    - Test empty datasets, single-row datasets
    - Test extreme class imbalance (99.99% legitimate)
    - Test all features missing for a transaction
    - Test feature contract with zero required features
    - Test calibration with perfectly calibrated probabilities
    - _Requirements: Robustness testing_

  - [ ]* 21.3 Write unit tests for SageMaker error handling
    - Test handling of OutOfMemory failure
    - Test handling of MaxRuntimeExceeded failure
    - Test handling of SpotInstanceInterruption failure
    - Test retry logic recommendations
    - _Requirements: SageMaker integration error handling_

- [ ] 22. Implement integration test suite
  - [ ]* 22.1 Write integration test: Dataset audit to feature contract
    - Load IEEE-CIS dataset
    - Run audit, generate Dataset Card
    - Create Feature Contract from audited features
    - Verify contract contains only clean features (no leakage)
    - Verify contract version initialized to 1.0.0
    - _Requirements: 1-4_

  - [ ]* 22.2 Write integration test: Training to serialization
    - Train baseline and GBT models on small dataset
    - Serialize models with SHA-256 hashing
    - Deserialize models and verify hash matches
    - Predict on test set with deserialized models
    - Verify predictions match original models
    - _Requirements: 7-8, 15, 29_

  - [ ]* 22.3 Write integration test: Model promotion workflow
    - Serialize trained model
    - Create artifact manifest
    - Request promotion (mock user approval)
    - Approve promotion, upload to S3 (mock)
    - Verify manifest updated with approval metadata
    - _Requirements: 15-17_

  - [ ]* 22.4 Write integration test: Inference container startup
    - Create mock model artifact and manifest
    - Load model in inference container
    - Verify SHA-256 hash checked
    - Verify feature contract compatibility checked
    - Verify warmup inference executed
    - Verify startup logs contain model metadata
    - _Requirements: 18-19_

  - [ ]* 22.5 Write integration test: Fallback scoring
    - Load primary and fallback models
    - Create transaction with missing optional features
    - Score transaction (should use fallback)
    - Verify fallback flag set in FraudScore
    - Verify risk score within [0, 100]
    - _Requirements: 20_

- [ ] 23. Implement performance reporting
  - [ ] 23.1 Create performance report generator
    - Compile metrics from Metrics Calculator: recall, FPR, AUC, PR-AUC by split
    - Generate confusion matrices for each split at selected threshold
    - Extract feature importance rankings from GBT model
    - Generate calibration plots with predicted vs observed fraud rates
    - Compare GBT and Baseline side-by-side
    - Include latency benchmarks (p50, p95, p99)
    - Include threshold-recall-FPR curve
    - Format as HTML report with tables and plots (use matplotlib, seaborn)
    - Include disclaimer for synthetic data performance (if applicable)
    - Persist report to performance_reports/ directory
    - _Requirements: 24.1-24.7, 25.1-25.4_

  - [ ]* 23.2 Write unit tests for performance report
    - Test report generation with known metrics
    - Verify all sections present (metrics, confusion matrices, feature importance, calibration)
    - Test synthetic data disclaimer inclusion
    - Test HTML formatting
    - _Requirements: 24.1-24.7, 25.1-25.4_

- [ ] 24. Implement training reproducibility verification
  - [ ]* 24.1 Write determinism test for full pipeline
    - Run training pipeline twice with same seed
    - Compute SHA-256 hash of serialized models from both runs
    - Verify hashes match (identical models)
    - Verify split IDs match
    - Verify metrics match within floating-point tolerance
    - _Requirements: 21.1-21.5_

  - [ ]* 24.2 Write test for dependency version pinning
    - Verify requirements.txt contains exact versions (no ~= or >=)
    - Test that manifest captures library versions
    - _Requirements: 21.5, 29.3_

- [ ] 25. Final checkpoint - All components and tests complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 26. Create documentation and deployment guides
  - [ ] 26.1 Write README.md
    - Project overview and architecture diagram
    - Installation instructions: Python version, dependencies, setup commands
    - Usage guide: training pipeline execution, model promotion, inference container deployment
    - Configuration options: hyperparameter tuning, instance types, cost budgets
    - Performance benchmarks and optimization tips
    - _Requirements: Documentation for deployment_

  - [ ] 26.2 Write DEPLOYMENT.md
    - SageMaker training job setup and execution
    - Cost approval workflow
    - Model promotion workflow
    - Inference container deployment on AWS (ECS/Fargate or Lambda)
    - Monitoring and alerting setup
    - Rollback procedures
    - _Requirements: Deployment documentation_

  - [ ] 26.3 Write TESTING.md
    - Unit test organization and execution
    - Property-based test strategy and configuration
    - Integration test scenarios
    - Performance benchmark setup
    - CI/CD integration (GitHub Actions or similar)
    - _Requirements: Testing documentation_

  - [ ] 26.4 Create example notebooks
    - Exploratory data analysis on IEEE-CIS dataset
    - Feature engineering examples
    - Model training and evaluation walkthrough
    - Hyperparameter tuning analysis
    - Calibration and threshold selection examples
    - _Requirements: User guidance_

## Notes

- **Property-Based Testing**: Tasks marked with `*` are property-based tests. Each property test should run a minimum of 100 iterations to ensure thorough input space coverage. Tag format: `# Feature: ml-model-pipeline, Property {N}: {property_text}`

- **Requirements Coverage**: Each task explicitly references the requirements it implements for traceability. All 32 requirements are covered across the task list.

- **Checkpoints**: Regular checkpoints ensure incremental validation. At each checkpoint, run all relevant tests and verify functionality before proceeding.

- **Optional Tasks**: Tasks marked with `*` are optional and can be skipped for a faster MVP. However, property-based tests are strongly recommended for correctness guarantees.

- **Testing Strategy**: The plan includes comprehensive testing at all levels:
  - Unit tests for individual components
  - Property-based tests for universal correctness properties (9 properties)
  - Integration tests for end-to-end workflows
  - Performance tests for latency and throughput

- **Incremental Development**: The task order supports incremental development and testing:
  1. Data layer → Feature engineering → Training → Evaluation
  2. Artifact management → SageMaker integration → Deployment
  3. Orchestration → Performance benchmarking → Comprehensive testing

- **AWS Integration**: SageMaker and S3 integration tasks may require AWS credentials and permissions. Mocking is recommended for unit tests; actual AWS calls should be in integration tests.

- **Reproducibility**: Multiple tasks enforce reproducibility: seeded random operations, dependency pinning, SHA-256 verification, git commit tracking.
