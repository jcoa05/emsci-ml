# %% [markdown]
# # Stacked Ensemble Learning: Python Migration

# Migrating SCI ML workflow from tidymodels to scikit-learn
# Target: `motor_score_12m` (continuous), grouped by `center_id`

# %%
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

print("Packages loaded.")

# %% [markdown]
# ## Configuration

# %%
PARAMS = {
    'cv_folds': 5,
    'cv_repeats': 3,
    'grid_size': 25,
    'bootstrap_iters': 1000,
    'mcid_val': 4,
    'recovery_thresh': 60
}

print("Control parameters:")
for key, value in PARAMS.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Data Loading

# %%
df = pd.read_csv("data.csv")

print(f"Dataset: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Outcome: motor_score_12m (range: {df['motor_score_12m'].min()} - {df['motor_score_12m'].max()})")
print(f"Groups: center_id ({df['center_id'].nunique()} unique centers)")

# %%
# Column definitions
target_col = 'motor_score_12m'
group_col = 'center_id'
id_cols = ['patient_id', 'patient_id_global', 'center_id']

feature_cols = [col for col in df.columns if col not in [target_col] + id_cols]

# Separate numeric and categorical features
numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

print(f"\nFeatures: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
print(f"Categorical: {categorical_cols}")

# %% [markdown]
# ## Preprocessing Pipeline
# 
# Equivalent to tidymodels recipe: `step_dummy()` + `step_normalize()` + `step_zv()`

# %%
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    # drop='first' matches step_dummy(one_hot = FALSE) reference coding
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='drop'  # Drops ID columns
)

print("Preprocessing pipeline defined.")

# %% [markdown]
# ## Cross-Validation Strategy
# 
# GroupKFold doesn't support repeats natively, so implementing manually.

# %%
def create_grouped_cv_splits(X, y, groups, n_splits=5, n_repeats=3, random_state=123):
    """
    Mimics group_vfold_cv(df, group = center_id, v = 5, repeats = 3).
    Returns list of dicts with train/val indices.
    """
    all_splits = []
    
    for repeat in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits)
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            all_splits.append({
                'repeat': repeat + 1,
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'n_train': len(train_idx),
                'n_val': len(val_idx)
            })
    
    return all_splits

# Prepare data
X = df[feature_cols]
y = df[target_col]
groups = df[group_col]

cv_splits = create_grouped_cv_splits(
    X, y, groups,
    n_splits=PARAMS['cv_folds'],
    n_repeats=PARAMS['cv_repeats'],
    random_state=RANDOM_STATE
)

print(f"CV: {PARAMS['cv_folds']}-fold grouped, {PARAMS['cv_repeats']} repeats = {len(cv_splits)} resamples")

# Verify group separation
print("\nVerifying group separation (first 3 splits):")
for split in cv_splits[:3]:
    train_groups = set(groups.iloc[split['train_idx']].unique())
    val_groups = set(groups.iloc[split['val_idx']].unique())
    overlap = train_groups.intersection(val_groups)
    print(f"  Repeat {split['repeat']}, Fold {split['fold']}: overlap = {len(overlap)} (should be 0)")

# %% [markdown]
# ## Hyperparameter Grids

# %%
np.random.seed(RANDOM_STATE)

# ElasticNet grid
# sklearn: alpha (penalty strength), l1_ratio (mixture)
elnet.param.grid = [
    {
        'alpha': 10 ** np.random.uniform(-5, 1),
        'l1_ratio': np.random.uniform(0, 1)
    }
    for _ in range(PARAMS['grid_size'])
]

print(f"ElasticNet grid: {len(elnet.param.grid)} configurations")

# %%
np.random.seed(RANDOM_STATE)

# XGBoost grid
# max_depth ~ tree_depth, learning_rate ~ learn_rate, min_child_weight ~ min_n,
# gamma ~ loss_reduction, subsample ~ sample_size
xgb.param.grid = []
for _ in range(PARAMS['grid_size']):
    config = {
        'max_depth': int(np.random.uniform(3, 10)),
        'learning_rate': 10 ** np.random.uniform(-3, np.log10(0.3)),
        'min_child_weight': int(np.random.uniform(2, 20)),
        'gamma': 10 ** np.random.uniform(-3, 1),
        'subsample': np.random.uniform(0.5, 1.0),
        'n_estimators': 2000,
        'random_state': RANDOM_STATE,
        'n_jobs': 1,  # Single-threaded to prevent oversubscription
        'verbosity': 0
    }
    xgb.param.grid.append(config)

print(f"XGBoost grid: {len(xgb.param.grid)} configurations")

# %%
# Training complexity
total_elnet_episodes = len(elnet.param.grid) * len(cv_splits)
total_xgb_episodes = len(xgb.param.grid) * len(cv_splits)

print(f"\nTraining episodes:")
print(f"  ElasticNet: {total_elnet_episodes}")
print(f"  XGBoost: {total_xgb_episodes}")
print(f"  Total: {total_elnet_episodes + total_xgb_episodes}")

# %% [markdown]
# ## Base Learner Training
# 
# Using explicit loop over folds to inspect indices and track progress.

# %%
def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, R-squared."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'rsq': r2_score(y_true, y_pred)
    }

# %%
# === ELASTICNET TUNING ===

print("=" * 50)
print("TRAINING ELASTICNET")
print("=" * 50)

elnet_results = []
start_time = datetime.now()

for param_idx, params in enumerate(elnet.param.grid):
    
    fold_metrics = []
    
    for split in cv_splits:
        X_train = X.iloc[split['train_idx']]
        X_val = X.iloc[split['val_idx']]
        y_train = y.iloc[split['train_idx']]
        y_val = y.iloc[split['val_idx']]
        
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', ElasticNet(
                alpha=params['alpha'],
                l1_ratio=params['l1_ratio'],
                max_iter=10000,
                random_state=RANDOM_STATE
            ))
        ])
        
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)
        
        metrics = calculate_metrics(y_val, y_pred)
        metrics['repeat'] = split['repeat']
        metrics['fold'] = split['fold']
        fold_metrics.append(metrics)
    
    # Aggregate across folds
    fold_df = pd.DataFrame(fold_metrics)
    
    result = {
        'param_idx': param_idx,
        'alpha': params['alpha'],
        'l1_ratio': params['l1_ratio'],
        'rmse_mean': fold_df['rmse'].mean(),
        'rmse_std': fold_df['rmse'].std(),
        'mae_mean': fold_df['mae'].mean(),
        'mae_std': fold_df['mae'].std(),
        'rsq_mean': fold_df['rsq'].mean(),
        'rsq_std': fold_df['rsq'].std()
    }
    elnet_results.append(result)
    
    if (param_idx + 1) % 5 == 0:
        print(f"  Completed {param_idx + 1}/{len(elnet.param.grid)} configurations...")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\nElasticNet complete: {elapsed:.1f} seconds")

elnet_results_df = pd.DataFrame(elnet_results)

best_elnet_idx = elnet_results_df['rmse_mean'].idxmin()
best_elnet = elnet_results_df.loc[best_elnet_idx]

print(f"\nBest ElasticNet:")
print(f"  alpha: {best_elnet['alpha']:.6f}")
print(f"  l1_ratio: {best_elnet['l1_ratio']:.4f}")
print(f"  RMSE: {best_elnet['rmse_mean']:.4f} (±{best_elnet['rmse_std']:.4f})")
print(f"  R²: {best_elnet['rsq_mean']:.4f}")

# %%
# === XGBOOST TUNING ===

print("=" * 50)
print("TRAINING XGBOOST")
print("=" * 50)

xgb_results = []
start_time = datetime.now()

for param_idx, params in enumerate(xgb.param.grid):
    
    fold_metrics = []
    
    for split in cv_splits:
        X_train = X.iloc[split['train_idx']]
        X_val = X.iloc[split['val_idx']]
        y_train = y.iloc[split['train_idx']]
        y_val = y.iloc[split['val_idx']]
        
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(**params))
        ])
        
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)
        
        metrics = calculate_metrics(y_val, y_pred)
        metrics['repeat'] = split['repeat']
        metrics['fold'] = split['fold']
        fold_metrics.append(metrics)
    
    fold_df = pd.DataFrame(fold_metrics)
    
    result = {
        'param_idx': param_idx,
        'max_depth': params['max_depth'],
        'learning_rate': params['learning_rate'],
        'min_child_weight': params['min_child_weight'],
        'gamma': params['gamma'],
        'subsample': params['subsample'],
        'rmse_mean': fold_df['rmse'].mean(),
        'rmse_std': fold_df['rmse'].std(),
        'mae_mean': fold_df['mae'].mean(),
        'mae_std': fold_df['mae'].std(),
        'rsq_mean': fold_df['rsq'].mean(),
        'rsq_std': fold_df['rsq'].std()
    }
    xgb_results.append(result)
    
    if (param_idx + 1) % 5 == 0:
        elapsed_so_far = (datetime.now() - start_time).total_seconds()
        estimated_total = elapsed_so_far / (param_idx + 1) * len(xgb.param.grid)
        print(f"  Completed {param_idx + 1}/{len(xgb.param.grid)} configs... "
              f"(~{estimated_total - elapsed_so_far:.0f}s remaining)")

elapsed = (datetime.now() - start_time).total_seconds()
print(f"\nXGBoost complete: {elapsed:.1f} seconds")

xgb_results_df = pd.DataFrame(xgb_results)

best_xgb_idx = xgb_results_df['rmse_mean'].idxmin()
best_xgb = xgb_results_df.loc[best_xgb_idx]

print(f"\nBest XGBoost:")
print(f"  max_depth: {best_xgb['max_depth']}")
print(f"  learning_rate: {best_xgb['learning_rate']:.6f}")
print(f"  min_child_weight: {best_xgb['min_child_weight']}")
print(f"  gamma: {best_xgb['gamma']:.6f}")
print(f"  subsample: {best_xgb['subsample']:.4f}")
print(f"  RMSE: {best_xgb['rmse_mean']:.4f} (±{best_xgb['rmse_std']:.4f})")
print(f"  R²: {best_xgb['rsq_mean']:.4f}")

# %% [markdown]
# ## Results Summary

# %%
print("=" * 50)
print("BASE LEARNER COMPARISON")
print("=" * 50)

comparison_df = pd.DataFrame([
    {
        'Model': 'ElasticNet',
        'RMSE': f"{best_elnet['rmse_mean']:.4f} (±{best_elnet['rmse_std']:.4f})",
        'MAE': f"{best_elnet['mae_mean']:.4f} (±{best_elnet['mae_std']:.4f})",
        'R²': f"{best_elnet['rsq_mean']:.4f} (±{best_elnet['rsq_std']:.4f})"
    },
    {
        'Model': 'XGBoost',
        'RMSE': f"{best_xgb['rmse_mean']:.4f} (±{best_xgb['rmse_std']:.4f})",
        'MAE': f"{best_xgb['mae_mean']:.4f} (±{best_xgb['mae_std']:.4f})",
        'R²': f"{best_xgb['rsq_mean']:.4f} (±{best_xgb['rsq_std']:.4f})"
    }
])

print(comparison_df.to_string(index=False))

# %%
# Save results
elnet_results_df.to_csv("elnet_tuning_results.csv", index=False)
xgb_results_df.to_csv("xgb_tuning_results.csv", index=False)

best_params = {
    'elnet': {
        'alpha': float(best_elnet['alpha']),
        'l1_ratio': float(best_elnet['l1_ratio'])
    },
    'xgb': {
        'max_depth': int(best_xgb['max_depth']),
        'learning_rate': float(best_xgb['learning_rate']),
        'min_child_weight': int(best_xgb['min_child_weight']),
        'gamma': float(best_xgb['gamma']),
        'subsample': float(best_xgb['subsample']),
        'n_estimators': 2000,
        'random_state': RANDOM_STATE
    }
}

joblib.dump(best_params, "best_params.joblib")

print("\nSaved:")
print("  elnet_tuning_results.csv")
print("  xgb_tuning_results.csv")
print("  best_params.joblib")

# %%
# Fit final models on full data
print("\nFitting final models on full data...")

final_elnet_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', ElasticNet(
        alpha=best_params['elnet']['alpha'],
        l1_ratio=best_params['elnet']['l1_ratio'],
        max_iter=10000,
        random_state=RANDOM_STATE
    ))
])
final_elnet_pipeline.fit(X, y)

final_xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(**best_params['xgb']))
])
final_xgb_pipeline.fit(X, y)

joblib.dump(final_elnet_pipeline, "final_elnet_model.joblib")
joblib.dump(final_xgb_pipeline, "final_xgb_model.joblib")

print("Saved:")
print("  final_elnet_model.joblib")
print("  final_xgb_model.joblib")

# %%
print("\n" + "=" * 50)
print("PHASE 5 COMPLETE")
print("=" * 50)
print("Base learner training finished. Ready for stacking phase.")

# TODO: Add PyTorch neural network as third base learner.
# Sticking to sklearn-compatible models for now to get the pipeline working.
