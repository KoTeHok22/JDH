"""
Этот скрипт обучает классификатор LightGBM на обработанных данных,
оценивает его и сохраняет обученный конвейер модели.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def train_model(data_dir: str = './data', model_dir: str = './model'):
    """
    Обучает и сохраняет модель машинного обучения.

    Аргументы:
        data_dir (str): Каталог, где хранятся обработанные данные.
        model_dir (str): Каталог для сохранения обученной модели.
    """
    print("Starting model training...")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'processed_train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'processed_test.csv'))
        print("Successfully loaded processed data.")
    except FileNotFoundError:
        print(f"Error: Processed data not found in '{data_dir}'. Please run prepare_data.py first.")
        return

    TARGET = 'is_done'
    FEATURES = [
        'bid_increase_ratio',
        'price_start_local',
        'price_bid_local',
        'hour_of_day',
        'day_of_week',
        'driver_experience_days',
        'platform',
        'car_class'
    ]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    lgbm = lgb.LGBMClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgbm)
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [20, 31],
        'classifier__max_depth': [5, 7]
    }

    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Hyperparameter tuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")

    model_pipeline = grid_search.best_estimator_

    print("Evaluating model performance...")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model ROC-AUC on test data: {auc:.4f}")

    
    model_path = os.path.join(model_dir, 'bid_recommender_model.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"Model pipeline saved to: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train_model()
