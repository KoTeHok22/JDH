
import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def train_model(data_dir: str = './data', model_dir: str = './model'):
    print("Starting model training...")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'processed_train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'processed_test.csv'))
        print("Successfully loaded processed data (with manual encoding).")
    except FileNotFoundError:
        print(f"Error: Processed data not found in '{data_dir}'. Please run prepare_data.py first.")
        return

    TARGET = 'is_done'
    FEATURES = [
        'price_start_local', 'price_bid_local', 'driver_rating',
        'hour_of_day', 'day_of_week', 'driver_experience_days', 'bid_increase_ratio',
        'platform', 'car_class'
    ]
    
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    categorical_features = ['platform', 'car_class']

    model = lgb.LGBMClassifier(
        random_state=42, 
        n_estimators=200, 
        learning_rate=0.1, 
        num_leaves=31
    )

    print("Training the LightGBM model directly...")
    model.fit(
        X_train, 
        y_train,
        categorical_feature=categorical_features
    )
    print("Model training complete.")

    print("Evaluating model performance on the test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model ROC-AUC on TEST SET: {auc:.4f}")

    model_path = os.path.join(model_dir, 'bid_recommender_model.joblib')
    joblib.dump(model, model_path)
    print(f"Raw model saved to {model_path}")

if __name__ == '__main__':
    train_model()