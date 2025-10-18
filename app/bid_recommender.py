"""
Этот скрипт содержит логику для загрузки обученной модели
и генерации рекомендаций по ставкам на основе входных данных заказа.
"""
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "model" / "bid_recommender_model.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
    model = None

def get_recommendations(order_data: dict) -> list[dict]:
    """
    Генерирует список рекомендуемых цен ставок, их вероятностей успеха
    и ожидаемого дохода для данного заказа.

    Аргументы:
        order_data (dict): Словарь, содержащий параметры заказа:
            - price_start_local (float): Начальная цена заказа.
            - platform (str): Платформа (например, 'android', 'ios').
            - car_class (str): Класс автомобиля (например, 'econom', 'comfort', 'business').
            - driver_reg_date (str): Дата регистрации водителя (ГГГГ-ММ-ДД).

    Возвращает:
        list[dict]: Список словарей, каждый из которых содержит:
            - recommended_price (float): Рекомендуемая цена ставки.
            - success_probability (float): Прогнозируемая вероятность успеха.
            - expected_income (float): Рассчитанный ожидаемый доход.
    """
    if model is None:
        return [{
            "recommended_price": order_data.get('price_start_local', 0.0),
            "success_probability": 0.0,
            "expected_income": 0.0
        }]

    price_start_local = float(order_data.get('price_start_local', 150.0))
    platform = order_data.get('platform', 'android')
    car_class = order_data.get('car_class', 'econom')
    driver_reg_date_str = order_data.get('driver_reg_date', '2022-01-15')

    bid_range = np.arange(price_start_local, price_start_local * 1.5, 10)
    if not bid_range.any():
        bid_range = [price_start_local]

    order_timestamp = datetime.now()
    driver_reg_date = datetime.strptime(driver_reg_date_str, '%Y-%m-%d')

    test_data = []
    for bid in bid_range:
        bid_increase_ratio = (bid - price_start_local) / price_start_local if price_start_local > 0 else 0
        test_data.append({
            'bid_increase_ratio': bid_increase_ratio,
            'price_start_local': price_start_local,
            'price_bid_local': bid,
            'hour_of_day': order_timestamp.hour,
            'day_of_week': order_timestamp.weekday(),
            'driver_experience_days': (order_timestamp - driver_reg_date).days,
            'platform': platform,
            'car_class': car_class
        })

    df_test = pd.DataFrame(test_data)

    try:
        probabilities = model.predict_proba(df_test)[:, 1]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return [{
            "recommended_price": price_start_local,
            "success_probability": 0.0,
            "expected_income": 0.0
        }]

    df_test['probability'] = probabilities
    df_test['expected_income'] = df_test['price_bid_local'] * df_test['probability']

    if df_test.empty or 'expected_income' not in df_test.columns or df_test['expected_income'].isnull().all():
        recommendations = [{
            "recommended_price": round(price_start_local, 2),
            "success_probability": 0.0,
            "expected_income": 0.0
        }]
    else:
        all_bids = df_test.sort_values(by='expected_income', ascending=False)
        recommendations = []
        for _, row in all_bids.iterrows():
            recommendations.append({
                "recommended_price": round(row['price_bid_local'], 2),
                "success_probability": round(row['probability'], 4),
                "expected_income": round(row['expected_income'], 2)
            })

    return recommendations
