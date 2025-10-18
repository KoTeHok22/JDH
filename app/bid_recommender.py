"""
Этот скрипт содержит логику для загрузки обученной модели
и генерации рекомендаций по ставкам на основе входных данных заказа.
"""
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.optimize import minimize_scalar

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".." / "model" / "bid_recommender_model.joblib"
model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Raw model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
    model = None

# --- Product-driven Guardrails ---
MAX_PRICE_MULTIPLIER = 1.8
# Lowered threshold to be less conservative
BAD_SCENARIO_PROBABILITY_THRESHOLD = 0.30

# --- Manual Encoding Maps ---
# These must match the mappings in prepare_data.py
PLATFORM_MAP = {'android': 0, 'ios': 1}
CAR_CLASS_MAP = {'econom': 0, 'comfort': 1, 'business': 2}

# --- Final Business Logic Override ---
# Subtler adjustment multiplier to avoid pushing bids into the 'risky' zone too often.
CAR_CLASS_FINAL_ADJUSTMENT = {'econom': 1.0, 'comfort': 1.03, 'business': 1.08}

# The feature order must match the list used in train_model.py
FEATURE_ORDER = [
    'price_start_local', 'price_bid_local', 'driver_rating',
    'hour_of_day', 'day_of_week', 'driver_experience_days', 'bid_increase_ratio',
    'platform', 'car_class'
]

def get_recommendations(order_data: dict) -> dict:
    """
    Генерирует контекстно-зависимые рекомендации по ставкам,
    включая ограничители и финальные бизнес-корректировки.
    """
    if model is None:
        return {"error": "Model not loaded."}

    # 1. Извлечение и базовая подготовка данных
    price_start_local = float(order_data.get('price_start_local', 150.0))
    platform_str = order_data.get('platform', 'android')
    car_class_str = order_data.get('car_class', 'econom')
    driver_reg_date_str = order_data.get('driver_reg_date', '2022-01-01')
    driver_rating = float(order_data.get('driver_rating', 4.9))

    order_timestamp_str = order_data.get('order_timestamp')
    if order_timestamp_str:
        order_timestamp = datetime.fromisoformat(order_timestamp_str)
    else:
        order_timestamp = datetime.now()
    driver_reg_date = datetime.strptime(driver_reg_date_str, '%Y-%m-%d')
    driver_experience_days = (order_timestamp - driver_reg_date).days

    # 2. Ручное кодирование категориальных признаков
    platform_encoded = PLATFORM_MAP.get(platform_str, -1)
    car_class_encoded = CAR_CLASS_MAP.get(car_class_str, -1)

    # 3. Базовый набор признаков (числовых)
    base_features_dict = {
        'price_start_local': price_start_local,
        'driver_rating': driver_rating,
        'hour_of_day': order_timestamp.hour,
        'day_of_week': order_timestamp.weekday(),
        'driver_experience_days': driver_experience_days,
        'platform': platform_encoded,
        'car_class': car_class_encoded
    }

    # 4. Целевая функция для предсказания вероятности
    def get_prob_success(bid, features_dict):
        features_dict['price_bid_local'] = bid
        features_dict['bid_increase_ratio'] = (bid - price_start_local) / price_start_local if price_start_local > 0 else 0
        
        try:
            # Создаем DataFrame с правильным порядком столбцов
            df = pd.DataFrame([features_dict], columns=FEATURE_ORDER)
            return model.predict_proba(df)[:, 1][0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0.0

    # 5. Целевая функция для оптимизатора
    def objective_function(bid, features_dict):
        prob_success = get_prob_success(bid, features_dict)
        expected_income = bid * prob_success
        return -expected_income

    # 6. Запуск оптимизации
    bounds = (price_start_local, price_start_local * MAX_PRICE_MULTIPLIER)
    result = minimize_scalar(
        objective_function,
        bounds=bounds,
        method='bounded',
        args=(base_features_dict.copy(),)
    )

    # Математически оптимальная ставка
    optimal_bid = result.x
    
    # 7. **Финальная Бизнес-Корректировка**
    # Применяем жесткий множитель на основе класса авто.
    adjustment_multiplier = CAR_CLASS_FINAL_ADJUSTMENT.get(car_class_str, 1.0)
    adjusted_bid = optimal_bid * adjustment_multiplier

    # Убедимся, что скорректированная ставка не превышает максимальную границу
    adjusted_bid = min(adjusted_bid, bounds[1])

    # Пересчитываем финальные метрики с учетом скорректированной ставки
    final_prob = get_prob_success(adjusted_bid, base_features_dict.copy())
    final_expected_income = adjusted_bid * final_prob

    # 8. Define Recommendation Tiers
    # The optimal bid is the direct result from the optimizer
    optimal_prob = final_prob
    optimal_income = final_expected_income

    # Safe bid: 15% lower than optimal, for conservative plays
    safe_bid = adjusted_bid * 0.85
    safe_prob = get_prob_success(safe_bid, base_features_dict.copy())
    safe_income = safe_bid * safe_prob

    # Risky bid: 15% higher than optimal, for aggressive plays
    risky_bid = adjusted_bid * 1.15
    # Ensure risky bid doesn't exceed the hard cap
    risky_bid = min(risky_bid, bounds[1])
    risky_prob = get_prob_success(risky_bid, base_features_dict.copy())
    risky_income = risky_bid * risky_prob

    # 9. Apply Final Contextual Strategy & Format Output
    prob_at_start_price = get_prob_success(price_start_local, base_features_dict.copy())
    expected_income_at_start = price_start_local * prob_at_start_price

    # Core Economic Decision: Only proceed if the optimized bid offers higher expected income.
    if final_expected_income <= expected_income_at_start:
        return {
            "strategy_type": "unprofitable_order",
            "recommendation_text": f"Повышение ставки для этого заказа невыгодно. " \
                                   f"Рекомендуем принять начальную цену {price_start_local:.2f} руб. " \
                                   f"(ожидаемый доход: {expected_income_at_start:.2f} руб.) или пропустить заказ.",
            "options": [
                {
                    "strategy": "initial_price",
                    "recommended_price": round(price_start_local, 2),
                    "success_probability": round(float(prob_at_start_price), 4),
                    "expected_income": round(expected_income_at_start, 2)
                }
            ]
        }

    # If profitable, generate the full spectrum of strategic options
    optimal_prob = final_prob
    optimal_income = final_expected_income

    # Safe bid: 15% lower than optimal
    safe_bid = adjusted_bid * 0.85
    safe_prob = get_prob_success(safe_bid, base_features_dict.copy())
    safe_income = safe_bid * safe_prob

    # Risky bid: 15% higher than optimal
    risky_bid = adjusted_bid * 1.15
    risky_bid = min(risky_bid, bounds[1]) # Adhere to hard cap
    risky_prob = get_prob_success(risky_bid, base_features_dict.copy())
    risky_income = risky_bid * risky_prob

    return {
        "strategy_type": "multi_option",
        "recommendation_text": f"Оптимальная ставка: {adjusted_bid:.2f} руб. Выберите свою стратегию.",
        "options": [
            {
                "strategy": "safe",
                "recommended_price": round(safe_bid, 2),
                "success_probability": round(float(safe_prob), 4),
                "expected_income": round(safe_income, 2)
            },
            {
                "strategy": "optimal",
                "recommended_price": round(adjusted_bid, 2),
                "success_probability": round(float(optimal_prob), 4),
                "expected_income": round(optimal_income, 2)
            },
            {
                "strategy": "risky",
                "recommended_price": round(risky_bid, 2),
                "success_probability": round(float(risky_prob), 4),
                "expected_income": round(risky_income, 2)
            }
        ]
    }