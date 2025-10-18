"""
Этот скрипт запускает Flask-приложение для рекомендателя ставок.
"""
from flask import Flask, request, jsonify, render_template
from bid_recommender import get_recommendations

app = Flask(__name__)

@app.route("/", methods=['GET'])
def read_root():
    """Обслуживает главную HTML-страницу для демонстрации."""
    return render_template("index.html")

@app.route("/recommend_price", methods=['POST'])
def recommend_price():
    """
    Рекомендует оптимальную цену ставки для данного заказа.
    """
    order_data = request.get_json()
    print(f"Received order: {order_data}")

    recommendations = get_recommendations(order_data)

    if recommendations and "error" in recommendations[0]:
        return jsonify(recommendations[0]), 500

    print(f"All recommendations: {recommendations}")

    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
