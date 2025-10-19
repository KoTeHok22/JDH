from flask import Flask, request, jsonify, render_template
from bid_recommender import get_recommendations

app = Flask(__name__)

@app.route("/", methods=['GET'])
def read_root():
    return render_template("index.html")

@app.route("/recommend_price", methods=['POST'])
def recommend_price():
    order_data = request.get_json()
    print(f"Received order: {order_data}")

    recommendations = get_recommendations(order_data)

    if "error" in recommendations:
        return jsonify({"error": recommendations["error"]}), 500

    print(f"Generated recommendation: {recommendations}")

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8961, debug=True)