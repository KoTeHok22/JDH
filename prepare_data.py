import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_path: str = 'train.csv', output_dir: str = './data'):
    print("Starting data preparation...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {input_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please place it in the root directory.")
        return

    df.drop(columns=['tender_id', 'carmodel'], inplace=True, errors='ignore')

    print("Performing feature engineering...")
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
    df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'])

    df['bid_increase_ratio'] = (df['price_bid_local'] - df['price_start_local']) / df['price_start_local']
    df['bid_increase_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
    df['bid_increase_ratio'].fillna(0, inplace=True)

    df['hour_of_day'] = df['order_timestamp'].dt.hour
    df['day_of_week'] = df['order_timestamp'].dt.weekday
    df['driver_experience_days'] = (df['order_timestamp'] - df['driver_reg_date']).dt.days

    def map_car_to_class(car_name):
        if not isinstance(car_name, str):
            return 'econom'
        car_name = car_name.lower()
        business_brands = ['mercedes-benz', 'bmw', 'lexus', 'audi', 'porsche', 'genesis']
        comfort_brands = ['volkswagen', 'skoda', 'ford', 'kia', 'hyundai', 'mazda', 'honda', 'subaru', 'nissan', 'mitsubishi']
        if car_name in business_brands:
            return 'business'
        elif car_name in comfort_brands:
            return 'comfort'
        else:
            return 'econom'

    df['car_class'] = df['carname'].apply(map_car_to_class)
    df.drop(columns=['carname'], inplace=True)

    df['is_done'] = df['is_done'].apply(lambda x: 1 if x == 'done' else 0)

    print("Mapping categorical features to integers...")
    platform_map = {'android': 0, 'ios': 1}
    car_class_map = {'econom': 0, 'comfort': 1, 'business': 2}
    df['platform'] = df['platform'].map(platform_map)
    df['car_class'] = df['car_class'].map(car_class_map)
    df['platform'].fillna(-1, inplace=True)
    df['car_class'].fillna(-1, inplace=True)

    print("Feature engineering complete.")

    median_driver_rating = df['driver_rating'].median()
    if 'user_rating' in df.columns:
        median_user_rating = df['user_rating'].median()
        df['user_rating'].fillna(median_user_rating, inplace=True)
        print("NaN values in user_rating filled with median.")

    df['driver_rating'].fillna(median_driver_rating, inplace=True)
    print("NaN values in driver_rating filled with median.")

    initial_rows = len(df)
    df = df[df['driver_experience_days'] >= 0]
    filtered_rows = initial_rows - len(df)
    print(f"Filtered out {filtered_rows} anomalous rows (e.g., negative driver experience).")

    print("Splitting data into training and testing sets (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_done'])

    train_output_path = os.path.join(output_dir, 'processed_train.csv')
    test_output_path = os.path.join(output_dir, 'processed_test.csv')

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Processed data saved to '{train_output_path}' and '{test_output_path}'.")

if __name__ == '__main__':
    prepare_data()