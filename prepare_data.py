"""
Этот скрипт подготавливает исходные данные, выполняя генерацию признаков,
очистку и разделение их на обучающие и тестовые наборы.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_path: str = 'train.csv', output_dir: str = './data'):
    """
    Загружает, обрабатывает и разделяет набор данных.

    Аргументы:
        input_path (str): Путь к исходным данным CSV.
        output_dir (str): Каталог для сохранения обработанных файлов.
    """
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

    print("Performing feature engineering...")
    df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])
    df['driver_reg_date'] = pd.to_datetime(df['driver_reg_date'])

    df['bid_increase_ratio'] = (df['price_bid_local'] - df['price_start_local']) / df['price_start_local']
    df['bid_increase_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
    df['bid_increase_ratio'].fillna(0, inplace=True)

    df['hour_of_day'] = df['order_timestamp'].dt.hour
    df['day_of_week'] = df['order_timestamp'].dt.dayofweek

    df['driver_experience_days'] = (df['order_timestamp'] - df['driver_reg_date']).dt.days

    def map_car_to_class(car_name):
        business_brands = ['Toyota', 'Mercedes-Benz', 'BMW', 'Lexus', 'Audi', 'Porsche', 'Genesis']
        comfort_brands = ['Volkswagen', 'Skoda', 'Ford', 'Kia', 'Hyundai', 'Mazda', 'Honda', 'Subaru', 'Nissan', 'Mitsubishi']
        
        if car_name in business_brands:
            return 'business'
        elif car_name in comfort_brands:
            return 'comfort'
        else:
            return 'econom'

    df['car_class'] = df['carname'].apply(map_car_to_class)
    df.drop(columns=['carname'], inplace=True)

    df['is_done'] = df['is_done'].apply(lambda x: 1 if x == 'done' else 0)
    print("Feature engineering complete.")

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

    print("\nData preparation finished.")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape:  {test_df.shape}")
    print(f"Processed files saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    prepare_data()
