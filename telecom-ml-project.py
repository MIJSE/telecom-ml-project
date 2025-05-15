import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle

# Загрузка данных из Excel
def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        if not all(col in data.columns for col in ['calls', 'duration', 'load']):
            raise ValueError("Отсутствуют необходимые столбцы.")
        return data
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

# Подготовка данных
def prepare_data(data):
    X = data[['calls', 'duration']]
    y = data['load']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

# Оценка модели
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"MAE модели: {error:.4f}")

# Сохранение модели
def save_model(model, filename='station_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# Основной процесс
def main():
    excel_file = "telephone_data.xlsx"
    data = load_data(excel_file)
    if data is None:
        return

    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
