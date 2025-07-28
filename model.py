from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preparation import prepare_data
import pickle as pk
from config import settings


def build_model(): 
    df = prepare_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_train_test(X,y)
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    print(score)
    save_model(model)
    return score

def get_X_y(data, col_X = ["area", "constraction_year", "bedrooms", "garden", "balcony_yes", "parking_yes", "furnished_yes", "garage_yes", "storage_yes"], col_y = "rent"):
    return data[col_X], data[col_y]


def split_train_test(X,y): 
    """
    input: 
    ||---->  X, y
    output: 
    ||----> X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    return model.score(X_test, y_test)


def save_model(model):
    pk.dump(model, open(f"{settings.model_path}/{settings.model_name}", "wb"))
