from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preparation import prepare_data
import pickle as pk
from config import settings
from loguru import logger


def build_model(): 
    logger.info("Starting model build process")
    df = prepare_data()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = train_model(X_train, y_train)
    score = evaluate_model(model, X_test, y_test)
    logger.info(f"Model evaluation score: {score}")
    save_model(model)
    logger.info("Model build process completed")
    return score

def get_X_y(data, col_X = ["area", "constraction_year", "bedrooms", "garden", "balcony_yes", "parking_yes", "furnished_yes", "garage_yes", "storage_yes"], col_y = "rent"):
    logger.info(f"Extracting features {col_X} and target '{col_y}' from data")
    return data[col_X], data[col_y]

def split_train_test(X, y): 
    logger.info("Splitting data into train and test sets")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logger.info("Training RandomForestRegressor model")
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model performance on test set")
    score = model.score(X_test, y_test)
    logger.info(f"Model R^2 score: {score}")
    return score

def save_model(model):
    logger.info(f"Saving model to {settings.model_path}/{settings.model_name}")
    pk.dump(model, open(f"{settings.model_path}/{settings.model_name}", "wb"))
    logger.info("Model saved successfully")
