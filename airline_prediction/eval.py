from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy

def evaluate_model(name, y_true, y_pred):
    rmse = numpy.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nPerformance of {name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}