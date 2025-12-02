
import joblib
model_dict = joblib.load("model/model_dict.pkl")
print(type(model_dict["model"]))
print(model_dict["model"])
print(dir(model_dict["model"]))
print(dir(model_dict["model"].get_params()))
