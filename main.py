# Put the code for your API here.
import fastapi
import pydantic
import joblib
import pandas
import ml.model
import ml.data

model_dict = joblib.load("model/model_dict.pkl")
app = fastapi.FastAPI()

example_payload = {"age": 39,
                   "workclass": "Private",
                   "fnlgt": 9999,
                   "education": "Bachelors",
                   "marital-status": "Married-civ-spouse",
                   "occupation": "Exec-managerial",
                   "relationship": "Husband",
                   "race": "White",
                   "sex": "Male",
                   "capital-gain": 0,
                   "capital-loss": 0,
                   "hours-per-week": 40,
                   "native-country": "United-States"}

@app.get("/")
async def welcome_message():
  msg = ("Welcome to the app use post with path /infer and pass a dictionary "
         "of the example form to predict salary.")
  return {"message": msg, "example_payload": example_payload}


class InferenceRequest(pydantic.BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    marital_status: str = pydantic.Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = pydantic.Field(..., alias="capital-gain")
    capital_loss: int = pydantic.Field(..., alias="capital-loss")
    hours_per_week: int = pydantic.Field(..., alias="hours-per-week")
    native_country: str = pydantic.Field(..., alias="native-country")


@app.post("/infer")
async def inference(request: InferenceRequest):
    try:
      data = pandas.DataFrame([fastapi.encoders.jsonable_encoder(request)])
      x, _, _, _ = ml.data.process_data(data, categorical_features=ml.data.categorical_features, training=False,
                                        encoder=model_dict["encoder"])
      salary_prediction = model_dict["model"].predict(x).reshape(-1, 1)
      salaries = model_dict["label_binarizer"].inverse_transform(salary_prediction)
      return {"salary": salaries[0]}
    except Exception as e:
      return {"error": str(e)}
