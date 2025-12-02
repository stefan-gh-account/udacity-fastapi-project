import requests

url = "https://udacity-fastapi-project.onrender.com/infer"
input_dict = {"age":39,
              "workclass":"Private",
              "fnlgt":9999,
              "education":"Bachelors",
              "marital-status":"Married-civ-spouse",
              "occupation":"Exec-managerial",
              "relationship":"Husband",
              "race":"White",
              "sex":"Male",
              "capital-gain":0,
              "capital-loss":0,
              "hours-per-week":40,
              "native-country":"United-States"}

response = requests.post(url, json=input_dict)
print(response.status_code)
print(response.text)
print(response.json())
