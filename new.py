from roboflow import Roboflow

rf = Roboflow(api_key="V8EhAppumdHyBRAQ9XSY")
project = rf.workspace().project("football-team-separation")
model = project.version(15).model

# infer on a local image
model.predict("img.png", confidence=40, overlap=30).save("prediction.jpg")
