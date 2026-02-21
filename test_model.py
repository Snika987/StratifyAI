
from Config.model import get_model

model = get_model()

response = model.invoke("Say hello in one sentence.")

print(response.content)
