# Load Model
from keras.models import load_model
new_model = load_model(model_file_path)

new_model.summary()
#new_model.get_weights()
new_model.optimizer

# JSON: only architecture
json_string = model.to_json()

print(json_string)

from keras.models import model_from_json
model_architecture = model_from_json(json_string)

model_architecture.summary()

# Only weights
#model.sample_weights('models/my_model_weights.h5')
#model2.load_weights('models/my_model_weights.h5')
