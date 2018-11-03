
import pandas as pd
import policy
import time

df = pd.read_feather("./cache_files/fpolicy/ohe.feather")[:1000]
df = policy.to_trainable(df)
df.drop("index",axis = 1,inplace = True)

nn_model = policy.nn()
pred = nn_model.predict(df)
print(pred)
actions = policy.get_action(pred,greedy = 0.99)
print(actions)

"""
model =  policy.create_reward_model(nn_model)
start = time.time()
actions = model.predict(df)
print(time.time() - start)
print(actions.shape)
print(actions.mean(axis = 1))
"""

