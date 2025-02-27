import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🔍 Load data
data_dict = pickle.load(open('./final_data.pickle', 'rb'))
data = np.asarray(data_dict['data'], dtype=np.float32)  
labels = np.asarray(data_dict['labels'])
# print unique labels
# print(np.unique(labels))

# 🎯 Train-Test Split 
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,  
    max_depth=20,  
    random_state=42,
    n_jobs=-1,  
    verbose=1
)

# 🏋️‍♂️ Train the model
model.fit(x_train, y_train)

# 🔍 Make predictions
y_predict = model.predict(x_test)

# 📊 Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# ✅ Save the trained model
with open("model.p", "wb") as f:
    pickle.dump({"model": model}, f)

print("Model saved successfully!")
