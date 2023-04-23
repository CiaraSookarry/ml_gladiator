import pandas as pd

from sklearn.model_selection import train_test_split

# Read the data file & print first 2 entries
# filepath is relative to dir script is called from
iris = pd.read_csv("./dataset/iris_data.csv") 
print(iris.head(2))

# Split data into train, test, & validation
# Split is train=0.7, test=0.2, validation=0.1
train, test_val = train_test_split(iris, train_size=0.7, random_state=5, shuffle=True)
test, val = train_test_split(test_val, train_size=0.66, random_state=5, shuffle=True)

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")
print(f"Validation size: {len(val)}")