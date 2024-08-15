import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'advertising.csv'
data = pd.read_csv(file_path)

column_to_exclude = 'Sales'

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

test_data_without_column = test_data.drop(columns=[column_to_exclude])

train_data.to_csv('advertising_train.csv', index=False)
test_data_without_column.to_csv('advertising_test.csv', index=False)

print("Train and test datasets have been created and saved as 'advertising_train.csv' and 'advertising_test.csv'.")