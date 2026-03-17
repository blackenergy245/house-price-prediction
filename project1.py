from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import joblib

data = pd.read_excel('House prices.xlsx')
df = pd.DataFrame(data)

model = LinearRegression()
X = df[['Area (sqft)', 'Bedrooms','Age (years)' ,'LocationScore (1-10)']]
y = df['Price ($)']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=32)

model.fit(X_train,y_train)

predicted_prices = model.predict(X_test)
print("The predicted prices are:",predicted_prices)
print(y_test)
plt.scatter(x=y_test,y=predicted_prices,color = "Blue")
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual VS predictions")
plt.savefig("Predictions.png")
plt.show()

r2 = r2_score(y_test,predicted_prices)
mae = mean_absolute_error(y_test,predicted_prices)
mse = mean_squared_error(y_test,predicted_prices)

print("R2 score:",r2)
print("Mean absoulute error: ",mae)
print("Mean squared error: ",mse)

joblib.dump(model,'House price predictor.pkl')
print("Saved in disk!")

new_houses = pd.DataFrame({
    'Area (sqft)': [900, 1200, 1500, 800, 1700],
    'Bedrooms': [2, 3, 4, 1, 4],
    'Age (years)': [3, 7, 10, 2, 12],
    'LocationScore (1-10)': [8, 7, 6, 9, 5]
})

a = joblib.load('House price predictor.pkl')
predicted = a.predict(new_houses)
print("New predicted prices are:",predicted)

with pd.ExcelWriter('Overview.xlsx',engine='xlsxwriter') as writer:
    df.to_excel(writer,sheet_name='Raw data',index=False)

    workbook = writer.book
    worksheet = workbook.add_worksheet("Predicted prices")

    worksheet.insert_image('B2',"Predictions.png")