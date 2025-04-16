import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("🧁 Cupcake Sales Predictor")
st.markdown("Predict tomorrow's cupcake sales based on weather and past sales")

# Sample historical data
data = {
    'temperature': [22, 25, 30, 35, 28, 20, 18, 33, 31, 27],
    'rain_mm': [0, 2, 5, 0, 1, 10, 15, 0, 0, 3],
    'yesterday_sales': [120, 135, 150, 180, 160, 100, 90, 175, 170, 140],
    'sales_today': [130, 140, 155, 190, 165, 95, 85, 185, 175, 150]
}
df = pd.DataFrame(data)

# Train the model
X = df[['temperature', 'rain_mm', 'yesterday_sales']]
y = df['sales_today']
model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter Tomorrow's Weather and Sales Info")

temperature = st.number_input("Temperature (°C)", value=30)
rain_mm = st.number_input("Rain (mm)", value=0)
yesterday_sales = st.number_input("Yesterday's Sales", value=150)

# Prediction
if st.button("Predict Tomorrow's Sales"):
    input_df = pd.DataFrame([[temperature, rain_mm, yesterday_sales]],
                            columns=['temperature', 'rain_mm', 'yesterday_sales'])
    prediction = model.predict(input_df)[0]
    st.success(f"🌟 Predicted cupcake sales for tomorrow: **{prediction:.2f}**")

# Optional: Show the data used
with st.expander("See training data"):
    st.dataframe(df)
