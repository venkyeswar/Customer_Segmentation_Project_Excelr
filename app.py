from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

# Group analysis with names, behavior, and product recommendations
group_analysis = {
    0: {
        "name": "Moderate Buyers",
        "behavior": "Spend less on wines and other products, with minimal purchases of luxury items.",
        "recommendation": "Focus on basic needs and family-oriented offers."
    },
    1: {
        "name": "Top Spenders",
        "behavior": "Spend high on wines, meat, fruits, and gold products.",
        "recommendation": "Offer premium and luxury items."
    },
    2: {
        "name": "Moderate to Low Buyers",
        "behavior": "Similar to Group 0, but slightly higher on meat and wine.",
        "recommendation": "Prioritize value-for-money deals."
    },
    3: {
        "name": "Luxury Shoppers",
        "behavior": "Heavy spenders on high-end wines, fruits, and gold items.",
        "recommendation": "Target exclusive deals and premium bundles."
    },
    4: {
        "name": "Occasional Buyers",
        "behavior": "Purchase infrequently and focus on selective items.",
        "recommendation": "Offer discounts on occasional high-value purchases."
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data in the specified order
        inputs = [
            request.form['Marital_Status'],                # Categorical: Marital_Status
            int(request.form['MntWines']),                # Numerical: MntWines
            int(request.form['MntFruits']),               # Numerical: MntFruits
            int(request.form['MntMeatProducts']),         # Numerical: MntMeatProducts
            int(request.form['MntFishProducts']),         # Numerical: MntFishProducts
            int(request.form['MntSweetProducts']),        # Numerical: MntSweetProducts
            int(request.form['MntGoldProds']),            # Numerical: MntGoldProds
            int(request.form['NumDealsPurchases']),       # Numerical: NumDealsPurchases
            int(request.form['NumStorePurchases']),       # Numerical: NumStorePurchases
            int(request.form['Years_Since_Join']),        # Numerical: Years_Since_Join
            int(request.form['No_of_Childrens']),         # Numerical: No_of_Childrens
            request.form['is_parent'],                    # Categorical: Is_Parent
            int(request.form['Memebers_In_Family'])       # Numerical: Members_In_Family
        ]

        # Convert categorical data to numeric
        if inputs[0] == 'Single':
            inputs[0] = 0
        elif inputs[0] == 'Married':
            inputs[0] = 1
        else:
            inputs[0] = 1  # Default to 1 if not explicitly "Single" or "Married"

        inputs[11] = 1 if inputs[11] == 'Yes' else 0  # Convert 'is_parent' to numeric

        # Calculate the total amount spent on products
        total_amount = sum(inputs[1:7])  # Sum of MntWines to MntGoldProds

        # Create input DataFrame in the specified order
        input_data = {
            "Marital_Status": [inputs[0]],
            "MntWines": [inputs[1]],
            "MntFruits": [inputs[2]],
            "MntMeatProducts": [inputs[3]],
            "MntFishProducts": [inputs[4]],
            "MntSweetProducts": [inputs[5]],
            "MntGoldProds": [inputs[6]],
            "NumDealsPurchases": [inputs[7]],
            "NumStorePurchases": [inputs[8]],
            "Years_Since_Join": [inputs[9]],
            "Total_Amnt_Spend": [total_amount],
            "No_of_Childrens": [inputs[10]],
            "Is_Parent": [inputs[11]],
            "Memebers_In_Family": [inputs[12]]
        }

        input_df = pd.DataFrame(input_data)

        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Predict using the trained model
        prediction = model.predict(scaled_input)[0]  # Extract the single prediction value

        # Get the analysis for the predicted label
        group_info = group_analysis.get(prediction, {})
        group_name = group_info.get("name", "Unknown Group")
        behavior = group_info.get("behavior", "No behavior analysis available.")
        recommendation = group_info.get("recommendation", "No recommendation available.")

        return render_template(
            'result.html',
            label=prediction,
            group_name=group_name,
            behavior=behavior,
            recommendation=recommendation
        )

    except Exception as e:
        # Handle errors gracefully
        return f"An error occurred: {e}"

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0" ,port=5000,debug=False)
