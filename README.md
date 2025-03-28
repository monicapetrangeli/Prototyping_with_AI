# s1_MonicaClara_Petrangeli
Assignment 1 for Prototyping with AI
Run the following command in your terminal to install all dependencies:
pip install -r requirements_assignment_1.txt

## Why did I choose this model?
I chose the *Random Forest classifier model* to predict whether a flight should be recommended or not. This model was specifically created for this assignment, though it was inspired by my bachelor thesis. I wanted to delve deeper into airline reviews and explore the untapped value they hold.

## What is the utility of the prototype?
The **Airline Reviews Prediction App** is a valuable tool for travelers looking to make informed decisions on flights—whether it's deciding if they should recommend the flight or simply wanting to visualize what others thought of the airline. Leveraging machine learning, the app predicts whether a flight could be recommended to friends and family, considering factors such as seat comfort, food quality, cabin service, entertainment, and WiFi. Trained on over six thousand real reviews, the model provides insights into what passengers truly care about—*a penny for their thoughts*, if you will.
The app allows users to input their flight details and ratings to receive personalized recommendations, while also offering an overview of average ratings and trends over time. It empowers users to avoid subpar flights and stay informed about airline performance, ultimately enhancing their travel experience. The utility extends beyond just prediction, with visualizations and trend analysis that help users uncover broader patterns, making it an excellent resource for frequent travelers and anyone looking to make the best choice when booking a flight.

## What main difficulties did I find?
Despite the large volume and variety of data available, several challenges arose during the development of the app. One significant issue was inconsistent input for the flight route. Sometimes, the route was recorded using city names, while other times it was provided as IATA airport codes. This inconsistency made it difficult to process the route correctly within the code. Converting all airport codes into cities is a time-consuming task, and although I’m actively working on it, this issue currently limits the ability to use the origin in the machine learning model and prevents the use of maps within the app.
Another challenge was the limitations of Streamlit. While an easy-to-use and excellent tool for rapid prototyping, its lack of flexibility in design proved frustrating. For someone accustomed to working with HTML and CSS, the inability to adjust font sizes, margins, and other design elements made it difficult to achieve the desired level of customization and control over the app’s appearance.

## Screen Capture requested
Link:
https://urledu-my.sharepoint.com/:v:/g/personal/monicaclara_petrangeli_alumni_esade_edu/EezzQ9_3KvNEgehxUP1rjJ4Bvto3A7CMm3oPn0Hj90-A2A?e=KMRB9T&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D
