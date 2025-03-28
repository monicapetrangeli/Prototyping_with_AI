# s2_MonicaClara_Petrangeli
Assignment 2 for Prototyping with AI is build upon assignment 1, imporving the previous version of the app

## What is the utility of the prototype?
The **Airline Reviews Prediction App** is a comprehensive tool designed to help travelers make informed decisions about their flights. Whether it's determining if a flight is worth recommending or exploring what others thought of an airline, the app provides actionable insights. By leveraging machine learning, the app predicts whether a flight is recommendable based on key factors such as seat comfort, food quality, cabin service, entertainment, and WiFi. Trained on over six thousand real reviews, the model captures what passengers truly value, offering a glimpse into the collective travel experience.

The app allows users to input their flight details and ratings to receive personalized recommendations. Beyond predictions, it provides an overview of average ratings and trends over time, empowering users to avoid subpar flights and stay informed about airline performance. The added features from the initial iteration significantly enhance its utility. Users can now visualize the possible destinations and origins of an airline's flights, helping them determine if the airline serves their country or meets their travel needs.

Additionally, the integration of an LLM-powered sentiment analysis enables users to go beyond overall ratings and understand the majority sentiment behind reviews—whether positive, neutral, or negative. This feature provides a deeper layer of insight into passenger experiences. The app also introduces the ability to track past predictions, allowing users to compare flights side by side. Furthermore, users can download their recommendations as a PDF, making it easy to save and share their findings for future reference.

In summary, the Airline Reviews Prediction App is not just a prediction tool but a comprehensive travel companion. It combines machine learning, sentiment analysis, and data visualization to empower users with the knowledge they need to make the best flight choices, enhancing their overall travel experience.

## What are the main design decisions chosen?
The **Airline Reviews Prediction App** is a comprehensive tool designed to help travelers make informed decisions about their flights. Whether it's determining if a flight is worth recommending or exploring what others thought of an airline, the app provides actionable insights. By leveraging machine learning, the app predicts whether a flight is recommendable based on key factors such as seat comfort, food quality, cabin service, entertainment, and WiFi. Trained on over six thousand real reviews, the model captures what passengers truly value, offering a glimpse into the collective travel experience.

The app allows users to input their flight details and ratings to receive personalized recommendations. Beyond predictions, it provides an overview of average ratings and trends over time, empowering users to avoid subpar flights and stay informed about airline performance. The added features from the initial iteration significantly enhance its utility. Users can now visualize the possible destinations and origins of an airline's flights, helping them determine if the airline serves their country or meets their travel needs.

Additionally, the integration of an LLM-powered sentiment analysis enables users to go beyond overall ratings and understand the majority sentiment behind reviews—whether positive, neutral, or negative. This feature provides a deeper layer of insight into passenger experiences. The app also introduces the ability to track past predictions, allowing users to compare flights side by side. Furthermore, users can download their recommendations as a PDF, making it easy to save and share their findings for future reference.

In summary, the Airline Reviews Prediction App is not just a prediction tool but a comprehensive travel companion. It combines machine learning, sentiment analysis, and data visualization to empower users with the knowledge they need to make the best flight choices, enhancing their overall travel experience.

## What main difficulties did I find?
One of the main challenges I faced was implementing a chatbot to simplify the input of variables for predictions. While the idea was to make the process more user-friendly, I encountered difficulties in creating a seamless flow for the chatbot to collect all the required inputs. Additionally, I struggled with correctly storing all the prediction inputs before running the model, which led to incomplete or inconsistent data being passed to the prediction function.

Another significant challenge was with the map visualization and sentiment analysis. The map often timed out when too many requests were sent in a short period, as the geocoding service has rate limits. Similarly, the LLM used for sentiment analysis frequently reached its maximum call limit, causing delays and errors in the app's functionality.

To address these issues, I implemented an SQL database to store the coordinates and sentiment analysis results for the filtered data. This ensures that the app does not redundantly process the same observations multiple times, improving efficiency and performance. To achieve this, I created a unique key as the primary key for each observation, allowing the app to identify and retrieve previously processed data regardless of its index in the filtered or original dataset.

By caching the results in the database, I was able to significantly reduce the number of API calls to both the geocoding service and the LLM, ensuring smoother operation and a better user experience. This approach not only resolved the performance bottlenecks but also made the app more scalable and reliable for future use.

## Screen Capture requested
Link:
https://urledu-my.sharepoint.com/:v:/g/personal/monicaclara_petrangeli_alumni_esade_edu/EezzQ9_3KvNEgehxUP1rjJ4Bvto3A7CMm3oPn0Hj90-A2A?e=KMRB9T&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D
