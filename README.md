# s2_MonicaClara_Petrangeli
Assignment 2 for Prototyping with AI is build upon assignment 1, imporving the previous version of the app

## What is the utility of the prototype?
The **Airline Reviews Prediction App** is a valuable tool for travelers looking to make smarter flight decisions—whether it’s figuring out if a flight is worth recommending or getting a sense of what others thought of an airline. Powered by machine learning, the app predicts whether a flight is likely to be recommended based on key factors like seat comfort, food quality, cabin service, entertainment, and WiFi. Trained on over six thousand real reviews, the model captures what passengers really care about—a glimpse into the collective travel experience.

Users can input their flight details and ratings to receive tailored recommendations, while also getting a big-picture view of average ratings and trends over time. This makes it easier to steer clear of disappointing flights and stay updated on airline performance. The latest updates take things up a notch—users can now explore where an airline flies, making it simple to check if it serves their country or fits their travel plans.

The app also includes LLM-powered sentiment analysis, giving users a deeper understanding of passenger experiences. Beyond just ratings, it reveals the overall vibe—whether reviews are mostly positive, neutral, or negative. Another handy feature is the ability to track past predictions and compare flights side by side, helping users spot patterns and make better choices. Plus, users can download their recommendations as a PDF, making it easy to save and share insights for future trips.

In short, the Airline Reviews Prediction App isn’t just about predictions—it’s a full-on travel companion. By combining machine learning, sentiment analysis, and smart data visualization, it gives users the knowledge they need to make confident flight choices, ultimately enhancing their travel experience.

## What are the main design decisions chosen?
The main design decisions I took ranged from the SQL implementation for efficiency to sentiment analysis with LLM to the downloadable predictions.

One key design decision was implementing an SQL database to improve efficiency and reduce redundant processing. By creating unique IDs for each observation, the app can store and retrieve cached results quickly, cutting down on API calls and improving overall performance.

Sentiment analysis using an LLM was another important feature. The app leverages GPT to analyze passenger reviews and classify them as positive, neutral, or negative. Caching the results in the SQL database prevents exceeding API limits and ensures faster processing.

To make the app more user-friendly, downloadable predictions were added. Users can save and share their insights by generating a structured PDF using the FPDF library, making it easy to reference past recommendations.

Lastly, an interactive map helps users visualize flight routes. The app shows flight origins and destinations, with coordinates cached in the SQL database to speed up loading times and improve reliability.

## What main difficulties did I find?
One of the biggest challenges I faced was setting up a chatbot to simplify how users input variables for predictions. The goal was to make the process more intuitive, but getting the chatbot to smoothly collect all the necessary inputs turned out to be tricky. On top of that, I had trouble storing the prediction inputs correctly before running the model, which sometimes led to incomplete or inconsistent data being passed to the prediction function.

The map visualization and sentiment analysis posed their own headaches. The map would often time out when too many requests hit the geocoding service in a short period, thanks to rate limits. Similarly, the LLM handling sentiment analysis would max out on calls, causing delays and occasional errors in the app's performance.

To tackle these issues, I introduced an SQL database to store both the coordinates and sentiment analysis results for filtered data. This way, the app doesn’t need to repeatedly process the same observations—improving both efficiency and speed. I created a unique key for each observation, which lets the app recognize and pull up previously processed data, even if the observation’s order in the dataset changes.

By caching the results in the database, I was able to drastically cut down on the number of API calls to both the geocoding service and the LLM. This made the app run smoother and faster, while also making it more scalable and reliable for future use.

## Screen Capture requested
Link:
https://urledu-my.sharepoint.com/:v:/g/personal/monicaclara_petrangeli_alumni_esade_edu/EezzQ9_3KvNEgehxUP1rjJ4Bvto3A7CMm3oPn0Hj90-A2A?e=KMRB9T&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D
