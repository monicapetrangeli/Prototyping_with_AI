# Monica Clara Petrangeli - Prototyping with AI Project
The **Airline Reviews Prediction App** is a comprehensive tool designed to help travelers make informed decisions about flights. It leverages machine learning, sentiment analysis, and interactive visualizations to provide actionable insights into airline performance and passenger experiences. The app is user-friendly and offers a range of features to enhance the travel planning process.

# How to Use the App
1. Filtering Options:
     - Use the sidebar to select the airline and seat type you want to explore.
     - The app filters the data based on your selections to provide personalized insights.
2. Overview Tab:
     - Learn about the app's features and how it works.
     - This tab introduces the app's purpose and highlights its key functionalities.
3. Visuals Tab:
     - Explore detailed visualizations of airline performance:
         - *Average Ratings*: View average ratings for seat comfort, food, cabin service, and more.
         - *Time Series*: Analyze trends in ratings over time for selected features.
         - *Map Visualization*: See the origins and destinations of flights, with airport codes mapped to city names.
         - *Sentiment Analysis*: Understand the sentiment breakdown (positive, neutral, negative) of passenger reviews.
         - *Word Cloud*: Discover the most frequently mentioned keywords in reviews.
4. Predictions Tab:
     - Input your flight details and ratings to predict whether the flight is recommendable.
     - View the prediction result, including the model's confidence level.
     - Track your past predictions and download them as a PDF for future reference.

# What the App is For
The app is designed to:
- Help travelers decide whether to recommend a flight to friends and family.
- Provide insights into airline performance based on real passenger reviews.
- Visualize trends and patterns in airline ratings and sentiment.
- Allow users to compare flights and make data-driven travel decisions.

# Main Components in Terms of Architecture
- Frontend (User Interface):
    - Built using Streamlit, providing an interactive and visually appealing interface.
    - Organized into tabs (Overview, Visuals, Predictions) for a clean and intuitive user experience.
- Backend (Data Processing and Machine Learning):
    - Machine Learning Model:
      A predictive model trained on over 6,000 real reviews to classify flights as "Recommend" or "Not Recommend."
    - Sentiment Analysis:
      Powered by OpenAI's GPT model to classify reviews as positive, neutral, or negative.
    - SQL Database:
      Used to cache geolocation data and sentiment analysis results, reducing redundant API calls and improving efficiency.
    - Data Visualization:
        - Matplotlib and Streamlit Charts:
          Used for pie charts, bar charts, and word clouds to visualize ratings and sentiment.
        - Folium:
          Used for interactive map visualizations of flight origins and destinations.
- Data Storage and Caching:
    - SQLite Databases:
      Two databases are used:
        - One for storing geolocation data (coordinates of origins and destinations).
        - Another for storing sentiment analysis results.
        - This ensures efficient data retrieval and avoids repeated computations.
- PDF Generation:
    - FPDF Library:
      Used to generate downloadable PDFs of past predictions, allowing users to save and share their results.
    - Error Handling and Scalability:
      Includes mechanisms to handle API rate limits and timeouts for geolocation and sentiment analysis.
      Caching results in the database ensures the app remains efficient and scalable.

# Key Features
- *Personalized Predictions*: Input your flight details to receive a recommendation.
- *Sentiment Analysis*: Understand the sentiment behind passenger reviews.
- *Interactive Visualizations*: Explore trends, maps, and keyword insights.
- *Downloadable Reports*: Save your predictions as a PDF for future reference.
- *Efficient Caching*: Avoid redundant computations with SQL-based caching.

This app is a powerful travel companion for frequent flyers and anyone looking to make smarter, data-driven decisions when booking flights. It combines advanced machine learning with user-friendly design to deliver a seamless and insightful experience.
