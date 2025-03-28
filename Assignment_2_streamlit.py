import streamlit as st
import pandas as pd
import joblib
import math
from config import OPENAI_API_KEY
import folium
from streamlit_folium import st_folium
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from fpdf import FPDF
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import openai
import time
import sqlite3 
import random


#Setting up openai API key
openai.api_key=OPENAI_API_KEY

#Importing the predictive ML
model=joblib.load('airline_review_model.joblib')

data=pd.read_csv('Airline_review_with_cities.csv')

# Initialize session state for predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

#Setting up an SQL table to store the sentiment analysis results and the coordinates or origin and destination. So that data will not be computed twice for the same observation
#To do so a unique id is created in the original dataset will be implemented to use as Primary Key
data['Unique_ID'] = data['Airline Name'] + '_' + data['Seat Type'] + '_' + data['Review'].str[:50]

def create_sentiment_table():
    """Creates the sentiment_results table if it doesn't exist."""
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_results (
            unique_id TEXT PRIMARY KEY,
            sentiment TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_coordinates_table():
    """Creates the coordinates table if it doesn't exist."""
    conn = sqlite3.connect('coordinates_cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordinates (
            city TEXT PRIMARY KEY,
            latitude REAL,
            longitude REAL
        )
    ''')
    conn.commit()
    conn.close()

create_sentiment_table()
create_coordinates_table()

#Defining a function to retrieve a previously analyzed review sentiment
def get_sentiment_from_db(unique_id):
    """Retrieves the sentiment for a unique identifier from the database."""
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('SELECT sentiment FROM sentiment_results WHERE unique_id = ?', (unique_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

#Defining a function to save the new sentiment analysis
def save_sentiment_to_db(unique_id, sentiment):
    """Saves the sentiment for a unique identifier to the database."""
    conn = sqlite3.connect('sentiment_analysis.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO sentiment_results (unique_id, sentiment) VALUES (?, ?)', (unique_id, sentiment))
    conn.commit()
    conn.close()

#Defining functions to save and get the coordinates to and from the table
def get_coordinates_from_db(city):
    """Retrieves the coordinates for a city from the database."""
    conn = sqlite3.connect('coordinates_cache.db')
    cursor = conn.cursor()
    cursor.execute('SELECT latitude, longitude FROM coordinates WHERE city = ?', (city,))
    result = cursor.fetchone()
    conn.close()
    return result if result else (None, None)

def save_coordinates_to_db(city, latitude, longitude):
    """Saves the coordinates for a city to the database."""
    conn = sqlite3.connect('coordinates_cache.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO coordinates (city, latitude, longitude) VALUES (?, ?, ?)', (city, latitude, longitude))
    conn.commit()
    conn.close()

#Extracting all possible values of Airline Name and Seat Type to input within the sidebar select box
airlines=data['Airline Name'].unique()
seat_types=data['Seat Type'].unique()
#Creating visuals to be filtered for each airline name and seat type
st.sidebar.title('Filtering Options')
airline_choice=st.sidebar.selectbox('What airline did you travel?',airlines)
seat_type_choice=st.sidebar.selectbox('What seat did you have?',seat_types)

col1, col2 = st.columns([1, 5])
with col1:
    st.image('logo.webp',use_container_width=True)
with col2:
    st.title(':blue[Airline Reviews]')
st.markdown("### Discover the best flights based on real reviews!")

#Creating tap for a less crowded and more clean visualization of the data and model
tab_overview,tab_visuals,tab_predictions = st.tabs(["Overview","Visuals", "Predictions"])

with tab_overview:
    #Introducing the user to the features and use of the 'app' created
    st.markdown("""
    ## Welcome aboard! 🛫  
    Ready to embark on a journey of **data-driven flight recommendations**?  
    This app is your **co-pilot** for making smarter travel decisions! 💡  

    ### What’s it about?  
    Imagine having a **personal travel assistant** that reads thousands of airline reviews, analyzes them, and tells you whether your flight is worth recommending.  
    Well, you don’t have to imagine anymore — it’s here! 🚀  

    ### Key Features:
    1. **Sentiment Analysis**:  
       Ever wondered how passengers *really* feel about their flights?  
       Our AI dives into reviews and tells you if the vibes are **positive**, **negative**, or just **neutral**.  
       Bonus: You get a cool pie chart to visualize the sentiment breakdown! 🥧  

    2. **Interactive Airline Logo Display**:  
       Curious about the airline's branding?  
       The app fetches and displays the airline's logo dynamically in the sidebar. 🖼️  

    3. **Flight Recommendation Model**:  
       Get a **thumbs-up** or **thumbs-down** on your flight, backed by AI.  
       Plus, see how confident the model is about its prediction. (Spoiler: It’s pretty confident.) 👍👎  

    4. **Data Visualization**:  
       Explore average ratings for airlines and seat types over time with interactive charts.  
       Check out flight origins and destinations on a dynamic map. 🌍  

    5. **Keyword Insights**:  
       Discover the top buzzwords from reviews with a visually stunning word cloud.  
       (Yes, it’s as cool as it sounds.) ☁️  

    6. **Downloadable Predictions**:  
       Want to brag about your data-driven travel decisions?  
       Save your predictions as a PDF and share them with friends. 📄  

    ### How it works:
    1. **Step 1**: Enter your flight details and rate your experience (or let the chatbot do the heavy lifting).  
    2. **Step 2**: Explore real-time visualizations of reviews and ratings.  
    3. **Step 3**: Hit the **big blue button** to get your **seal of approval** prediction. 👍👎  

    ### Why use it?  
    - Help your friends avoid bad flights (and earn their eternal gratitude).  
    - Understand what matters most to passengers (hint: it’s not just the food).  
    - Stay informed with real-time insights into airline performance.  

    ### Ready to predict and explore? Let’s go! 🚀  
    P.S.: The team (me) 👩‍💻 is constantly working on new features to make your experience even better.  
    Stay tuned for more awesomeness! 🔥  
    """)
    #Working on changing the airports abbreviations in origin, destination and transit into city names to implement them within the model and use a more complete map as visualization
    
with tab_visuals:
    #Filtering the data set based on airline and seat type chosen by the user
    filtered_data_airline=data[data['Airline Name']==airline_choice]
    filtered_data=filtered_data_airline[(filtered_data_airline['Seat Type']==seat_type_choice)]

    #Transforming the overall rating column into numeric to perform a mean and extract the airline's average overall rating
    filtered_data_airline['Overall_Rating'] = pd.to_numeric(filtered_data_airline['Overall_Rating'], errors='coerce')
    overall_average=filtered_data_airline['Overall_Rating'].mean(skipna=True)
    
    #    -----------------------
    #           First Row      
    #    -----------------------
    #Implementing columns for a better user experience in visualizing the data displayed
    col1,col2=st.columns([2,2])

    with col1:
        st.subheader(f'{airline_choice}')
        #Visualizing overall average of the airline out of 10
        st.subheader(f'{overall_average:.2f}/10 ⭐️')

        #Adding a space for visual purposes
        st.markdown("\n")
        #Displaying all features selected average rating based on seat type
        st.markdown(f"**Average Ratings for :blue[{seat_type_choice}]**")

        #Creating the columns for which the average rating based on seat type is to be calculated and displayed
        cols_to_average=['Seat Comfort', 'Cabin Staff Service', 'Food & Beverages',
            'Ground Service', 'Inflight Entertainment', 'Wifi & Connectivity']
        average_ratings = filtered_data[cols_to_average].mean()
        
        #Creating the start visulaization for the average ratings
        def rating_to_stars(rating):
            if math.isnan(rating):
                return 'Not available'
            full_stars = math.floor(rating)
            half_star = rating - full_stars >= 0.5
            stars = "★" * full_stars
            if half_star:
                stars += "⯨"
            stars = stars.ljust(5, "☆")
            return stars

        for col in cols_to_average:
            if col in average_ratings:
                star_rating = rating_to_stars(average_ratings[col])
                st.markdown(f'_{col}_: {star_rating} ({average_ratings[col]:.2f})')

    with col2:
        #Creating a bar chart visualization to showcase the filtered feature's average rating over the available time period
        scoring_choice=st.selectbox(label='',options=['Seat Comfort','Cabin Staff Service','Food & Beverages','Ground Service','Inflight Entertainment','Wifi & Connectivity','Value For Money'])
        data_visual=filtered_data
        #Tranforming the Date flown column into date format to then extract its year and month
        data_visual['Date Flown'] = pd.to_datetime(data_visual['Date Flown'])
        data_visual['Month-Year']=data_visual['Date Flown'].dt.to_period('M').astype(str)
        average_per_month = data_visual.groupby('Month-Year')[scoring_choice].mean().reset_index()
        average_per_month = average_per_month.sort_values('Month-Year')
        st.markdown(f'**{scoring_choice} Time Series**')
        st.bar_chart(data=average_per_month, x='Month-Year', y=scoring_choice, y_label='Average Rating',x_label='Date', color='#0C5BA0')
    
    #    -----------------------
    #          Second Row      
    #    -----------------------

    #Implementing a map visualizationt to showcase the origin and destination of the flights
    geolocator = Nominatim(user_agent="streamlit_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=10)

    #Implementing a function to get coordinates for a given city
    def get_coordinates(city):
        """Gets the coordinates for a city, using the database cache if available."""
        # Check if the coordinates are already in the database
        latitude, longitude = get_coordinates_from_db(city)
        if latitude is not None and longitude is not None:
            return latitude, longitude

        # If not cached, fetch the coordinates using geopy
        try:
            location = geocode(city)
            if location:
                latitude, longitude = location.latitude, location.longitude
                # Save the coordinates to the database
                save_coordinates_to_db(city, latitude, longitude)
                return latitude, longitude
            else:
                return None, None
        except Exception as e:
            st.error(f"Error fetching coordinates for {city}: {e}")
            return None, None
        
    # Adding the coordinates to the dataset for the filtered data
    filtered_data['Origin_Coordinates'] = filtered_data['Origin'].apply(lambda x: get_coordinates(x))
    filtered_data['Destination_Coordinates'] = filtered_data['Destination'].apply(lambda x: get_coordinates(x))
    filtered_data['Origin_Latitude'], filtered_data['Origin_Longitude'] = zip(*filtered_data['Origin_Coordinates'])
    filtered_data['Destination_Latitude'], filtered_data['Destination_Longitude'] = zip(*filtered_data['Destination_Coordinates'])

    # Filter out rows with missing coordinates
    filtered_data = filtered_data.dropna(subset=['Origin_Latitude', 'Origin_Longitude', 'Destination_Latitude', 'Destination_Longitude'])

    # Check if there are valid rows to display on the map
    if not filtered_data.empty:
        map_center = [filtered_data['Origin_Latitude'].mean(), filtered_data['Origin_Longitude'].mean()]
        m = folium.Map(location=map_center, zoom_start=2)

        for _, row in filtered_data.iterrows():
            # Adding the origin marker
            folium.CircleMarker(
                location=[row['Origin_Latitude'], row['Origin_Longitude']],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m)

            # Adding the destination marker and line
            folium.CircleMarker(
                location=[row['Destination_Latitude'], row['Destination_Longitude']],
                radius=5,
                color='green',
                fill=True,
                fill_color='green'
            ).add_to(m)
            folium.PolyLine(
                locations=[[row['Origin_Latitude'], row['Origin_Longitude']],
                        [row['Destination_Latitude'], row['Destination_Longitude']]],
                color='gray',
                weight=1,
                opacity=0.5
            ).add_to(m)

        # Displaying the map in Streamlit
        st_folium(m, width=700, height=300)
    else:
        st.warning("No valid coordinates available to display on the map.")
    st.markdown('Destination and Origin of the flights, color coded <span style="color:green;font-style:italic;">green</span> and <span style="color:blue;font-style:italic;">blue</span> respectively.', unsafe_allow_html=True)

    #    -----------------------
    #          Third Row      
    #    -----------------------
    st.text('')
    st.subheader('Sentiment Analysis')
    col4, col5 =st.columns([2,2])
    
    with col4:
        #Utilizing the LLM to implement a sentiment analysis for each review
        def analyze_sentiment(row):
            unique_id = row['Unique_ID']
            review = row['Review']

            # Check if the sentiment is already in the database
            cached_sentiment = get_sentiment_from_db(unique_id)
            if cached_sentiment:
                return cached_sentiment

            # If not cached, analyze sentiment using OpenAI
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis assistant."},
                        {"role": "user", "content": f"""
                                Analyze the sentiment of the following text: '{review}'.
                                Respond with 'positive', 'negative', or 'neutral' only.
                                
                                Example:
                                'The flight was amazing, the staff were friendly, and the seats were comfortable.'
                                positive

                                'The flight was delayed, the food was terrible, and the seats were cramped.'
                                negative

                                'The flight was okay, nothing special to mention.'
                                neutral

                                Now analyze the sentiment of the given text."""}
                    ]
                )
                sentiment = response['choices'][0]['message']['content'].strip().lower()
                # Save the result to the database
                save_sentiment_to_db(unique_id, sentiment)
                return sentiment
            except openai.error.RateLimitError:
                st.warning("Rate limit reached. Waiting for 20 seconds before retrying...")
                time.sleep(20)
                return analyze_sentiment(row)
            except Exception as e:
                st.error(f"Error analyzing sentiment with OpenAI: {e}")
                return "unknown"

        filtered_data['Sentiment'] = filtered_data.apply(analyze_sentiment, axis=1)
        sentiment_counts=filtered_data['Sentiment'].value_counts()
        sentiment_counts=sentiment_counts.reindex(['positive','negative','neutral'],fill_value=0)
        fig, ax= plt.subplots()
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#F0FFFF','#89CFF0','#A7C7E7'],
            textprops={'color': 'black'}
        )
        ax.axis('equal')
        st.pyplot(fig)
    
    with col5:
        # Define the counts for each sentiment
        total_reviews = len(filtered_data)
        positive_reviews = sentiment_counts.get('positive', 0)
        neutral_reviews = sentiment_counts.get('neutral', 0)
        negative_reviews = sentiment_counts.get('negative', 0)

        # Define the HTML and CSS for the layout
        col5_style = f"""
        <style>
            .col5-container {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%; /* Ensure it takes the full height of the column */
            }}
            .col5-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 10px;
                background-color: #0047AB; /* Blue background */
                padding: 10px;
                border-radius: 10px;
            }}
            .col5-box {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                background-color: #0047AB; /* Blue background */
                color: white; /* White text */
                font-size: 20px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }}
            .col5-box p {{
                margin: 0;
                font-size: 14px;
                font-weight: normal;
            }}
        </style>
        <div class="col5-container">
            <div class="col5-grid">
                <div class="col5-box">
                    <div>{total_reviews}</div>
                    <p>Total Reviews</p>
                </div>
                <div class="col5-box">
                    <div>{positive_reviews}</div>
                    <p>Positive Reviews</p>
                </div>
                <div class="col5-box">
                    <div>{neutral_reviews}</div>
                    <p>Neutral Reviews</p>
                </div>
                <div class="col5-box">
                    <div>{negative_reviews}</div>
                    <p>Negative Reviews</p>
                </div>
            </div>
        </div>
        """

        # Render the styled HTML in Streamlit
        st.markdown(col5_style, unsafe_allow_html=True)

    #Visualizing the top 10 keywords, color-coded by sentiment as a wordcloud implementing mathplot
    # Define a custom color function for shades of blue
    def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return f"hsl({random.randint(200, 240)}, 100%, {random.randint(30, 70)}%)"

    st.subheader('Top 10 Keywords')
    all_reviews = ' '.join(filtered_data['Review'].tolist())
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=blue_color_func, 
        stopwords=STOPWORDS
    ).generate(all_reviews)

    # Display the word cloud
    plt.figure(figsize=(15, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    #    -----------------------
    #          Fourth Row      
    #    -----------------------

    #Displaying all data for the airline and seat type selected
    st.subheader(f'Data available for {airline_choice}')
    #filtering the dataset colusmn to filter out the unique_id and not show it in the dataframe
    columns_to_display = [col for col in filtered_data.columns if col != 'Unique_ID']
    st.dataframe(filtered_data[columns_to_display])

with tab_predictions:
    # Implementing columns for easier user input of the required data to predict Recommend or No Recommend
    col3, col4 = st.columns(2)
    with col3:
        airline_choice = st.selectbox('**What airline did you travel?**', airlines, placeholder='Select airline...')
        seat_type_choice = st.selectbox('**What seat did you have?**', seat_types, placeholder='Select seat type...')
        type_traveller_choice = st.selectbox('**What traveler are you?**', data['Type Of Traveller'].unique(), placeholder='Select traveler type...')
        overall_rating_choice = st.selectbox('**What overall rating would you give the flight?**', range(1, 11, 1))
        seat_comfort_choice = st.selectbox('**How was the seat comfort?**', range(1, 6, 1))
    with col4:
        cabin_staff_choice = st.selectbox('**How was the cabin staff service?**', range(1, 6, 1))
        food_choice = st.selectbox('**How was the food & beverages?**', range(1, 6, 1))
        ground_service_choice = st.selectbox('**How was the ground service?**', range(1, 6, 1))
        entrateinment_choice = st.selectbox('**How was the entertainment?**', range(1, 6, 1))
        wifi_choice = st.selectbox('**How was the wifi onboard?**', range(1, 6, 1))
        money_choice = st.selectbox('**What would you rate the value for money?**', range(1, 6, 1))

    # Add custom CSS for the button
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #0C5BA0; /* Blue background */
            color: white; /* White text */
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #004080; /* Darker blue on hover */
            color: white !important; /* Keep text white on hover */
        }
        </style>
    """, unsafe_allow_html=True)

    # Creating a button such that the model is called to predict only when all input data has been chosen by the user
    if st.button('Would I recommend this flight to a friend?', type='secondary'):
        # Transforming all data inputted by the user into a DataFrame to feed the model
        input_data = pd.DataFrame([{
            'Airline Name': airline_choice,
            'Seat Type': seat_type_choice,
            'Type Of Traveller': type_traveller_choice,
            'Overall_Rating': overall_rating_choice,
            'Seat Comfort': seat_comfort_choice,
            'Cabin Staff Service': cabin_staff_choice,
            'Food & Beverages': food_choice,
            'Ground Service': ground_service_choice,
            'Inflight Entertainment': entrateinment_choice,
            'Wifi & Connectivity': wifi_choice,
            'Value For Money': money_choice
        }])

        # Predicting
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Displaying the prediction result
        if prediction == 1:
            st.success('Looks like a good flight - I would recommend it to my friends! 😊')
        else:
            st.warning('Hmm, I think your friends could find something better. 😕')

        st.write(f'Based on the data, I’m {prob:.2%} sure this flight is recommendable.')

        # Updating the session state with the user's prediction
        try:
            st.session_state.predictions.append({
                'Airline Name': airline_choice,
                'Seat Type': seat_type_choice,
                'Type Of Traveller': type_traveller_choice,
                'Overall_Rating': overall_rating_choice,
                'Seat Comfort': seat_comfort_choice,
                'Cabin Staff Service': cabin_staff_choice,
                'Food & Beverages': food_choice,
                'Ground Service': ground_service_choice,
                'Inflight Entertainment': entrateinment_choice,
                'Wifi & Connectivity': wifi_choice,
                'Value For Money': money_choice,
                'Prediction': 'Recommend' if prediction == 1 else 'Not Recommend',
                'Probability': f"{prob:.2%}"
            })
        except Exception as e:
            st.error(f"An error occurred while saving the prediction: {e}")

    # Add custom CSS for all buttons, including the download button
    st.markdown("""
        <style>
        div.stButton > button, div.stDownloadButton > button {
            background-color: #0C5BA0; /* Blue background */
            color: white; /* White text */
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background-color: #004080; /* Darker blue on hover */
            color: white !important; /* Keep text white on hover */
        }
        </style>
    """, unsafe_allow_html=True)

    # Displaying the user's previous predictions
    if st.session_state.predictions:
        st.subheader('A glimpse into your past predictions 🔮')
        # Convert predictions to a DataFrame for display
        predictions_df = pd.DataFrame(st.session_state.predictions)
        st.dataframe(predictions_df)

        # Generate the PDF content
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', size=12)
        pdf.cell(200, 10, txt='Your Predictions', ln=True, align='C')

        for i, prediction in enumerate(st.session_state.predictions, start=1):
            pdf.cell(200, 10, txt=f"Prediction {i}:", ln=True, align='L')
            for key, value in prediction.items():
                pdf.cell(200, 10, txt=f'{key}: {value}', ln=True)
            # Adding a blank line between predictions for easier reading
            pdf.cell(200, 10, txt=' ', ln=True)

        # Generate the PDF as a binary string
        pdf_output = pdf.output(dest='S').encode('latin1')

        # Provide a download button for the PDF
        st.download_button(
            label='Download Predictions',
            data=pdf_output,
            file_name='predictions.pdf',
            mime='application/pdf'
        )