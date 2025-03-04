import streamlit as st
import pandas as pd
import joblib
import math
import re
#from geopy.geocoders import Nominatim
#import time

#Importing the predictive ML
model=joblib.load('airline_review_model.joblib')

data=pd.read_csv('Airline_review.csv')
data=data.dropna(subset=['Aircraft','Type Of Traveller','Seat Type','Route','Date Flown'])
#Imputing missing values for specific columns with the average grouped by airline name and seat type
data[['Seat Comfort', 'Cabin Staff Service','Food & Beverages','Ground Service','Inflight Entertainment','Wifi & Connectivity']] = data.groupby(['Airline Name', 'Seat Type'])[['Seat Comfort', 'Cabin Staff Service','Food & Beverages','Ground Service','Inflight Entertainment','Wifi & Connectivity']].transform(lambda x: x.fillna(x.mean()))
#Dropping all additional missing values
data=data.dropna()

#Extracting all possible values of Airline Name and Seat Type to input within the sidebar select box
airlines=data['Airline Name'].unique()
seat_types=data['Seat Type'].unique()
#Creating visuals to be filtered for each airline name and seat type
st.sidebar.title('Filter Options')
airline_choice=st.sidebar.selectbox('Airline:',airlines)
seat_type_choice=st.sidebar.selectbox('Seat Type:',seat_types)

#Splitting the data feature Route into origin, destination and transit if present
def clean_text(text):
    return re.sub(r'[^\w\s]', '', text) 

def split_route(route):
    parts=route.split(' to ',1)

    if len(parts)<2:
        return None,None,None
    Origin=clean_text(parts[0].strip())
    Destination=clean_text(parts[1].strip())

    if 'via' in Destination:
        Destination, Transit=Destination.split(' via ',1)
        Transit=clean_text(Transit.strip())
    else:
        Transit=None
    return Origin, Destination, Transit

data[['Origin', 'Destination', 'Transit']] = data['Route'].apply(lambda x: pd.Series(split_route(x)))

#Dropping columns with no value for the intended purpose
data=data.drop(columns=['Unnamed: 0','Review_Title','Review Date','Verified','Route','Aircraft'])

#The following is an approach to implement a map visualization within the tap_visuals but for the sake of running it was not implemented at the end because it takes time to run
#Extracting the latitude and longitute of each city in the origin and destination with geopy.geocoders to create a map visual
#geolocator = Nominatim(user_agent="streamlit_app")
#Defining a function to fetch the coordinates and then implementing it for each observation
#def get_coordinates(city):
#    location = geolocator.geocode(city, timeout=10)
#    time.sleep(1)
#    if location:
#        return location.latitude,location.longitude
#    else:
#        return None,None
#data['Origin_Latitude'] = data['Origin'].apply(lambda city: get_coordinates(city)[0])
#data['Origin_Longitude'] = data['Origin'].apply(lambda city: get_coordinates(city)[1])
#data['Destination_Latitude'] = data['Destination'].apply(lambda city: get_coordinates(city)[0])
#data['Destination_Longitude'] = data['Destination'].apply(lambda city: get_coordinates(city)[1])

st.title(':blue[Airline Reviews] :airplane:')

#Creating tap for a less crowded and more clean visualization of the data and model
tab_overview,tab_visuals,tab_predictions = st.tabs(["Overview","Visuals", "Predictions"])

with tab_overview:
    #Introducing the user to the features and use of the 'app' created
    st.markdown("""
    Welcome aboard! 🛫  
    Ever wondered if you should **recommend a flight to your friends**? This app helps you decide! 💡

    ##### What’s it about?
    Based on thousands of real airline reviews, our model predicts whether you’d recommend a flight to your friends — considering factors like seat comfort, food quality, cabin crew, entertainment, and WiFi. 📊

    ##### How it works:
    1. Enter your flight details and rate your experience.
    2. Hit the **big blue button** to get your **seal of approval** prediction. 👍👎

    ##### Stay updated:
    Along with your prediction, you’ll get access to:
    - **Average ratings** of airlines.
    - Real-time **reviews** from fellow travelers. 🌍📈

    ##### Why use it?
    - Help friends avoid bad flights.
    - Know what matters to passengers.
    - Stay in the loop with airline performance.

    Ready to predict and explore? Let’s go! 🚀
                
    P.S.: The team (me) 👩‍💻 is working on some new cool features ✨ to be incorporated soon... Stay tuned! 🔥
    """)
    #Working on changing the airports abbreviations in origin, destination and transit into city names to implement them within the model and use a more complete map as visualization
    
with tab_visuals:
    #Filtering the data set based on airline and seat type chosen by the user
    filtered_data_airline=data[data['Airline Name']==airline_choice]
    filtered_data=filtered_data_airline[filtered_data_airline['Seat Type']==seat_type_choice]

    #Transforming the overall rating column into numeric to perform a mean and extract the airline's average overall rating
    filtered_data_airline['Overall_Rating'] = pd.to_numeric(filtered_data_airline['Overall_Rating'], errors='coerce')
    overall_average=filtered_data_airline['Overall_Rating'].mean(skipna=True)
    
    #Implementing columns for a better user experience in visualizing the data displayed
    col1,col2=st.columns(2)

    with col1:
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

        st.subheader(f"{airline_choice}")
        #Visualizing overall average of the airline out of 10
        st.subheader(f'{overall_average:.2f}/10 ⭐️')

        #Adding space for visual purposes
        st.markdown("\n")
        #Displaying all features selected average rating based on seat type
        st.markdown(f"**Average Ratings for :blue[{seat_type_choice}]**")
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
    
    #Displaying Origin and Destination within a map visualization
    #st.divider()
    #dest=st.checkbox('Show the destinations')
    #if dest:
        #st.map(filtered_data[['Destination_Latitude','Destination_Longitude']])
    #else:
        #st.map(filtered_data[['Origin_Latitude','Origin_Longitude']])
    
    #Displaying all data for the airline and seat type selected
    st.divider()
    st.subheader(f'Data available for {airline_choice}')
    st.dataframe(filtered_data)

with tab_predictions:
    #Implementing columsn for an easier user input of the required data to predict Recommend or No Recommend
    col3,col4=st.columns(2)
    with col3:
        airline_choice=st.selectbox('**What airline did you travel?**',airlines,placeholder='Select airline...')
        seat_type_choice=st.selectbox('**What seat did you have?**',seat_types, placeholder='Selct seat type...')
        type_traveller_choice=st.selectbox('**What traveler are you?**',data['Type Of Traveller'].unique(),placeholder='Select traveler type...')
        overall_rating_choice=st.selectbox('**What overall rating would you give the flight?**',range(1,11,1))
        seat_comfort_choice=st.selectbox('**How was the seat comfort?**',range(1,6,1))
    with col4:
        cabin_staff_choice=st.selectbox('**How was the cabin staff service?**',range(1,6,1))
        food_choice=st.selectbox('**How was the food & beverages?**',range(1,6,1))
        ground_service_choice=st.selectbox('**How was the ground service?**',range(1,6,1))
        entrateinment_choice=st.selectbox('**How was the entertainment?**',range(1,6,1))
        wifi_choice=st.selectbox('**How was the wifi onboard?**',range(1,6,1))
        money_choice=st.selectbox('**What would you rate the value for money?**',range(1,6,1))

    #Creating a button such that the model is called to predict only when all input data has been chosen by the user
    if st.button('Would I recommend this flight to a friend? 👍', type='secondary'):
        #Transforming all data inputed by the user into a data frame to feed the model
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

        #Predicting
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success('Looks like a good flight - I would racommend it to my friends! 😊')
            #videos would make the prediction more interactive but make the running of the code slower. For now I did not include them but kept the code for others to do so if wanted.
            #st.video('https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW9sZ3VyNTE2eXIzcHFnYzVhcWh1YjY0MWVib2dlNm02dmIxcHBjdiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5C472t1RGNuq4/giphy.gif',loop=True)
        else:
            st.warning('Hmm, I think your friends could find something better. 😕')
            #st.video('https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnRpZHJmZ3k1YjU0d2UyNTQydGZ5dGszNTdhY3hjZXR0N2JneDZnNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/p3hZ9PbhVBO9gM6LoE/giphy.gif',loop=True)
        
        #Displaying the probability of predictions being 1 or 0 for users to decide their own threshold if wanting
        prob = model.predict_proba(input_data)[0][1]
        st.write(f'Based on the data, I’m {prob:.2%} sure this flight is recommendable.')

