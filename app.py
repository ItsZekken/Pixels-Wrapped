import streamlit as st
import json
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from collections import Counter
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import gaussian_kde

# --- NLTK Setup (Download resources if not present) ---
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Data Loading and Processing ---

def load_data(filepath):
    """Loads data from a JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def calculate_average_scores(data):
    """Calculates the average mood score for each entry."""
    for entry in data:
        if 'scores' in entry and entry['scores']:
            entry['average_score'] = sum(entry['scores']) / len(entry['scores'])
        else:
            entry['average_score'] = 0  # Assign 0 if no scores are available
    return data

def filter_by_year(data, year):
    """Filters data by the specified year."""
    filtered_data = []
    for entry in data:
        if isinstance(entry.get("date"), str):
            try:
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
                if entry_date.year == year:
                    filtered_data.append(entry)
            except ValueError:
                continue
    return filtered_data

# --- Mood Analysis ---

def analyze_mood_data(data, year):
    """Analyzes mood data to extract key metrics and top tags."""
    filtered_data = filter_by_year(data, year)
    mood_scores = [entry['average_score'] for entry in filtered_data if entry['average_score'] is not None]
    
    if not mood_scores:
        return {}, {}

    avg_mood = sum(mood_scores) / len(mood_scores)
    max_mood = max(mood_scores)
    min_mood = min(mood_scores)

    tag_counts = {} # Dictionary to store tag counts by category

    for entry in filtered_data:
        if 'tags' in entry:
            for tag in entry['tags']:
                tag_type = tag['type']
                if tag_type not in tag_counts:
                    tag_counts[tag_type] = Counter()
                tag_counts[tag_type].update(tag['entries'])

    top_tags_by_category = {}
    for tag_type, counts in tag_counts.items():
        top_tags_by_category[tag_type] = counts.most_common(5)

    
    notes = [entry['note'] for entry in filtered_data if 'note' in entry and entry['note']]

    return {
        'average_mood': avg_mood,
        'max_mood': max_mood,
        'min_mood': min_mood,
    }, {
        'top_tags_by_category': top_tags_by_category,
        'notes': notes,
        'all_data': filtered_data
    }

# --- I. New Metrics and Stats ---

def calculate_streaks(data, year):
    """Calculates the longest positive/negative streaks and the current streak."""
    filtered_data = filter_by_year(data, year)
    if not filtered_data:
        return None

    df = pd.DataFrame(filtered_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Calculate streaks
    df['mood_direction'] = np.sign(df['average_score'] - df['average_score'].mean())
    df['streak_start'] = df['mood_direction'].ne(df['mood_direction'].shift())
    df['streak_id'] = df['streak_start'].cumsum()
    df['streak_length'] = df.groupby('streak_id')['streak_id'].transform('count')

    # Filter streaks that are positive or negative only
    positive_streaks = df[df['mood_direction'] > 0]['streak_length']
    negative_streaks = df[df['mood_direction'] < 0]['streak_length']
    
    longest_positive_streak = positive_streaks.max() if not positive_streaks.empty else 0
    longest_negative_streak = negative_streaks.max() if not negative_streaks.empty else 0

    # Current streak
    current_streak_id = df['streak_id'].iloc[-1]
    current_streak = df[df['streak_id'] == current_streak_id]['streak_length'].iloc[-1]

    return {
        'longest_positive_streak': longest_positive_streak,
        'longest_negative_streak': longest_negative_streak,
        'current_streak': current_streak,
        'current_streak_type': "positive" if df['mood_direction'].iloc[-1] > 0 else "negative" if df['mood_direction'].iloc[-1] < 0 else "neutral"
    }

def compare_with_previous_year(data, year):
    """Compares mood metrics with the previous year."""
    current_year_data = analyze_mood_data(data, year)[0]
    previous_year_data = analyze_mood_data(data, year - 1)[0]

    if not previous_year_data:
        return None

    comparison = {}
    if current_year_data and previous_year_data:
      comparison['average_mood_change'] = current_year_data['average_mood'] - previous_year_data['average_mood']
    
    current_year_tags = analyze_mood_data(data, year)[1]
    previous_year_tags = analyze_mood_data(data, year -1)[1]
    
    if current_year_tags and previous_year_tags:
        comparison['tag_differences'] = {}
        for tag_type in current_year_tags['top_tags_by_category'].keys():
          
            current_tags = Counter(dict(current_year_tags['top_tags_by_category'].get(tag_type, [])))
            previous_tags = Counter(dict(previous_year_tags['top_tags_by_category'].get(tag_type, [])))
            
            tag_diff = current_tags - previous_tags
            if tag_diff:
                comparison['tag_differences'][tag_type] = tag_diff.most_common()

    return comparison

def create_comparative_heatmap(data, year):
    """Creates a comparative heatmap for the current and previous year."""
    current_year_heatmap = create_calendar_heatmap(data, year)
    previous_year_heatmap = create_calendar_heatmap(data, year - 1)

    if not previous_year_heatmap:
        return None
    
    return current_year_heatmap, previous_year_heatmap

def calculate_tag_correlations(data, year):
    """Calculates correlations between tags."""
    filtered_data = filter_by_year(data, year)

    if not filtered_data:
        return None, None
    
    all_tags = set()
    for entry in filtered_data:
        if 'tags' in entry:
            for tag in entry['tags']:
                all_tags.update(tag['entries'])

    if not all_tags:
        return None, None
    
    all_tags = sorted(list(all_tags))

    # Crear el DataFrame con listas en lugar de sets
    tag_matrix = pd.DataFrame(0, index=all_tags, columns=all_tags)

    for entry in filtered_data:
        if 'tags' in entry:
            tags_in_entry = set()
            for tag in entry['tags']:
                tags_in_entry.update(tag['entries'])
            for tag1 in tags_in_entry:
                for tag2 in tags_in_entry:
                    if tag1 != tag2:
                        tag_matrix.loc[tag1, tag2] += 1

    # Calcular probabilidades condicionales
    conditional_probs = {}
    for tag1 in all_tags:
        for tag2 in all_tags:
            if tag1 != tag2:
                count_tag1 = tag_matrix.loc[tag1, :].sum()
                count_both = tag_matrix.loc[tag1, tag2]
                if count_tag1 > 0:
                    conditional_probs[f"P({tag2}|{tag1})"] = count_both / count_tag1
    
    # Filtrar probabilidades relevantes
    relevant_probs = {k: v for k, v in conditional_probs.items() if v >= 0.4 and v < 1}

    return tag_matrix, relevant_probs

def analyze_notes_sentiment(data, year):
    """Analyzes the sentiment of notes."""
    filtered_data = filter_by_year(data, year)
    notes = [entry['note'] for entry in filtered_data if 'note' in entry and entry['note']]

    if not notes:
        return None, None

    # Combine all notes into a single string
    text = " ".join(notes)

    # Remove stopwords
    stop_words = set(STOPWORDS)
    stop_words.update(nltk.corpus.stopwords.words('spanish'))
    stop_words.update(nltk.corpus.stopwords.words('english'))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        stopwords=stop_words,
        background_color='white'
    ).generate(text)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(note)['compound'] for note in notes]
    average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    return wordcloud, average_sentiment_score

def analyze_special_days(data, year):
    """Identifies the best and worst days and compares weekends vs. weekdays."""
    filtered_data = filter_by_year(data, year)
    if not filtered_data:
        return None

    df = pd.DataFrame(filtered_data)
    df['date'] = pd.to_datetime(df['date'])

    best_day = df.loc[df['average_score'].idxmax()]
    worst_day = df.loc[df['average_score'].idxmin()]

    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    weekend_mood = df[df['day_of_week'] >= 5]['average_score'].mean()
    weekday_mood = df[df['day_of_week'] < 5]['average_score'].mean()

    return {
        'best_day': best_day,
        'worst_day': worst_day,
        'weekend_mood': weekend_mood,
        'weekday_mood': weekday_mood
    }

# --- II. Enhanced and New Visualizations ---

def create_mood_evolution_chart(data, year, selected_tag=None):
    """Creates a line chart showing mood evolution with trend line and smoothed curve."""
    filtered_data = filter_by_year(data, year)
    if not filtered_data:
      return None

    df = pd.DataFrame(filtered_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Filter by tag if selected
    if selected_tag:
        df['tag_present'] = df['tags'].apply(lambda tags: any(selected_tag in tag['entries'] for tag in tags if 'entries' in tag))
        df = df[df['tag_present']]

    if df.empty:
        return None
    
    fig = go.Figure()

    # Datos originales
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['average_score'],
        mode='lines+markers',
        name='Mood Score',
        marker=dict(color='#175a77', size=6),
        line=dict(color='#174c64', width=1)
    ))

    # Curva suavizada
    window_size = max(3, len(df) // 20)  # Tamaño de ventana adaptativo
    smoothed_scores = df['average_score'].rolling(window=window_size, center=True).mean()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=smoothed_scores,
        mode='lines',
        name='Smoothed Score',
        line=dict(color='#48d1cc', width=3)
    ))

    # Línea de tendencia
    z = np.polyfit(range(len(df)), df['average_score'], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=p(range(len(df))),
        mode='lines',
        name='Trend Line',
        line=dict(color='white', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"<b>Mood Evolution {year}</b>",
        xaxis_title="Date",
        yaxis_title="Mood Score",
        font=dict(family="Arial", size=14, color="#333"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a1a',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        )
    )

    return fig

def create_radial_chart(tag_data, title, color):
    """Creates a radial chart for top tags."""
    
    if not tag_data:
        return None

    tags, counts = zip(*tag_data)
    
    fig = go.Figure(data=go.Scatterpolar(
      r=counts,
      theta=tags,
      fill='toself',
      marker=dict(color=color)
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, max(counts) + max(counts)*0.1]
        )),
      showlegend=False,
      title=f"<b>{title}</b>",
      font=dict(family="Arial", size=14, color="#333"),
      plot_bgcolor='rgba(0,0,0,0)',
      paper_bgcolor='#1a1a1a'
    )

    return fig

# --- Visualizations ---

def create_calendar_heatmap(data, year):
    """Generates a calendar heatmap of mood scores."""
    filtered_data = filter_by_year(data, year)
    dates = []
    scores = []
    notes = []

    for entry in filtered_data:
        if isinstance(entry.get("date"), str):
            try:
                entry_date = datetime.strptime(entry['date'], '%Y-%m-%d')
                if entry['average_score'] is not None:
                    dates.append(entry_date)
                    scores.append(entry['average_score'])
                    notes.append(entry.get('note', ''))  # Add note for hover data
            except ValueError:
                continue

    if not dates or not scores:
        return None

    df = pd.DataFrame({'Date': dates, 'Score': scores, 'Note': notes})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Weekday'] = df['Date'].dt.day_name()

    fig = go.Figure(data=go.Heatmap(
        x=df['Day'],
        y=df['Month'],
        z=df['Score'],
        colorscale='RdYlGn',
        text=df['Note'],  # Show note on hover
        hovertemplate="<b>Date:</b> %{x}/%{y}/" + str(year) + "<br><b>Score:</b> %{z:.1f}<br><b>Note:</b> %{text}<extra></extra>",
        hoverongaps=False,
    ))
    
    fig.update_yaxes(categoryorder='array', categoryarray=list(range(1, 13)))

    fig.update_layout(
        title=f"<b>Mood Heatmap {year}</b>",
        xaxis=dict(title='Day', showgrid=True, tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(title='Month', tickvals=list(range(1, 13)),
                   ticktext=[
                       "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                   ]),
        font=dict(family="Arial", size=14, color="#333"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a1a'
    )
    return fig

def create_mood_distribution_chart(data, year):
    """Creates a KDE plot showing the distribution of mood scores."""
    filtered_data = filter_by_year(data, year)
    mood_scores = [entry['average_score'] for entry in filtered_data if entry['average_score'] is not None]

    if not mood_scores:
        return None

    # Crear un array de puntos para la curva suave
    kde_x = np.linspace(min(mood_scores), max(mood_scores), 100)
    kde = gaussian_kde(mood_scores)
    kde_y = kde(kde_x)

    fig = go.Figure()
    
    # Añadir la curva de densidad
    fig.add_trace(go.Scatter(
        x=kde_x,
        y=kde_y,
        mode='lines',
        fill='tozeroy',
        name='Density',
        line=dict(color='#008080', width=3),
        fillcolor='rgba(0, 128, 128, 0.3)'
    ))

    fig.update_layout(
        title=f"<b>Mood Distribution {year}</b>",
        xaxis_title="Mood Score",
        yaxis_title="Density",
        font=dict(family="Arial", size=14, color="#333"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a1a',
        showlegend=False
    )
    
    return fig
  
def create_top_tags_bar_chart(tag_data, tag_type, title, color):
    """Creates a horizontal bar chart for top tags."""
    
    if not tag_data:
        st.write(f"No data found for {tag_type}.")
        return None

    tags, counts = zip(*tag_data)
    
    fig = px.bar(
        x=counts,
        y=tags,
        orientation='h',
        title=f"<b>{title}</b>",
        labels={'x': 'Count', 'y': tag_type},
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="",
        font=dict(family="Arial", size=14, color="#333"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='#1a1a1a'
    )
    
    return fig

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Pixels Wrapped", page_icon="😊", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: white;
        }
        .st-emotion-cache-18ni7ap {
            background-color: #2d4059;
        }
        div[data-testid="stMetricValue"] {
            color: white !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #cccccc !important;
        }
        h1, h2, h3, p {
            color: white !important;
        }
        .streamlit-expanderHeader {
            color: white !important;
        }
        div.row-widget.stSelectbox > div {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌈 Pixels Wrapped - Your Year in Moods")
    st.markdown(
        """
        Welcome to your personalized Pixels Wrapped! Upload your Pixels JSON file and take a colorful journey 
        through your year. Discover your mood trends, top emotions, most frequent activities, and much more.
        """
    )

    uploaded_file = st.file_uploader("📤 Upload Your Pixels JSON File", type="json")

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            data = calculate_average_scores(data)
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
            return
    else:
        st.info("☝️ Please upload your Pixels data to get started.")
        return

    year = st.selectbox("**Select Year**", options=list(range(2020, 2025)), index=4)

    mood_summary, top_tags_and_data = analyze_mood_data(data, year)
    top_tags = top_tags_and_data['top_tags_by_category']

    if not mood_summary and not top_tags:
        st.warning(f"😔 No data available for the selected year: {year}")
        return

    # --- Mood Summary ---
    
    st.markdown(f"## ✨ Mood Summary - {year}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Mood", value=f"{mood_summary.get('average_mood', 'N/A'):.2f}", delta=None)
    with col2:
        st.metric("Happiest Day", value=f"{mood_summary.get('max_mood', 'N/A'):.2f}", delta=None)
    with col3:
        st.metric("Toughest Day", value=f"{mood_summary.get('min_mood', 'N/A'):.2f}", delta=None)

    # --- Mood Distribution Chart ---
    mood_distribution_fig = create_mood_distribution_chart(data, year)
    if mood_distribution_fig:
        st.plotly_chart(mood_distribution_fig, use_container_width=True, key="mood_dist")

    # --- I. New Metrics and Stats ---

    st.markdown("## 📊 Advanced Insights")

    # --- Streaks ---
    st.markdown("### 🏃 Streaks")
    st.markdown("#### Discover your longest positive and negative streaks, and see how your current streak is going.")
    streaks = calculate_streaks(data, year)
    if streaks:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div style='border: 2px solid #008080; border-radius: 10px; padding: 10px; margin-bottom: 20px;'>
                    <h4 style='text-align: center; color: #008080;'>Longest Positive Streak</h4>
                    <div style='display: flex; justify-content: center; align-items: center; height: 80px;'>
                        <span style='font-size: 2.5em; color: white; font-weight: bold;'>{streaks['longest_positive_streak']} days</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div style='border: 2px solid #ff6b6b; border-radius: 10px; padding: 10px; margin-bottom: 20px;'>
                    <h4 style='text-align: center; color: #ff6b6b;'>Longest Negative Streak</h4>
                    <div style='display: flex; justify-content: center; align-items: center; height: 80px;'>
                        <span style='font-size: 2.5em; color: white; font-weight: bold;'>{streaks['longest_negative_streak']} days</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div style='border: 2px solid #ffcc66; border-radius: 10px; padding: 10px; margin-bottom: 20px;'>
                    <h4 style='text-align: center; color: #ffcc66;'>Current Streak</h4>
                    <div style='display: flex; justify-content: center; align-items: center; height: 80px;'>
                        <span style='font-size: 2.5em; color: white; font-weight: bold;'>{streaks['current_streak']} days</span>
                    </div>
                    <p style='text-align: center; color: white;'>
                        <small>
                            This streak is currently: <span style='color: #ffcc66; font-weight: bold;'>{streaks['current_streak_type'].capitalize()}</span>
                        </small>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No streaks data available.", icon="ℹ️")

    # --- Calendar Heatmap ---
    st.markdown("## 🗓️ Mood Calendar Heatmap")
    heatmap_fig = create_calendar_heatmap(data, year)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True, key="calendar_heatmap")
    else:
        st.warning(f"No data to show on Mood Heatmap for: {year}")

    # --- Notes Sentiment Analysis ---
    st.markdown("### 📝 Notes Analysis")
    wordcloud, avg_sentiment_score = analyze_notes_sentiment(data, year)
    if wordcloud and avg_sentiment_score is not None:
        st.markdown("#### Word Cloud")
        st.image(wordcloud.to_array(), use_column_width=True)
        st.metric("Average Sentiment Score of Notes", f"{avg_sentiment_score:.2f}")
    else:
        st.write("Notes analysis not available.")

    # --- Special Days ---
    st.markdown("### 🎉 Special Days")
    special_days = analyze_special_days(data, year)
    if special_days:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Best Day**")
            st.write(f"Date: {special_days['best_day']['date'].strftime('%Y-%m-%d')}")
            st.write(f"Score: {special_days['best_day']['average_score']:.2f}")
            if 'note' in special_days['best_day']:
                st.write(f"Note: {special_days['best_day']['note']}")
        
        with col2:
            st.markdown("**Worst Day**")
            st.write(f"Date: {special_days['worst_day']['date'].strftime('%Y-%m-%d')}")
            st.write(f"Score: {special_days['worst_day']['average_score']:.2f}")
            if 'note' in special_days['worst_day']:
                st.write(f"Note: {special_days['worst_day']['note']}")
        
        with col3:
            st.markdown("**Weekend vs. Weekday Mood**")
            st.write(f"Weekend Mood: {special_days['weekend_mood']:.2f}")
            st.write(f"Weekday Mood: {special_days['weekday_mood']:.2f}")
    else:
        st.write("Special days analysis not available.")

    # --- II. Enhanced and New Visualizations ---

    # --- Top Tags ---
    st.markdown("## 🔥 Top Tags")

    num_tag_types = len(top_tags)
    cols = st.columns(num_tag_types)
    
    color_palette = ["#ff6b6b", "#4ecdc4", "#ffcc66", "#f787be", "#66cc99", "#008080"]

    for i, (tag_type, tag_data) in enumerate(top_tags.items()):
        with cols[i]:
            color = color_palette[i % len(color_palette)]
            tags_fig = create_top_tags_bar_chart(tag_data, tag_type, f"Top {tag_type} - {year}", color)
            if tags_fig:
                st.plotly_chart(tags_fig, use_container_width=True, key=f"{tag_type}_bar")

    # --- Mood Evolution Chart ---
    st.markdown("## 📈 Mood Evolution")

    # Get all unique tags
    all_tags = set()
    for entry in top_tags_and_data['all_data']:
        if 'tags' in entry:
            for tag in entry['tags']:
                all_tags.update(tag['entries'])

    # Filter for mood evolution chart
    selected_tag = st.selectbox("Filter by Tag (Optional)", options=[""] + list(all_tags))

    mood_evolution_fig = create_mood_evolution_chart(data, year, selected_tag)
    if mood_evolution_fig:
        st.plotly_chart(mood_evolution_fig, use_container_width=True, key="mood_evolution")
    else:
        st.write("Mood evolution chart not available for the selected tag.")
        
    # --- Radial Chart for Top Tags ---
    st.markdown("## 🥏 Radial Charts for Top Tags")

    num_tag_types = len(top_tags)
    cols = st.columns(num_tag_types)

    color_palette = ["#ff6b6b", "#4ecdc4", "#ffcc66", "#f787be", "#66cc99", "#008080"]

    for i, (tag_type, tag_data) in enumerate(top_tags.items()):
        with cols[i]:
            color = color_palette[i % len(color_palette)]
            
            radial_fig = create_radial_chart(tag_data, f"Top {tag_type} - {year}", color)
            if radial_fig:
                radial_fig.update_layout(
                    plot_bgcolor="gray",
                    font=dict(color="white"),
                    polar=dict(
                        bgcolor="#1a1a1a",
                        radialaxis=dict(
                            tickfont=dict(color="lightgray"),
                            gridcolor="gray"
                        ),
                        angularaxis=dict(
                            tickfont=dict(color="lightgray"),
                            gridcolor="gray"
                        )
                    )
                )
                st.plotly_chart(radial_fig, use_container_width=True, key=f"{tag_type}_radial")

    # --- Notes Section ---
    if top_tags_and_data['notes']:
        with st.expander("📖 A glimpse into your notes..."):
            for note in top_tags_and_data['notes'][:5]:  # Display the first 5 notes
                st.write(note)

    st.markdown("---")
    st.markdown(
        """
        **About Pixels Wrapped**: This app is inspired by the concept of a 'year in review' but tailored for your personal mood tracking data from the Pixels app.
        It's a way to reflect on your year, understand your emotional landscape, and celebrate the journey you've had.
        """
    )

if __name__ == "__main__":
    main()