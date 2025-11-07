# FIFA World Cup 2026 Finalist Prediction

## Overview

This is a machine learning application that predicts the top 2 finalists for the FIFA World Cup 2026. The system uses a Random Forest Classifier trained on team statistics including FIFA rankings, goals scored/conceded, average age, and win rates. The application features an interactive Streamlit interface with data visualizations and real-time predictions.

The project handles the unique challenge that only 28 of 48 teams have qualified for the tournament, implementing a simulation approach to generate data for the remaining 20 teams based on top FIFA rankings.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Decision:** Streamlit-based web interface
- **Rationale:** Streamlit provides rapid development of data-driven applications with minimal frontend code
- **Components:**
  - Interactive dashboard with configurable layout (`layout="wide"`)
  - Data visualizations using Plotly and Matplotlib/Seaborn
  - Cached data loading for performance optimization
  - Info panels for user guidance

### Machine Learning Pipeline

**Decision:** Scikit-learn Random Forest Classifier
- **Rationale:** Random Forests handle non-linear relationships well and provide feature importance insights for sports predictions
- **Features Used:**
  - FIFA_Ranking: Official team rankings
  - Goals_Scored: Offensive capability metric
  - Goals_Conceded: Defensive capability metric
  - Avg_Age: Team maturity indicator
  - Win_Rate: Historical performance measure
- **Training Approach:** Train-test split for model validation

**Data Strategy:**
- **Problem:** Only 28 of 48 teams have qualified
- **Solution:** Simulate remaining 20 teams from top 100 FIFA rankings with generated statistics
- **Alternative Considered:** Wait for all teams to qualify (dynamic updates)
- **Pros:** Allows immediate predictions and testing
- **Cons:** Simulated team data may not reflect actual qualifiers

### Data Management

**Decision:** In-memory data processing with caching
- **Rationale:** Small dataset size (48 teams) doesn't require persistent database
- **Implementation:** Streamlit's `@st.cache_data` decorator for performance
- **Data Structure:** Pandas DataFrames for tabular operations

**Qualified Teams Data:**
- 28 pre-defined qualified teams with real statistics
- Hardcoded in `generate_team_data()` function
- Includes top teams: Argentina, France, Brazil, England, etc.

### Visualization Strategy

**Decision:** Dual visualization library approach
- **Plotly:** Interactive charts for web interface (plotly.express, plotly.graph_objects)
- **Matplotlib/Seaborn:** Statistical visualizations (confusion matrix, classification reports)
- **Rationale:** Plotly provides interactivity for user engagement, while Matplotlib offers statistical analysis depth

## External Dependencies

### Python Libraries

**Data Processing & ML:**
- `pandas`: Dataframe operations and data manipulation
- `numpy`: Numerical computing and array operations
- `scikit-learn`: Machine learning algorithms and metrics
  - `RandomForestClassifier`: Main prediction model
  - `train_test_split`: Data splitting
  - `accuracy_score`, `classification_report`, `confusion_matrix`: Model evaluation

**Visualization:**
- `plotly`: Interactive visualizations (express and graph_objects modules)
- `matplotlib`: Static plotting
- `seaborn`: Statistical data visualization

**Web Framework:**
- `streamlit`: Web application framework and UI components

### Data Sources

**Current Implementation:**
- Static data embedded in code (qualified teams list)
- Simulated data for non-qualified teams using random generation

**Future Considerations:**
- FIFA.com API or web scraping for real-time rankings
- Historical match data for improved predictions
- Dynamic updates as teams qualify

### No External Services Currently

The application runs entirely standalone without:
- External databases
- Authentication systems
- Third-party APIs (though FIFA data API integration is mentioned for future enhancement)
- Cloud storage services