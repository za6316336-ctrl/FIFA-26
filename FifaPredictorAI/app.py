import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from database import init_db, get_db, save_prediction, get_prediction_history, get_top_predictions_by_model

init_db()

st.set_page_config(page_title="FIFA World Cup 2026 Finalist Prediction", page_icon="🏆", layout="wide")

st.title("🏆 FIFA World Cup 2026 Finalist Prediction")
st.markdown("### AI-Powered Prediction System using Machine Learning")

st.info("""
**Project Overview:** This application predicts the top 2 finalists for FIFA World Cup 2026 using machine learning.
- **Current Status:** 28 teams have qualified, 20 teams are simulated from top FIFA rankings
- **Algorithm:** Random Forest Classifier
- **Features:** FIFA Ranking, Goals Scored/Conceded, Average Age, Win Rate
""")

def generate_team_data():
    columns = ['Team', 'FIFA_Ranking', 'Goals_Scored', 'Goals_Conceded', 'Avg_Age', 'Win_Rate', 'Qualified']
    
    qualified_teams = [
        ['Argentina', 1, 35, 10, 28.5, 0.83, True],
        ['France', 2, 33, 12, 27.9, 0.81, True],
        ['Brazil', 3, 40, 9, 26.7, 0.85, True],
        ['England', 4, 32, 11, 27.2, 0.79, True],
        ['Belgium', 5, 30, 13, 28.8, 0.77, True],
        ['Netherlands', 6, 28, 10, 27.5, 0.78, True],
        ['Portugal', 7, 34, 14, 28.1, 0.76, True],
        ['Spain', 8, 31, 12, 27.3, 0.80, True],
        ['Italy', 9, 29, 11, 28.4, 0.75, True],
        ['Croatia', 10, 27, 13, 29.1, 0.74, True],
        ['Uruguay', 11, 26, 10, 28.6, 0.73, True],
        ['Mexico', 12, 28, 15, 27.8, 0.70, True],
        ['Germany', 13, 33, 12, 27.4, 0.82, True],
        ['Switzerland', 14, 25, 11, 28.2, 0.71, True],
        ['Denmark', 15, 27, 13, 27.7, 0.72, True],
        ['Colombia', 16, 29, 14, 27.9, 0.69, True],
        ['Senegal', 17, 24, 12, 26.5, 0.68, True],
        ['Japan', 18, 23, 13, 27.1, 0.67, True],
        ['Morocco', 19, 26, 11, 27.6, 0.70, True],
        ['USA', 20, 28, 16, 26.8, 0.66, True],
        ['Canada', 21, 22, 14, 26.9, 0.65, True],
        ['South Korea', 22, 21, 15, 27.3, 0.64, True],
        ['Poland', 23, 25, 17, 28.3, 0.63, True],
        ['Australia', 24, 20, 16, 27.8, 0.62, True],
        ['Ecuador', 25, 24, 18, 26.7, 0.61, True],
        ['Iran', 26, 19, 14, 27.5, 0.60, True],
        ['Peru', 27, 22, 19, 28.1, 0.59, True],
        ['Nigeria', 28, 23, 17, 26.4, 0.61, True],
    ]
    
    df = pd.DataFrame(qualified_teams, columns=columns)
    
    if len(df) < 48:
        potential_teams = [
            'Sweden', 'Austria', 'Ukraine', 'Turkey', 'Russia', 'Wales', 
            'Czech Republic', 'Hungary', 'Serbia', 'Romania', 'Scotland',
            'Norway', 'Greece', 'Slovakia', 'Finland', 'Costa Rica',
            'Chile', 'Paraguay', 'Venezuela', 'Tunisia', 'Algeria',
            'Egypt', 'Cameroon', 'Ghana', 'Ivory Coast', 'Mali'
        ]
        
        teams_needed = 48 - len(df)
        simulated_teams = random.sample(potential_teams, min(teams_needed, len(potential_teams)))
        
        for team in simulated_teams:
            df.loc[len(df)] = [
                team, 
                random.randint(29, 80), 
                random.randint(15, 30),
                random.randint(10, 22), 
                round(random.uniform(25.5, 29.5), 1),
                round(random.uniform(0.50, 0.70), 2), 
                False
            ]
    
    return df

if 'df' not in st.session_state:
    st.session_state['df'] = generate_team_data()

df = st.session_state['df']

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Team Data", "✏️ Edit Data", "🤖 Model Training", "🏁 Predictions", "🏆 Bracket Simulation", "📊 Historical Tracking"])

with tab1:
    st.header("Team Statistics Overview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Teams", len(df))
        with col_b:
            st.metric("Qualified Teams", df['Qualified'].sum())
        with col_c:
            st.metric("Simulated Teams", 48 - df['Qualified'].sum())
        with col_d:
            st.metric("Features Used", 5)
    
    with col2:
        if st.button("🔄 Refresh Rankings", help="Simulate fetching latest FIFA rankings data"):
            with st.spinner("Fetching latest rankings..."):
                time.sleep(1)
                
                ranking_adjustments = {
                    'Argentina': {'FIFA_Ranking': 1, 'Win_Rate': min(0.85, df.loc[df['Team'] == 'Argentina', 'Win_Rate'].values[0] + 0.01)},
                    'France': {'FIFA_Ranking': 2, 'Win_Rate': min(0.83, df.loc[df['Team'] == 'France', 'Win_Rate'].values[0] + 0.01)},
                    'Brazil': {'FIFA_Ranking': 3, 'Win_Rate': min(0.87, df.loc[df['Team'] == 'Brazil', 'Win_Rate'].values[0] + 0.01)},
                    'England': {'FIFA_Ranking': 4, 'Win_Rate': min(0.81, df.loc[df['Team'] == 'England', 'Win_Rate'].values[0] + 0.01)},
                    'Germany': {'FIFA_Ranking': 5, 'Win_Rate': min(0.84, df.loc[df['Team'] == 'Germany', 'Win_Rate'].values[0] + 0.01)}
                }
                
                for team, updates in ranking_adjustments.items():
                    if team in st.session_state['df']['Team'].values:
                        for key, value in updates.items():
                            st.session_state['df'].loc[st.session_state['df']['Team'] == team, key] = value
                
                if 'model' in st.session_state:
                    del st.session_state['model']
                
                st.success("✅ Rankings refreshed! Model needs retraining.")
                st.rerun()
    
    st.subheader("All 48 Teams")
    
    show_qualified_only = st.checkbox("Show only qualified teams")
    if show_qualified_only:
        display_df = df[df['Qualified'] == True]
    else:
        display_df = df
    
    st.dataframe(
        display_df.style.background_gradient(subset=['FIFA_Ranking'], cmap='RdYlGn_r')
                       .background_gradient(subset=['Win_Rate'], cmap='Greens')
                       .format({'Win_Rate': '{:.2%}', 'Avg_Age': '{:.1f}'}),
        use_container_width=True,
        height=400
    )
    
    st.subheader("Team Statistics Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='FIFA_Ranking', nbins=20, 
                           title='FIFA Ranking Distribution',
                           labels={'FIFA_Ranking': 'FIFA Ranking', 'count': 'Number of Teams'},
                           color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(df, x='Win_Rate', nbins=15, 
                           title='Win Rate Distribution',
                           labels={'Win_Rate': 'Win Rate', 'count': 'Number of Teams'},
                           color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Edit Team Data")
    
    st.markdown("""
    **Data Management:** 
    - Edit existing team statistics by clicking on cells in the table below
    - Add new teams using the form
    - Changes are saved automatically and will be used in model training
    """)
    
    st.subheader("Edit Existing Teams")
    
    edited_df = st.data_editor(
        st.session_state['df'],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Team": st.column_config.TextColumn("Team Name", required=True),
            "FIFA_Ranking": st.column_config.NumberColumn("FIFA Ranking", min_value=1, max_value=211, step=1),
            "Goals_Scored": st.column_config.NumberColumn("Goals Scored", min_value=0, max_value=100, step=1),
            "Goals_Conceded": st.column_config.NumberColumn("Goals Conceded", min_value=0, max_value=100, step=1),
            "Avg_Age": st.column_config.NumberColumn("Average Age", min_value=20.0, max_value=35.0, format="%.1f"),
            "Win_Rate": st.column_config.NumberColumn("Win Rate", min_value=0.0, max_value=1.0, format="%.2f"),
            "Qualified": st.column_config.CheckboxColumn("Qualified", default=False)
        },
        hide_index=False,
        key="team_editor"
    )
    
    if not edited_df.equals(st.session_state['df']):
        st.session_state['df'] = edited_df
        if 'model' in st.session_state:
            del st.session_state['model']
        st.success("✅ Team data updated! Please retrain the model in the Model Training tab.")
        st.rerun()
    
    st.subheader("Add New Team")
    
    with st.form("add_team_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_team_name = st.text_input("Team Name", placeholder="e.g., Iceland")
            new_ranking = st.number_input("FIFA Ranking", min_value=1, max_value=211, value=50, step=1)
        
        with col2:
            new_goals_scored = st.number_input("Goals Scored", min_value=0, max_value=100, value=20, step=1)
            new_goals_conceded = st.number_input("Goals Conceded", min_value=0, max_value=100, value=15, step=1)
        
        with col3:
            new_avg_age = st.number_input("Average Age", min_value=20.0, max_value=35.0, value=27.5, step=0.1)
            new_win_rate = st.number_input("Win Rate", min_value=0.0, max_value=1.0, value=0.60, step=0.01)
        
        new_qualified = st.checkbox("Qualified Team", value=False)
        
        submitted = st.form_submit_button("➕ Add Team", type="primary")
        
        if submitted:
            if new_team_name and new_team_name.strip():
                if new_team_name in st.session_state['df']['Team'].values:
                    st.error(f"❌ Team '{new_team_name}' already exists!")
                else:
                    new_row = pd.DataFrame([[
                        new_team_name,
                        new_ranking,
                        new_goals_scored,
                        new_goals_conceded,
                        new_avg_age,
                        new_win_rate,
                        new_qualified
                    ]], columns=st.session_state['df'].columns)
                    
                    st.session_state['df'] = pd.concat([st.session_state['df'], new_row], ignore_index=True)
                    
                    if 'model' in st.session_state:
                        del st.session_state['model']
                    
                    st.success(f"✅ Team '{new_team_name}' added successfully!")
                    st.rerun()
            else:
                st.error("❌ Please enter a team name!")

with tab3:
    st.header("Machine Learning Model Training & Comparison")
    
    st.markdown("""
    **Training Process:**
    1. Features: FIFA Ranking, Goals Scored, Goals Conceded, Average Age, Win Rate
    2. Target: Strong contenders (Argentina, France, Brazil, England, Germany) labeled as 1
    3. Multiple Algorithms: Random Forest, XGBoost, Neural Network
    4. Split: 80% training, 20% testing
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training all models..."):
                X = df[['FIFA_Ranking', 'Goals_Scored', 'Goals_Conceded', 'Avg_Age', 'Win_Rate']]
                y = np.where(df['Team'].isin(['Argentina', 'France', 'Brazil', 'England', 'Germany']), 1, 0)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                models_results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Training Random Forest...")
                progress_bar.progress(10)
                start_time = time.time()
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_time = time.time() - start_time
                models_results['Random Forest'] = {
                    'model': rf_model,
                    'accuracy': accuracy_score(y_test, rf_pred),
                    'predictions': rf_pred,
                    'training_time': rf_time
                }
                progress_bar.progress(40)
                
                status_text.text("Training XGBoost...")
                start_time = time.time()
                xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_test)
                xgb_time = time.time() - start_time
                models_results['XGBoost'] = {
                    'model': xgb_model,
                    'accuracy': accuracy_score(y_test, xgb_pred),
                    'predictions': xgb_pred,
                    'training_time': xgb_time
                }
                progress_bar.progress(70)
                
                status_text.text("Training Neural Network...")
                start_time = time.time()
                nn_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
                nn_model.fit(X_train, y_train)
                nn_pred = nn_model.predict(X_test)
                nn_time = time.time() - start_time
                models_results['Neural Network'] = {
                    'model': nn_model,
                    'accuracy': accuracy_score(y_test, nn_pred),
                    'predictions': nn_pred,
                    'training_time': nn_time
                }
                progress_bar.progress(100)
                
                status_text.text("✅ All models trained successfully!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                st.session_state['models_results'] = models_results
                st.session_state['X'] = X
                st.session_state['y'] = y
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                best_model_name = max(models_results.items(), key=lambda x: x[1]['accuracy'])[0]
                st.session_state['model'] = models_results[best_model_name]['model']
                st.session_state['selected_model'] = best_model_name
                st.session_state['accuracy'] = models_results[best_model_name]['accuracy']
                st.session_state['y_pred'] = models_results[best_model_name]['predictions']
                
                st.success(f"✅ All models trained! Best performing model: {best_model_name}")
    
    with col2:
        if 'models_results' in st.session_state:
            selected_model = st.selectbox(
                "Select Model for Predictions",
                options=list(st.session_state['models_results'].keys()),
                index=list(st.session_state['models_results'].keys()).index(st.session_state.get('selected_model', 'Random Forest'))
            )
            
            if selected_model != st.session_state.get('selected_model'):
                st.session_state['selected_model'] = selected_model
                st.session_state['model'] = st.session_state['models_results'][selected_model]['model']
                st.session_state['accuracy'] = st.session_state['models_results'][selected_model]['accuracy']
                st.session_state['y_pred'] = st.session_state['models_results'][selected_model]['predictions']
                st.rerun()
    
    if 'models_results' in st.session_state:
        st.subheader("📊 Model Performance Comparison")
        
        comparison_data = []
        for model_name, results in st.session_state['models_results'].items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.2%}",
                'Training Time (s)': f"{results['training_time']:.3f}",
                'Status': '🏆 Best' if model_name == st.session_state.get('selected_model') else '✓ Trained'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        fig_comparison = px.bar(
            comparison_df,
            x='Model',
            y=[float(acc.strip('%'))/100 for acc in comparison_df['Accuracy']],
            title='Model Accuracy Comparison',
            labels={'y': 'Accuracy', 'x': 'Model'},
            color=[float(acc.strip('%'))/100 for acc in comparison_df['Accuracy']],
            color_continuous_scale='RdYlGn',
            text=[acc for acc in comparison_df['Accuracy']]
        )
        fig_comparison.update_traces(textposition='outside')
        fig_comparison.update_layout(showlegend=False)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.subheader(f"📈 {st.session_state.get('selected_model', 'Selected')} Model Details")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{st.session_state['accuracy']:.2%}")
        with col2:
            st.metric("Training Samples", len(st.session_state['X_train']))
        with col3:
            st.metric("Test Samples", len(st.session_state['y_test']))
        
        st.subheader("Classification Report")
        report = classification_report(
            st.session_state['y_test'],
            st.session_state['y_pred'],
            target_names=['Other Teams', 'Strong Contenders'],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format('{:.2f}'), use_container_width=True)
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Other', 'Contender'],
                   yticklabels=['Other', 'Contender'],
                   ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title(f'Confusion Matrix - {st.session_state.get("selected_model", "Selected Model")}')
        st.pyplot(fig)
        
        if hasattr(st.session_state['model'], 'feature_importances_'):
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['FIFA_Ranking', 'Goals_Scored', 'Goals_Conceded', 'Avg_Age', 'Win_Rate'],
                'Importance': st.session_state['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig3 = px.bar(feature_importance, x='Importance', y='Feature',
                         orientation='h',
                         title=f'Feature Importance - {st.session_state.get("selected_model", "Selected Model")}',
                         color='Importance',
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("⚠️ Please train the models first by clicking the 'Train All Models' button above.")

with tab4:
    st.header("FIFA 2026 Finalist Predictions")
    
    if 'model' in st.session_state:
        df['Finalist_Prob'] = st.session_state['model'].predict_proba(st.session_state['X'])[:, 1]
        finalists = df.sort_values(by='Finalist_Prob', ascending=False).head(10)
        
        st.subheader("🥇 Top 2 Predicted Finalists")
        
        top_2 = finalists.head(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🥇 Finalist 1: {top_2.iloc[0]['Team']}")
            st.metric("Probability", f"{top_2.iloc[0]['Finalist_Prob']:.2%}")
            st.metric("FIFA Ranking", int(top_2.iloc[0]['FIFA_Ranking']))
            st.metric("Win Rate", f"{top_2.iloc[0]['Win_Rate']:.2%}")
        
        with col2:
            st.markdown(f"### 🥈 Finalist 2: {top_2.iloc[1]['Team']}")
            st.metric("Probability", f"{top_2.iloc[1]['Finalist_Prob']:.2%}")
            st.metric("FIFA Ranking", int(top_2.iloc[1]['FIFA_Ranking']))
            st.metric("Win Rate", f"{top_2.iloc[1]['Win_Rate']:.2%}")
        
        st.subheader("Top 10 Contenders")
        
        fig4 = px.bar(finalists, x='Finalist_Prob', y='Team', 
                     orientation='h',
                     title='Top 10 Teams by Finalist Probability',
                     labels={'Finalist_Prob': 'Finalist Probability', 'Team': 'Team'},
                     color='Finalist_Prob',
                     color_continuous_scale='RdYlGn',
                     text='Finalist_Prob')
        fig4.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig4.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("Detailed Top 10 Statistics")
        st.dataframe(
            finalists[['Team', 'FIFA_Ranking', 'Goals_Scored', 'Goals_Conceded', 
                      'Avg_Age', 'Win_Rate', 'Finalist_Prob', 'Qualified']]
            .style.background_gradient(subset=['Finalist_Prob'], cmap='RdYlGn')
                  .format({'Finalist_Prob': '{:.2%}', 'Win_Rate': '{:.2%}', 'Avg_Age': '{:.1f}'}),
            use_container_width=True
        )
    else:
        st.warning("⚠️ Please train the model in the 'Model Training' tab first.")

with tab5:
    st.header("🏆 Tournament Bracket Simulation")
    
    if 'model' in st.session_state:
        st.markdown("""
        **Bracket Simulation:** This shows the predicted tournament progression from Round of 32 through the Finals.
        - Teams are seeded based on their finalist probability
        - Match winners are determined by the team with higher probability
        """)
        
        df['Finalist_Prob'] = st.session_state['model'].predict_proba(st.session_state['X'])[:, 1]
        sorted_teams = df.sort_values(by='Finalist_Prob', ascending=False).head(32)
        
        st.subheader("Round of 32")
        col1, col2 = st.columns(2)
        
        round_of_16_teams = []
        matches = []
        
        for i in range(0, 32, 2):
            team1 = sorted_teams.iloc[i]
            team2 = sorted_teams.iloc[i+1]
            
            winner = team1 if team1['Finalist_Prob'] > team2['Finalist_Prob'] else team2
            round_of_16_teams.append(winner)
            
            match_result = f"**{team1['Team']}** ({team1['Finalist_Prob']:.1%}) vs **{team2['Team']}** ({team2['Finalist_Prob']:.1%})"
            winner_text = f"➡️ Winner: **{winner['Team']}**"
            matches.append((match_result, winner_text))
        
        with col1:
            for i in range(0, 8):
                with st.container():
                    st.markdown(matches[i][0])
                    st.success(matches[i][1])
                    st.markdown("---")
        
        with col2:
            for i in range(8, 16):
                with st.container():
                    st.markdown(matches[i][0])
                    st.success(matches[i][1])
                    st.markdown("---")
        
        st.subheader("Round of 16")
        quarter_final_teams = []
        round_of_16_df = pd.DataFrame(round_of_16_teams)
        
        col1, col2 = st.columns(2)
        matches_16 = []
        
        for i in range(0, 16, 2):
            team1 = round_of_16_df.iloc[i]
            team2 = round_of_16_df.iloc[i+1]
            
            winner = team1 if team1['Finalist_Prob'] > team2['Finalist_Prob'] else team2
            quarter_final_teams.append(winner)
            
            match_result = f"**{team1['Team']}** vs **{team2['Team']}**"
            winner_text = f"➡️ **{winner['Team']}** advances"
            matches_16.append((match_result, winner_text))
        
        with col1:
            for i in range(0, 4):
                st.info(matches_16[i][0])
                st.success(matches_16[i][1])
        
        with col2:
            for i in range(4, 8):
                st.info(matches_16[i][0])
                st.success(matches_16[i][1])
        
        st.subheader("Quarter Finals")
        semi_final_teams = []
        quarter_finals_df = pd.DataFrame(quarter_final_teams)
        
        col1, col2 = st.columns(2)
        
        for i in range(0, 8, 2):
            team1 = quarter_finals_df.iloc[i]
            team2 = quarter_finals_df.iloc[i+1]
            
            winner = team1 if team1['Finalist_Prob'] > team2['Finalist_Prob'] else team2
            semi_final_teams.append(winner)
            
            col_target = col1 if i < 4 else col2
            with col_target:
                st.info(f"**{team1['Team']}** vs **{team2['Team']}**")
                st.success(f"➡️ **{winner['Team']}** to Semi-Finals")
        
        st.subheader("Semi Finals")
        semi_finals_df = pd.DataFrame(semi_final_teams)
        
        final_teams = []
        
        col1, col2 = st.columns(2)
        
        for i in range(0, 4, 2):
            team1 = semi_finals_df.iloc[i]
            team2 = semi_finals_df.iloc[i+1]
            
            winner = team1 if team1['Finalist_Prob'] > team2['Finalist_Prob'] else team2
            final_teams.append(winner)
            
            col_target = col1 if i == 0 else col2
            with col_target:
                st.warning(f"**{team1['Team']}** vs **{team2['Team']}**")
                st.success(f"🏆 **{winner['Team']}** to FINAL!")
        
        st.subheader("🏆 FINAL")
        finals_df = pd.DataFrame(final_teams)
        
        team1 = finals_df.iloc[0]
        team2 = finals_df.iloc[1]
        
        champion = team1 if team1['Finalist_Prob'] > team2['Finalist_Prob'] else team2
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"### 🥇 {team1['Team']} vs {team2['Team']} 🥇")
            st.markdown(f"**{team1['Team']}**: {team1['Finalist_Prob']:.2%} probability")
            st.markdown(f"**{team2['Team']}**: {team2['Finalist_Prob']:.2%} probability")
            st.markdown("---")
            st.success(f"# 🏆 CHAMPION: {champion['Team']} 🏆")
            st.balloons()
    else:
        st.warning("⚠️ Please train a model in the 'Model Training' tab first.")

with tab6:
    st.header("📊 Historical Prediction Tracking")
    
    if 'model' in st.session_state:
        st.markdown("""
        **Historical Tracking:** Save your predictions to track how model performance changes over time.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("💾 Save Current Predictions to Database", type="primary"):
                db = get_db()
                if db is None:
                    st.error("❌ Database not configured. Please set DATABASE_URL environment variable.")
                else:
                    try:
                        with st.spinner("Saving predictions..."):
                            df['Finalist_Prob'] = st.session_state['model'].predict_proba(st.session_state['X'])[:, 1]
                            finalists = df.sort_values(by='Finalist_Prob', ascending=False)
                            
                            model_name = st.session_state.get('selected_model', 'Random Forest')
                            accuracy = st.session_state.get('accuracy', 0)
                            
                            for idx, row in finalists.head(10).iterrows():
                                is_top_2 = idx in finalists.head(2).index
                                save_prediction(
                                    db,
                                    model_type=model_name,
                                    team_data=row.to_dict(),
                                    finalist_prob=row['Finalist_Prob'],
                                    accuracy=accuracy,
                                    is_top_2=is_top_2
                                )
                            
                            st.success(f"✅ Saved top 10 predictions from {model_name} model!")
                            st.rerun()
                    finally:
                        if db:
                            db.close()
        
        with col2:
            st.info(f"**Current Model:** {st.session_state.get('selected_model', 'None')}")
            st.metric("Model Accuracy", f"{st.session_state.get('accuracy', 0):.2%}")
        
        st.subheader("📈 Prediction History")
        
        db = get_db()
        if db is None:
            st.warning("⚠️ Database not configured. Historical tracking requires DATABASE_URL to be set.")
            history = []
            model_predictions_cache = {}
        else:
            try:
                history = get_prediction_history(db, limit=200)
                model_predictions_cache = {}
                if 'models_results' in st.session_state:
                    for model_name in st.session_state['models_results'].keys():
                        model_predictions_cache[model_name] = get_top_predictions_by_model(db, model_name, limit=10)
            finally:
                db.close()
        
        if history:
            history_data = []
            for pred in history:
                history_data.append({
                    'Date': pred.timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Model': pred.model_type,
                    'Team': pred.team_name,
                    'Probability': f"{pred.finalist_probability:.2%}",
                    'FIFA Rank': pred.fifa_ranking,
                    'Accuracy': f"{pred.accuracy:.2%}",
                    'Top 2': '🏆' if pred.is_top_2 else ''
                })
            
            history_df = pd.DataFrame(history_data)
            
            st.dataframe(history_df, use_container_width=True, height=400)
            
            st.subheader("📊 Model Performance Over Time")
            
            unique_timestamps = {}
            for pred in history:
                key = f"{pred.timestamp.strftime('%Y-%m-%d %H:%M')}_{pred.model_type}"
                if key not in unique_timestamps:
                    unique_timestamps[key] = {
                        'Timestamp': pred.timestamp,
                        'Model': pred.model_type,
                        'Accuracy': pred.accuracy
                    }
            
            if unique_timestamps:
                perf_df = pd.DataFrame(list(unique_timestamps.values()))
                
                fig = px.line(perf_df, x='Timestamp', y='Accuracy', color='Model',
                             title='Model Accuracy Trends Over Time',
                             labels={'Accuracy': 'Accuracy', 'Timestamp': 'Date/Time'},
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🏆 Top Predicted Teams by Model")
            
            if 'models_results' in st.session_state:
                model_tabs = st.tabs(list(st.session_state['models_results'].keys()))
                
                for idx, model_name in enumerate(st.session_state['models_results'].keys()):
                    with model_tabs[idx]:
                        model_predictions = model_predictions_cache.get(model_name, [])
                        
                        if model_predictions:
                            pred_data = []
                            for pred in model_predictions:
                                pred_data.append({
                                    'Team': pred.team_name,
                                    'Probability': pred.finalist_probability,
                                    'Last Updated': pred.timestamp.strftime('%Y-%m-%d %H:%M')
                                })
                            
                            pred_df = pd.DataFrame(pred_data)
                            
                            fig = px.bar(pred_df.head(10), x='Probability', y='Team',
                                       orientation='h',
                                       title=f'Top 10 Teams - {model_name}',
                                       labels={'Probability': 'Finalist Probability'},
                                       text='Probability')
                            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No predictions saved yet for {model_name}")
        else:
            st.info("No prediction history yet. Save some predictions to start tracking!")
    else:
        st.warning("⚠️ Please train a model in the 'Model Training' tab first.")

st.sidebar.title("About This Project")
st.sidebar.info("""
**FIFA World Cup 2026 Prediction**

AI-powered prediction system using advanced machine learning to forecast World Cup finalists.

**Features:**
- 📊 48 Teams (28 qualified + 20 simulated)
- ✏️ Editable team data management
- 🤖 3 ML Models (Random Forest, XGBoost, Neural Network)
- 🏁 Top 2 finalist predictions
- 🏆 Tournament bracket simulation
- 📊 Historical prediction tracking
- 📈 Interactive visualizations

**Based on:** Shivaprasad Sir's instructions

**Tech Stack:**
- Python, Streamlit, Scikit-learn
- XGBoost, Plotly, PostgreSQL
""")

st.sidebar.markdown("---")
if 'model' in st.session_state:
    st.sidebar.success(f"🤖 Active Model: {st.session_state.get('selected_model', 'Random Forest')}")
    st.sidebar.metric("Accuracy", f"{st.session_state.get('accuracy', 0):.2%}")
else:
    st.sidebar.markdown("🏆 **Train models to see predictions!**")
