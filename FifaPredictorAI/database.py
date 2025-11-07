import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = os.environ.get('DATABASE_URL')

engine = None
SessionLocal = None
Base = declarative_base()

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        print(f"Database connection error: {e}")
        engine = None
        SessionLocal = None

class PredictionHistory(Base):
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_type = Column(String(50))
    team_name = Column(String(100))
    finalist_probability = Column(Float)
    fifa_ranking = Column(Integer)
    goals_scored = Column(Integer)
    goals_conceded = Column(Integer)
    avg_age = Column(Float)
    win_rate = Column(Float)
    qualified = Column(Boolean)
    accuracy = Column(Float)
    is_top_2 = Column(Boolean, default=False)

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)
    else:
        print("Database not initialized - DATABASE_URL not configured")

def get_db():
    if SessionLocal is None:
        return None
    db = SessionLocal()
    return db

def save_prediction(db, model_type, team_data, finalist_prob, accuracy, is_top_2=False):
    prediction = PredictionHistory(
        model_type=model_type,
        team_name=team_data['Team'],
        finalist_probability=finalist_prob,
        fifa_ranking=team_data['FIFA_Ranking'],
        goals_scored=team_data['Goals_Scored'],
        goals_conceded=team_data['Goals_Conceded'],
        avg_age=team_data['Avg_Age'],
        win_rate=team_data['Win_Rate'],
        qualified=team_data['Qualified'],
        accuracy=accuracy,
        is_top_2=is_top_2
    )
    db.add(prediction)
    db.commit()
    return prediction

def get_prediction_history(db, limit=100):
    return db.query(PredictionHistory).order_by(PredictionHistory.timestamp.desc()).limit(limit).all()

def get_top_predictions_by_model(db, model_type, limit=10):
    return db.query(PredictionHistory).filter(
        PredictionHistory.model_type == model_type
    ).order_by(
        PredictionHistory.finalist_probability.desc()
    ).limit(limit).all()
