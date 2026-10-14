from sqlalchemy import Column, Integer, String, Float, DateTime
from src.database.session import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), index=True)
    horizon = Column(String)
    
    price = Column(Float)
    probability_down = Column(Float)
    probability_neutral = Column(Float)
    probability_up = Column(Float)
    
    predicted_class = Column(String)
    confidence = Column(Float)
    model_version = Column(String)
    
    # Eventual outcome populated later
    actual_outcome = Column(String, nullable=True)
    actual_return = Column(Float, nullable=True)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), index=True)
    action = Column(String)  # BUY or SELL
    entry_price = Column(Float)
    take_profit = Column(Float)
    stop_loss = Column(Float)
    confidence = Column(Float)
    status = Column(String, default="ACTIVE") # ACTIVE, CLOSED, etc.


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, default="BTCUSDT")
    side = Column(String)  # BUY or SELL
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity_btc = Column(Float)
    position_usd = Column(Float)
    pnl_usd = Column(Float)
    pnl_pct = Column(Float)
    exit_reason = Column(String)  # predictive_reversal, trailing_stop, stop_loss, emergency_stop
    opened_at = Column(DateTime(timezone=True))
    closed_at = Column(DateTime(timezone=True), index=True)
    balance_after = Column(Float)


class PositionRecord(Base):
    __tablename__ = "open_positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, default="BTCUSDT")
    side = Column(String)  # BUY or SELL
    entry_price = Column(Float)
    quantity_btc = Column(Float)
    usd_value = Column(Float)
    stop_loss_price = Column(Float)
    highest_price = Column(Float)
    lowest_price = Column(Float)
    opened_at = Column(DateTime(timezone=True))


