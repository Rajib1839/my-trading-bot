# deep_learning.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import requests
import ta
import datetime
import time

class LSTMTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model_file = "lstm_model.h5"
        self.scaler_file = "lstm_scaler.pkl"
        self.accuracy = 0
        self.load_model()
    
    def load_model(self):
        """সেভ করা মডেল লোড করে"""
        if os.path.exists(self.model_file):
            try:
                self.model = keras.models.load_model(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                print("✅ LSTM মডেল লোড হয়েছে!")
            except Exception as e:
                print(f"⚠️ LSTM মডেল লোড করতে সমস্যা: {e}")
                self.create_model()
        else:
            print("🆕 নতুন LSTM মডেল বানানো হবে")
            self.create_model()
    
    def create_model(self):
        """LSTM মডেল আর্কিটেকচার"""
        self.model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(60, 7)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(50),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ LSTM মডেল তৈরি হয়েছে!")
    
    def save_model(self):
        """মডেল সেভ করে"""
        if self.model:
            self.model.save(self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print("✅ LSTM মডেল সেভ হয়েছে!")
    
    def prepare_sequences(self, coin_id, days=180):
        """LSTM এর জন্য সিকোয়েন্স ডাটা প্রিপেয়ার করে"""
        try:
            print(f"📡 ডাটা আনছি {coin_id} থেকে, {days} দিনের...")
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ API error: {response.status_code}")
                return None, None
            
            data = response.json()
            prices = np.array([p[1] for p in data["prices"]])
            
            if len(prices) < 100:
                print(f"❌ পর্যাপ্ত ডাটা নেই: {len(prices)}")
                return None, None
            
            print(f"✅ {len(prices)} টি প্রাইস পয়েন্ট পাওয়া গেছে")
            
            features = []
            for i in range(60, len(prices)-5):
                sequence = prices[i-60:i]
                
                df = pd.Series(sequence)
                rsi = ta.momentum.RSIIndicator(df).rsi().fillna(50).values[-1]
                macd = ta.trend.MACD(df).macd().fillna(0).values[-1]
                
                price_change_1h = (sequence[-1] - sequence[-2]) / sequence[-2] * 100 if len(sequence) > 1 else 0
                price_change_4h = (sequence[-1] - sequence[-5]) / sequence[-5] * 100 if len(sequence) > 5 else 0
                price_change_24h = (sequence[-1] - sequence[-25]) / sequence[-25] * 100 if len(sequence) > 25 else 0
                
                volatility = np.std(sequence) / sequence[-1] if sequence[-1] != 0 else 0
                
                feature_vector = [
                    sequence[-1],
                    rsi,
                    macd,
                    price_change_1h,
                    price_change_4h,
                    price_change_24h,
                    volatility
                ]
                
                features.append(feature_vector)
            
            features = np.array(features)
            features_scaled = self.scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(60, len(features_scaled)-5):
                X.append(features_scaled[i-60:i])
                
                future_price = prices[i+60+5] if i+60+5 < len(prices) else prices[-1]
                current_price = prices[i+60]
                price_change_future = (future_price - current_price) / current_price * 100
                
                if price_change_future < -2:
                    label = 0
                elif price_change_future > 2:
                    label = 2
                else:
                    label = 1
                
                y.append(label)
            
            y = tf.keras.utils.to_categorical(y, num_classes=3)
            
            print(f"✅ {len(X)} টি সিকোয়েন্স তৈরি হয়েছে")
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"LSTM ডাটা প্রিপেয়ার করতে সমস্যা: {e}")
            return None, None
    
    def train(self, coin_id="bitcoin", epochs=30):
        """LSTM মডেল ট্রেন করে"""
        print(f"🔄 LSTM ট্রেনিং শুরু হচ্ছে... ({coin_id})")
        
        X, y = self.prepare_sequences(coin_id, days=90)
        
        if X is None or len(X) < 50:
            print("❌ ট্রেনিং এর জন্য পর্যাপ্ত ডাটা নেই")
            return False
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"📊 ট্রেন ডাটা: {len(X_train)}, টেস্ট ডাটা: {len(X_test)}")
        
        # নতুন মডেল বানাও (ফিক্স)
        self.create_model()
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stop],
            verbose=1
        )
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.accuracy = accuracy * 100
        
        print(f"✅ LSTM ট্রেনিং সম্পন্ন! একুরেসি: {self.accuracy:.2f}%")
        
        self.save_model()
        
        return True
    
    def predict(self, recent_prices, current_indicators):
        """বর্তমান ডাটা থেকে প্রেডিকশন করে"""
        if self.model is None:
            return None, 0, None
        
        try:
            features = np.array([[
                recent_prices[-1],
                current_indicators['rsi'],
                current_indicators['macd'],
                0, 0, 0,
                np.std(recent_prices) / recent_prices[-1] if recent_prices[-1] != 0 else 0
            ]])
            
            features_scaled = self.scaler.transform(features)
            sequence = np.array([features_scaled] * 60).reshape(1, 60, 7)
            
            predictions = self.model.predict(sequence, verbose=0)[0]
            
            class_names = ["SELL", "HOLD", "BUY"]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            
            return class_names[predicted_class], round(confidence, 2), predictions
            
        except Exception as e:
            print(f"LSTM প্রেডিকশন error: {e}")
            return None, 0, None

# LSTM মডেল ইনিশিয়ালাইজ
lstm_model = LSTMTradingModel()