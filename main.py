import discord
from deep_learning import lstm_model  # LSTM মডেল ইমপোর্ট
from discord.ext import commands
import requests
import pandas as pd
import ta
import os
import json
import datetime
import asyncio
from collections import defaultdict
from dotenv import load_dotenv
from database import db  # নতুন ডাটাবেস ইমপোর্ট
from auto_trader import AutoTradingBot

# ================ 🤖 মেশিন লার্নিং মডিউল ================ #
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class MLTradingModel:
    def __init__(self):
        self.model = None
        self.model_file = "ml_model.pkl"
        self.accuracy = 0
        self.load_model()
    
    def load_model(self):
        """সেভ করা মডেল লোড করে"""
        if os.path.exists(self.model_file):
            try:
                self.model = joblib.load(self.model_file)
                print("✅ ML মডেল লোড হয়েছে!")
            except:
                print("⚠️ মডেল লোড করতে সমস্যা, নতুন বানানো হবে")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            print("🆕 নতুন ML মডেল বানানো হবে")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def save_model(self):
        """মডেল সেভ করে"""
        if self.model:
            joblib.dump(self.model, self.model_file)
            print("✅ ML মডেল সেভ হয়েছে!")
    
    def prepare_features(self, coin_id, days=60):
        """ML এর জন্য ডাটা প্রিপেয়ার করে"""
        try:
            # ৬০ দিনের ডাটা আনি
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            prices = data["prices"]
            volumes = data["total_volumes"]
            
            features = []
            labels = []
            
            for i in range(50, len(prices)-5):  # পরবর্তী ৫ ঘন্টা প্রেডিক্ট করবো
                current_price = prices[i][1]
                
                # আগের ৫০ টি প্রাইস থেকে ফিচার বের করি
                last_50_prices = [p[1] for p in prices[i-50:i]]
                df = pd.Series(last_50_prices)
                
                # টেকনিক্যাল ইন্ডিকেটর
                rsi = ta.momentum.RSIIndicator(df).rsi().iloc[-1] if len(df) > 14 else 50
                macd = ta.trend.MACD(df).macd().iloc[-1] if len(df) > 26 else 0
                
                # প্রাইস চেঞ্জ
                price_change_1h = (current_price - prices[i-1][1]) / prices[i-1][1] * 100
                price_change_4h = (current_price - prices[i-4][1]) / prices[i-4][1] * 100 if i > 4 else 0
                price_change_24h = (current_price - prices[i-24][1]) / prices[i-24][1] * 100 if i > 24 else 0
                
                # ভলিউম
                current_volume = volumes[i][1] if i < len(volumes) else 0
                avg_volume = np.mean([v[1] for v in volumes[i-10:i]]) if i > 10 else current_volume
                
                # ফিচার ভেক্টর
                feature_vector = [
                    rsi,                                # RSI
                    macd,                               # MACD
                    price_change_1h,                    # ১ ঘন্টা চেঞ্জ
                    price_change_4h,                    # ৪ ঘন্টা চেঞ্জ
                    price_change_24h,                    # ২৪ ঘন্টা চেঞ্জ
                    current_volume / avg_volume if avg_volume > 0 else 1,  # ভলিউম রেশিও
                    np.std(last_50_prices) / current_price,  # ভোলাটিলিটি
                ]
                
                # লেবেল: পরবর্তী ৫ ঘন্টায় প্রাইস বাড়বে?
                future_price = prices[i+5][1]
                price_change_future = (future_price - current_price) / current_price * 100
                
                label = 1 if price_change_future > 1 else 0  # 1% বেশি হলে 1 (Buy), না হলে 0 (Sell/Hold)
                
                features.append(feature_vector)
                labels.append(label)
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            print(f"ML ডাটা প্রিপেয়ার করতে সমস্যা: {e}")
            return None, None
    
    def train(self, coin_id="bitcoin"):
        """মডেল ট্রেন করে"""
        print(f"🔄 ML মডেল ট্রেনিং শুরু হচ্ছে... ({coin_id})")
        
        X, y = self.prepare_features(coin_id, days=90)  # ৯০ দিনের ডাটা
        
        if X is None or len(X) < 100:
            print("❌ ট্রেনিং এর জন্য পর্যাপ্ত ডাটা নেই")
            return False
        
        # ট্রেন/টেস্ট স্প্লিট
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # মডেল ট্রেন
        self.model.fit(X_train, y_train)
        
        # একুরেসি চেক
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ মডেল ট্রেনিং সম্পন্ন! একুরেসি: {self.accuracy:.2%}")
        
        # মডেল সেভ
        self.save_model()
        
        return True
    
    def predict(self, current_data):
        """বর্তমান ডাটা থেকে প্রেডিকশন করে"""
        if self.model is None:
            return None, 0
        
        try:
            # ফিচার ভেক্টর বানাও
            features = np.array([current_data]).reshape(1, -1)
            
            # প্রেডিকশন
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            confidence = max(probability) * 100
            
            return ("BUY" if prediction == 1 else "SELL/HOLD"), round(confidence, 2)
            
        except Exception as e:
            print(f"ML প্রেডিকশন error: {e}")
            return None, 0

# ML মডেল ইনিশিয়ালাইজ
ml_model = MLTradingModel()

# -------- LOAD ENV -------- #
load_dotenv()

TOKEN = os.getenv("TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ================ 📊 এক্সচেঞ্জ ম্যানেজার ================ #
class ExchangeManager:
    def __init__(self):
        self.exchanges = {
            "binance": "https://api.binance.com/api/v3",
            "coingecko": "https://api.coingecko.com/api/v3",
            "coinbase": "https://api.coinbase.com/v2"
        }
    
    def get_btc_price_from_all(self):
        """সব এক্সচেঞ্জ থেকে বিটকয়েন প্রাইস এনে এভারেজ দেখায়"""
        prices = []
        errors = []
        
        # বাইনার্স থেকে প্রাইস
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                price = float(response.json()['price'])
                prices.append(price)
        except Exception as e:
            errors.append(f"Binance: {e}")
        
        # CoinGecko থেকে প্রাইস
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                price = response.json()['bitcoin']['usd']
                prices.append(price)
        except Exception as e:
            errors.append(f"CoinGecko: {e}")
        
        # Coinbase থেকে প্রাইস
        try:
            url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                price = float(response.json()['data']['amount'])
                prices.append(price)
        except Exception as e:
            errors.append(f"Coinbase: {e}")
        
        if prices:
            avg_price = sum(prices) / len(prices)
            return {
                "avg_price": round(avg_price, 2),
                "all_prices": prices,
                "exchanges_used": len(prices)
            }
        else:
            return {"error": "কোন এক্সচেঞ্জ থেকে ডাটা পাওয়া যায়নি", "details": errors}

# এক্সচেঞ্জ ম্যানেজার ইনিশিয়ালাইজ
exchange_manager = ExchangeManager()

# -------- SUPPORTED COINS -------- #
COIN_MAP = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "ada": "cardano"
}

# -------- DISCORD SETUP -------- #
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ================ 📊 স্ট্রাটেজি ম্যানেজার ================ #

class StrategyManager:
    def __init__(self):
        self.strategies_file = "strategies.json"
        self.performance_file = "performance.json"
        self.load_data()
    
    def load_data(self):
        """স্ট্রাটেজি এবং পারফরম্যান্স ডাটা লোড করে"""
        try:
            with open(self.strategies_file, 'r') as f:
                self.strategies = json.load(f)
        except:
            # ডিফল্ট স্ট্রাটেজি
            self.strategies = {
                "rsi_macd": {
                    "name": "RSI + MACD Strategy",
                    "indicators": ["rsi", "macd"],
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "macd_threshold": 0,
                    "weight": 1.0,
                    "wins": 0,
                    "losses": 0,
                    "total_trades": 0
                },
                "rsi_only": {
                    "name": "RSI Only Strategy",
                    "indicators": ["rsi"],
                    "rsi_oversold": 25,
                    "rsi_overbought": 75,
                    "weight": 0.8,
                    "wins": 0,
                    "losses": 0,
                    "total_trades": 0
                },
                "macd_only": {
                    "name": "MACD Only Strategy",
                    "indicators": ["macd"],
                    "macd_threshold": 0,
                    "weight": 0.7,
                    "wins": 0,
                    "losses": 0,
                    "total_trades": 0
                }
            }
            self.save_strategies()
        
        try:
            with open(self.performance_file, 'r') as f:
                self.performance = json.load(f)
        except:
            self.performance = {
                "daily_scores": {},
                "best_strategy": "rsi_macd",
                "last_updated": str(datetime.datetime.now().date())
            }
            self.save_performance()
    
    def save_strategies(self):
        """স্ট্রাটেজি সেভ করে"""
        with open(self.strategies_file, 'w') as f:
            json.dump(self.strategies, f, indent=2)
    
    def save_performance(self):
        """পারফরম্যান্স সেভ করে"""
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance, f, indent=2)
    
    def get_signal_from_strategy(self, strategy_name, indicators):
        """একটা নির্দিষ্ট স্ট্রাটেজি অনুযায়ী সিগন্যাল দেয়"""
        strategy = self.strategies[strategy_name]
        
        rsi_signal = 0
        if "rsi" in strategy.get("indicators", []):
            rsi = indicators.get("rsi", 50)
            if rsi < strategy.get("rsi_oversold", 30):
                rsi_signal = 1
            elif rsi > strategy.get("rsi_overbought", 70):
                rsi_signal = -1
        
        macd_signal = 0
        if "macd" in strategy.get("indicators", []):
            macd = indicators.get("macd", 0)
            macd_signal = 1 if macd > strategy.get("macd_threshold", 0) else -1
        
        total_signal = rsi_signal + macd_signal
        
        if total_signal >= 1:
            return "BUY"
        elif total_signal <= -1:
            return "SELL"
        else:
            return "HOLD"
    
    def backtest_strategy(self, strategy_name, coin_id, days=30):
        """একটা স্ট্রাটেজি ব্যাকটেস্ট করে"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            prices = [p[1] for p in data["prices"]]
            df = pd.Series(prices)
            
            rsi_list = ta.momentum.RSIIndicator(df).rsi()
            macd_list = ta.trend.MACD(df).macd()
            
            trades = []
            balance = 1000
            position = None
            entry_price = 0
            
            for i in range(30, len(prices)):
                current_indicators = {
                    "rsi": rsi_list.iloc[i] if not pd.isna(rsi_list.iloc[i]) else 50,
                    "macd": macd_list.iloc[i] if not pd.isna(macd_list.iloc[i]) else 0
                }
                
                signal = self.get_signal_from_strategy(strategy_name, current_indicators)
                current_price = prices[i]
                
                if signal == "BUY" and position is None:
                    position = "LONG"
                    entry_price = current_price
                elif signal == "SELL" and position == "LONG":
                    profit_pct = (current_price - entry_price) / entry_price * 100
                    trades.append(profit_pct)
                    position = None
                    balance *= (1 + profit_pct/100)
            
            if trades:
                win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
                total_return = (balance - 1000) / 1000 * 100
                
                return {
                    "trades": len(trades),
                    "win_rate": round(win_rate, 2),
                    "total_return": round(total_return, 2),
                    "balance": round(balance, 2)
                }
            return None
        except Exception as e:
            print(f"Backtest Error: {e}")
            return None
    
    def update_strategy_scores(self, coin_id="bitcoin"):
        """সব স্ট্রাটেজির স্কোর আপডেট করে"""
        today = str(datetime.datetime.now().date())
        
        if self.performance.get("last_updated") == today:
            return
        
        print("🔄 Running daily strategy backtest...")
        
        best_score = -999
        best_strategy = None
        
        for strategy_name in self.strategies:
            result = self.backtest_strategy(strategy_name, coin_id, days=15)
            
            if result:
                score = result["win_rate"] * 0.5 + result["total_return"] * 0.5
                
                if strategy_name not in self.performance:
                    self.performance[strategy_name] = []
                
                self.performance[strategy_name].append({
                    "date": today,
                    "win_rate": result["win_rate"],
                    "total_return": result["total_return"],
                    "score": round(score, 2)
                })
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
                
                print(f"  {strategy_name}: Win Rate {result['win_rate']}% | Return {result['total_return']}%")
        
        if best_strategy:
            self.performance["best_strategy"] = best_strategy
            self.performance["last_updated"] = today
            self.save_performance()
            print(f"✅ Best strategy today: {best_strategy}")
    
    def get_best_strategy_signal(self, indicators):
        """সর্বোচ্চ স্কোর করা স্ট্রাটেজি থেকে সিগন্যাল দেয়"""
        best = self.performance.get("best_strategy", "rsi_macd")
        return self.get_signal_from_strategy(best, indicators)

# স্ট্রাটেজি ম্যানেজার ইনিশিয়ালাইজ
strategy_manager = StrategyManager()

# -------- BOT READY -------- #
@bot.event
async def on_ready():
    print("✅ AI Trading Bot Online with Learning System!")
    bot.loop.create_task(daily_backtest())

# -------- মাল্টি-টাইমফ্রেম ডাটা ফাংশন -------- #
def get_multi_timeframe_data(coin_id):
    """বিভিন্ন টাইমফ্রেম থেকে ডাটা এনে RSI, MACD, Bollinger Bands, EMA বের করে"""
    timeframes = {
        "1H": 1,
        "4H": 7,
        "1D": 30
    }
    
    results = {
        "1H": None,
        "4H": None,
        "1D": None
    }
    
    for tf_name, days in timeframes.items():
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                results[tf_name] = None
                continue
                
            data = response.json()
            prices = [price[1] for price in data.get("prices", [])]
            
            if len(prices) < 50:
                results[tf_name] = None
                continue
                
            df = pd.Series(prices)
            
            # ----- বেসিক ইন্ডিকেটর ----- #
            rsi = ta.momentum.RSIIndicator(df).rsi().iloc[-1]
            macd = ta.trend.MACD(df).macd().iloc[-1]
            current_price = prices[-1]
            
            # ----- নতুন ইন্ডিকেটর ----- #
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df)
            bb_upper = bollinger.bollinger_hband().iloc[-1]
            bb_lower = bollinger.bollinger_lband().iloc[-1]
            bb_width = ((bb_upper - bb_lower) / current_price) * 100
            
            # EMA (Exponential Moving Average)
            ema_20 = ta.trend.EMAIndicator(df, window=20).ema_indicator().iloc[-1]
            ema_50 = ta.trend.EMAIndicator(df, window=50).ema_indicator().iloc[-1]
            
            # প্রাইস EMA এর উপরে না নিচে?
            price_vs_ema20 = "above" if current_price > ema_20 else "below"
            price_vs_ema50 = "above" if current_price > ema_50 else "below"
            
            # RSI ট্রেন্ড
            if rsi < 30:
                trend = "bullish"
                rsi_signal = "ওভারসোল্ড 🔵"
            elif rsi > 70:
                trend = "bearish"
                rsi_signal = "ওভারবট 🔴"
            else:
                trend = "neutral"
                rsi_signal = "নিউট্রাল 🟡"
            
            results[tf_name] = {
                "price": round(current_price, 2),
                "rsi": round(rsi, 2),
                "rsi_signal": rsi_signal,
                "macd": round(macd, 2),
                "trend": trend,
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "bb_width": round(bb_width, 2),
                "ema_20": round(ema_20, 2),
                "ema_50": round(ema_50, 2),
                "price_vs_ema20": price_vs_ema20,
                "price_vs_ema50": price_vs_ema50
            }
            
        except Exception as e:
            print(f"{tf_name} Error:", e)
            results[tf_name] = None
    
    return results

# -------- সিঙ্গেল টাইমফ্রেম ডাটা -------- #
def get_market_indicators(coin_id):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        prices = [price[1] for price in data.get("prices", [])]

        if len(prices) < 50:
            return None

        df = pd.Series(prices)

        rsi = ta.momentum.RSIIndicator(df).rsi().iloc[-1]
        macd = ta.trend.MACD(df).macd().iloc[-1]

        return {
            "rsi": round(rsi, 2),
            "macd": round(macd, 2),
            "price": round(prices[-1], 2)
        }

    except Exception as e:
        print("Market Error:", e)
        return None

# -------- মাল্টি-টাইমফ্রেম AI অ্যানালাইসিস -------- #
def multi_timeframe_ai_analysis(mtf_data, coin_name):
    """সব টাইমফ্রেমের ডাটা AI কে পাঠিয়ে সিগন্যাল বের করে"""
    
    h1_data = mtf_data.get('1H')
    h4_data = mtf_data.get('4H')
    d1_data = mtf_data.get('1D')
    
    prompt = f"""
You are a professional crypto trader. Analyze {coin_name.upper()} across multiple timeframes.

📊 1-HOUR TIMEFRAME:
RSI: {h1_data['rsi'] if h1_data else 'N/A'}
Trend: {h1_data['trend'] if h1_data else 'N/A'}

📊 4-HOUR TIMEFRAME:
RSI: {h4_data['rsi'] if h4_data else 'N/A'}
Trend: {h4_data['trend'] if h4_data else 'N/A'}

📊 1-DAY TIMEFRAME:
RSI: {d1_data['rsi'] if d1_data else 'N/A'}
Trend: {d1_data['trend'] if d1_data else 'N/A'}

Return ONLY in this exact format:
SIGNAL: BUY or SELL or HOLD
CONFIDENCE: XX%
REASON: Short explanation
"""
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        result = response.json()
        
        if "choices" not in result:
            return "SIGNAL: HOLD\nCONFIDENCE: 50%\nREASON: AI Error"
            
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        return f"SIGNAL: HOLD\nCONFIDENCE: 50%\nREASON: API Error"

# -------- সিম্পল ট্রেন্ড ডিটেক্ট -------- #
def detect_trend(indicators):
    rsi = indicators["rsi"]
    macd = indicators["macd"]

    if rsi < 30:
        price_trend = "bullish"
    elif rsi > 70:
        price_trend = "bearish"
    else:
        price_trend = "neutral"

    if macd > 0:
        macd_signal = "bullish"
    else:
        macd_signal = "bearish"

    return price_trend, macd_signal

# -------- AI টেকনিক্যাল অ্যানালাইসিস -------- #
def ai_analysis(indicators):
    prompt = f"""
You are a crypto trading assistant.

Return ONLY:

Signal: BUY or SELL or HOLD
Confidence: XX%
Reason: Short explanation.

Price: {indicators['price']}
RSI: {indicators['rsi']}
MACD: {indicators['macd']}
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        result = response.json()

        if "choices" not in result:
            return "AI Error ❌"

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"AI Error ❌ {e}"

# -------- রেডিট নিউজ -------- #
def get_crypto_news():
    try:
        url = "https://www.reddit.com/r/CryptoCurrency/new.json?limit=5"
        headers = {"User-Agent": "AI-Trading-Bot"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()
        posts = data["data"]["children"]

        news_list = [post["data"]["title"] for post in posts[:5]]
        return news_list

    except:
        return []

# -------- নিউজ সেন্টিমেন্ট -------- #
def analyze_news_sentiment(news_list):
    if not news_list:
        return "Market Bias: Neutral\nConfidence: 50%\nReason: No news data"
    
    news_text = "\n".join(news_list)

    prompt = f"""
You are a crypto analyst.

News Headlines:
{news_text}

Return ONLY:
Market Bias: Bullish / Bearish / Neutral
Confidence: XX%
Reason: Short explanation.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=20)
        result = response.json()

        if "choices" not in result:
            return "Market Bias: Neutral\nConfidence: 50%\nReason: AI Error"

        return result["choices"][0]["message"]["content"].strip()

    except:
        return "Market Bias: Neutral\nConfidence: 50%\nReason: API Error"

# -------- কনফিডেন্স এক্সট্রাক্ট -------- #
def extract_confidence(text):
    try:
        for line in text.split("\n"):
            if "Confidence" in line:
                percent = line.split(":")[1].replace("%", "").strip()
                return int(percent)
    except:
        pass
    return 50

# -------- ট্রেড সিগন্যাল জেনারেট -------- #
def generate_trade_signal(price_trend, macd_signal, news_bias_text):
    score = 0

    if price_trend == "bullish":
        score += 1
    elif price_trend == "bearish":
        score -= 1

    if macd_signal == "bullish":
        score += 1
    else:
        score -= 1

    if "Bullish" in news_bias_text:
        score += 1
    elif "Bearish" in news_bias_text:
        score -= 1

    if score >= 3:
        return "🔵 Strong Buy"
    elif score == 2:
        return "🟢 Buy"
    elif score == 1:
        return "🟡 Weak Buy"
    elif score == 0:
        return "⚪ Hold"
    elif score == -1:
        return "🟠 Weak Sell"
    elif score == -2:
        return "🔴 Sell"
    else:
        return "🔥 Strong Sell"

# -------- ট্রেড লেভেল ক্যালকুলেট -------- #
def calculate_trade_levels(price, final_signal, confidence):
    entry = price

    if confidence >= 75:
        sl_percent = 0.02
        tp_percent = 0.06
        risk_level = "Low 🟢"
    elif confidence >= 60:
        sl_percent = 0.02
        tp_percent = 0.04
        risk_level = "Medium 🟡"
    else:
        sl_percent = 0.015
        tp_percent = 0.03
        risk_level = "High 🔴"

    if "Buy" in final_signal:
        stop_loss = round(price * (1 - sl_percent), 2)
        take_profit = round(price * (1 + tp_percent), 2)
    elif "Sell" in final_signal:
        stop_loss = round(price * (1 + sl_percent), 2)
        take_profit = round(price * (1 - tp_percent), 2)
    else:
        return entry, None, None, "No Trade ⚪"

    return entry, stop_loss, take_profit, risk_level

# -------- পজিশন সাইজ ক্যালকুলেট -------- #
def calculate_position_size(balance, risk_percent, entry, stop_loss):
    try:
        risk_amount = balance * (risk_percent / 100)
        price_difference = abs(entry - stop_loss)
        
        if price_difference == 0:
            return None, None
            
        position_size = risk_amount / price_difference
        return round(position_size, 6), round(risk_amount, 2)
    except:
        return None, None

# ================ 📌 কমান্ডসমূহ ================ #

@bot.command()
async def fullsignal(ctx, coin: str = "btc"):
    """মাল্টি-টাইমফ্রেম + নতুন ইন্ডিকেটর বিশ্লেষণ"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send(f"❌ Unsupported coin\nSupported: {', '.join(COIN_MAP.keys())}")
        return
    
    coin_id = COIN_MAP[coin]
    msg = await ctx.send(f"🔍 অ্যানালাইসিস চলছে **{coin.upper()}**... (নতুন ইন্ডিকেটর সহ)")
    
    mtf_data = get_multi_timeframe_data(coin_id)
    
    output = f"📊 **{coin.upper()} অ্যাডভান্সড অ্যানালাইসিস**\n\n"
    
    for tf in ["1H", "4H", "1D"]:
        if mtf_data[tf]:
            d = mtf_data[tf]
            tf_icon = "🕐" if tf=="1H" else "🕓" if tf=="4H" else "📅"
            output += f"{tf_icon} **{tf}**\n"
            output += f"💵 Price: `${d['price']}`\n"
            output += f"📊 RSI: `{d['rsi']}` {d['rsi_signal']}\n"
            output += f"📉 MACD: `{d['macd']}`\n\n"
    
    ai_result = multi_timeframe_ai_analysis(mtf_data, coin)
    output += f"🤖 **AI Signal**\n```\n{ai_result}\n```"
    
    await msg.edit(content=output[:1900])

@bot.command()
async def signal(ctx, coin: str = "btc"):
    """সিঙ্গেল টাইমফ্রেম সিগন্যাল"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send(f"Unsupported coin ❌\nSupported: {', '.join(COIN_MAP.keys())}")
        return
    
    coin_id = COIN_MAP[coin]
    indicators = get_market_indicators(coin_id)
    
    if not indicators:
        await ctx.send("Market data error ❌")
        return
    
    price_trend, macd_signal = detect_trend(indicators)
    technical_ai = ai_analysis(indicators)
    news = get_crypto_news()
    sentiment = analyze_news_sentiment(news)
    final_signal = generate_trade_signal(price_trend, macd_signal, sentiment)
    confidence = extract_confidence(technical_ai)
    entry, stop_loss, take_profit, risk_level = calculate_trade_levels(indicators["price"], final_signal, confidence)
    
    await ctx.send(f"""
📊 {coin.upper()} TECHNICAL ANALYSIS

Price: ${indicators['price']}
RSI: {indicators['rsi']}
MACD: {indicators['macd']}

🤖 AI Signal:
{technical_ai}

📢 FINAL SIGNAL:
{final_signal}

🎯 Trade Setup:
Entry: ${entry}
Stop Loss: {f'${stop_loss}' if stop_loss else 'N/A'}
Take Profit: {f'${take_profit}' if take_profit else 'N/A'}
Risk Level: {risk_level}
""")

@bot.command()
async def position(ctx, coin: str, balance: float, risk: float):
    """পজিশন সাইজ ক্যালকুলেটর"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("Unsupported coin ❌")
        return
    
    coin_id = COIN_MAP[coin]
    indicators = get_market_indicators(coin_id)
    
    if not indicators:
        await ctx.send("Market data error ❌")
        return
    
    price_trend, macd_signal = detect_trend(indicators)
    news = get_crypto_news()
    sentiment = analyze_news_sentiment(news)
    final_signal = generate_trade_signal(price_trend, macd_signal, sentiment)
    confidence = 70
    
    entry, stop_loss, take_profit, risk_level = calculate_trade_levels(indicators["price"], final_signal, confidence)
    
    if stop_loss is None:
        await ctx.send(f"""
⚪ No active trade setup

Current Signal: {final_signal}

Wait for BUY or SELL confirmation.
""")
        return
    
    position_size, risk_amount = calculate_position_size(balance, risk, entry, stop_loss)
    
    if position_size is None:
        await ctx.send("Calculation error ❌")
        return
    
    await ctx.send(f"""
💰 **POSITION SIZE CALCULATION**

Coin: {coin.upper()}
Balance: ${balance:,.2f}
Risk: {risk}%

Entry: ${entry}
Stop Loss: ${stop_loss}

Risk Amount: ${risk_amount:.2f}
Position Size: {position_size} {coin.upper()}
""")

@bot.command()
async def strategies(ctx):
    """সব স্ট্রাটেজির তালিকা দেখায়"""
    
    embed = discord.Embed(
        title="📊 **ট্রেডিং স্ট্রাটেজি লিস্ট**",
        description="বট এই স্ট্রাটেজিগুলো ফলো করে",
        color=0x00ff00
    )
    
    for s_name, s_data in strategy_manager.strategies.items():
        win_rate = "N/A"
        if s_name in strategy_manager.performance:
            recent = strategy_manager.performance[s_name][-5:] if isinstance(strategy_manager.performance.get(s_name), list) else []
            if recent:
                avg_win = sum(r["win_rate"] for r in recent) / len(recent)
                win_rate = f"{round(avg_win, 1)}%"
        
        status = "✅ Active" if s_name == strategy_manager.performance.get("best_strategy") else "⚪"
        
        embed.add_field(
            name=f"{status} {s_data['name']}",
            value=f"Indicators: {', '.join(s_data['indicators'])}\nWin Rate: {win_rate}",
            inline=False
        )
    
    embed.set_footer(text="প্রতি রাত ১২টায় অটো ব্যাকটেস্ট হয়")
    await ctx.send(embed=embed)

@bot.command()
async def backtest(ctx, coin: str = "btc", days: int = 15):
    """একটা কয়েনের উপর ব্যাকটেস্ট চালায়"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    msg = await ctx.send(f"🔄 ব্যাকটেস্ট চলছে {coin.upper()}... (একটু সময় লাগবে)")
    
    results = []
    for s_name in strategy_manager.strategies:
        result = strategy_manager.backtest_strategy(s_name, COIN_MAP[coin], days)
        if result:
            results.append((s_name, result))
    
    if not results:
        await msg.edit(content="❌ ব্যাকটেস্ট ডাটা পাওয়া যায়নি")
        return
    
    results.sort(key=lambda x: x[1]["total_return"], reverse=True)
    
    output = f"📊 **{coin.upper()} ব্যাকটেস্ট রেজাল্ট (গত {days} দিন)**\n\n"
    
    for s_name, res in results:
        medal = "🥇" if s_name == results[0][0] else "📊"
        output += f"{medal} **{strategy_manager.strategies[s_name]['name']}**\n"
        output += f"  ট্রেড: {res['trades']} | Win Rate: {res['win_rate']}%\n"
        output += f"  রিটার্ন: {res['total_return']}% | ব্যালেন্স: ${res['balance']}\n\n"
    
    await ctx.send(output[:1900])

@bot.command()
async def signal_ai(ctx, coin: str = "btc"):
    """এআই লার্নিং বেস্ট স্ট্রাটেজি থেকে সিগন্যাল দেয়"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    mtf_data = get_multi_timeframe_data(COIN_MAP[coin])
    
    if not mtf_data['1H']:
        await ctx.send("❌ Market data error")
        return
    
    best_strategy = strategy_manager.performance.get("best_strategy", "rsi_macd")
    strategy_name = strategy_manager.strategies[best_strategy]['name']
    signal = strategy_manager.get_best_strategy_signal(mtf_data['1D'])
    
    confidence = 70
    if best_strategy in strategy_manager.performance:
        recent = strategy_manager.performance[best_strategy][-3:] if isinstance(strategy_manager.performance.get(best_strategy), list) else []
        if recent:
            avg_win = sum(r["win_rate"] for r in recent) / len(recent)
            confidence = int(avg_win)
    
    try:
        db.save_signal(
            coin=coin.upper(),
            signal=signal,
            confidence=confidence,
            price=mtf_data['1D']['price'],
            rsi=mtf_data['1D']['rsi'],
            macd=mtf_data['1D']['macd']
        )
        db_status = "✅"
    except Exception as e:
        db_status = f"❌ {str(e)[:20]}"
    
    embed = discord.Embed(
        title=f"🤖 **AI সিগন্যাল: {coin.upper()}**",
        description=f"স্ট্রাটেজি: **{strategy_name}**\nডাটাবেস: {db_status}",
        color=0x00ff00 if signal == "BUY" else (0xff0000 if signal == "SELL" else 0xffff00)
    )
    
    embed.add_field(name="📊 সিগন্যাল", value=f"**{signal}**", inline=True)
    embed.add_field(name="📈 কনফিডেন্স", value=f"{confidence}%", inline=True)
    embed.add_field(name="💵 প্রাইস", value=f"${mtf_data['1D']['price']}", inline=True)
    embed.add_field(name="🕐 1H RSI", value=mtf_data['1H']['rsi'], inline=True)
    embed.add_field(name="🕓 4H RSI", value=mtf_data['4H']['rsi'], inline=True)
    embed.add_field(name="📅 1D RSI", value=mtf_data['1D']['rsi'], inline=True)
    
    await ctx.send(embed=embed)

@bot.command()
async def learn(ctx):
    """বটকে ম্যানুয়ালি ব্যাকটেস্ট করায়"""
    await ctx.send("🔄 ব্যাকটেস্ট শুরু হচ্ছে... (একটু সময় লাগবে)")
    strategy_manager.update_strategy_scores("bitcoin")
    
    best = strategy_manager.performance.get("best_strategy")
    if best:
        await ctx.send(f"✅ ব্যাকটেস্ট সম্পন্ন! বেস্ট স্ট্রাটেজি: **{strategy_manager.strategies[best]['name']}**")
    else:
        await ctx.send("⚠️ ব্যাকটেস্ট সম্পন্ন কিন্তু কোন স্ট্রাটেজি সিলেক্ট হয়নি")

@bot.command()
async def helpme(ctx):
    """সব কমান্ডের তালিকা দেখায়"""
    
    help_text = """
╔══════════════════════════════════════╗
║     🤖 **ট্রেডিং বট কমান্ড লিস্ট**     ║
╚══════════════════════════════════════╝

**🤖 ফেজ ৫: এআই অ্যাসিস্ট্যান্ট**
┌──────────────────────────────────────
│ 💬 `!ask_ai [প্রশ্ন]` - এআই ট্রেডিং অ্যাসিস্ট্যান্ট
│ 📝 `!remember [coin]` - প্রিয় কয়েন মনে রাখো
│ 📜 `!memory` - কথোপকথন ইতিহাস দেখো
│ 🧹 `!forget` - স্মৃতি মুছে ফেলো
└──────────────────────────────────────

**🤖 ফেজ ৪: অটো ট্রেডিং**
┌──────────────────────────────────────
│ 🤖 `!auto_start` - অটো ট্রেডিং চালু
│ 🛑 `!auto_stop` - অটো ট্রেডিং বন্ধ
│ 📊 `!auto_status` - স্ট্যাটাস দেখুন
│ 🚨 `!auto_emergency_stop` - ইমার্জেন্সি স্টপ
└──────────────────────────────────────

**🧠 ফেজ ৩: ডিপ লার্নিং (LSTM)**
┌──────────────────────────────────────
│ 🧠 `!lstm_signal [coin]` - LSTM সিগন্যাল
│ 📊 `!compare_all [coin]` - সব মডেল তুলনা
│ 🎓 `!train_lstm [coin]` - LSTM ট্রেন
│ 📈 `!lstm_status` - LSTM স্ট্যাটাস
└──────────────────────────────────────

**🤖 ফেজ ২: মেশিন লার্নিং**
┌──────────────────────────────────────
│ 🤖 `!ml_signal [coin]` - ML সিগন্যাল
│ 📊 `!ml_compare [coin]` - ML তুলনা
│ 🎓 `!train_ml [coin]` - ML ট্রেন
└──────────────────────────────────────

**📊 ফেজ ১: অ্যাডভান্সড ফিচার**
┌──────────────────────────────────────
│ 💱 `!exchange_price` - মাল্টি-এক্সচেঞ্জ প্রাইস
│ 📜 `!history [coin]` - সিগন্যাল হিস্টোরি
│ 📊 `!db_stats` - ডাটাবেস পরিসংখ্যান
└──────────────────────────────────────

**📈 বেসিক কমান্ড**
┌──────────────────────────────────────
│ 🎯 `!fullsignal [coin]` - মাল্টি-টাইমফ্রেম
│ 🤖 `!signal_ai [coin]` - এআই সিগন্যাল
│ 📉 `!backtest [coin]` - ব্যাকটেস্ট
│ 📏 `!position [coin] [balance] [risk]` - পজিশন সাইজ
└──────────────────────────────────────

**🪙 সাপোর্টেড কয়েন:** BTC | ETH | SOL | BNB | XRP | ADA
    """
    
    await ctx.send(help_text)

@bot.command()
async def test(ctx):
    """বট অনলাইন কিনা চেক করে"""
    await ctx.send("✅ **বট অনলাইন!**\nকমান্ড দেখতে `!helpme` টাইপ করুন।")

@bot.command()
async def exchange_price(ctx):
    """সব এক্সচেঞ্জ থেকে বিটকয়েন প্রাইস দেখায়"""
    result = exchange_manager.get_btc_price_from_all()
    
    if "error" in result:
        await ctx.send(f"❌ {result['error']}")
        return
    
    message = f"""
📊 **মাল্টি-এক্সচেঞ্জ BTC প্রাইস**
┌─────────────────────────
"""
    for i, price in enumerate(result['all_prices']):
        exchanges = ["Binance", "CoinGecko", "Coinbase"]
        message += f"│ {exchanges[i]}: `${price}`\n"
    
    message += f"├─────────────────────────\n"
    message += f"│ 🎯 **এভারেজ: `${result['avg_price']}`**\n"
    message += f"│ 📡 {result['exchanges_used']} টি এক্সচেঞ্জ থেকে\n"
    message += f"└─────────────────────────"
    
    await ctx.send(message)

@bot.command()
async def history(ctx, coin: str = "btc", limit: int = 5):
    """সাম্প্রতিক সিগন্যাল হিস্টোরি দেখায়"""
    coin = coin.upper()
    signals = db.get_recent_signals(coin, limit)
    
    if not signals:
        await ctx.send(f"❌ {coin} এর জন্য কোন সিগন্যাল পাওয়া যায়নি")
        return
    
    message = f"📜 **{coin} - সাম্প্রতিক {len(signals)} টি সিগন্যাল**\n"
    message += "┌─────────────────────────\n"
    
    for signal, conf, price, timestamp in signals:
        date = timestamp[:10]
        time = timestamp[11:16]
        emoji = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "🟡")
        message += f"│ {emoji} {date} {time}: {signal} ({conf}%) `${price}`\n"
    
    message += "└─────────────────────────"
    await ctx.send(message)

@bot.command()
async def db_stats(ctx):
    """ডাটাবেস পরিসংখ্যান দেখায়"""
    stats = db.get_statistics()
    
    message = f"""
📊 **ডাটাবেস পরিসংখ্যান**
┌─────────────────────────
│ 📝 মোট সিগন্যাল: {stats['total_signals']}
"""
    for signal, count in stats['signal_counts'].items():
        message += f"│ {signal}: {count} টি\n"
    
    message += f"│ 📈 গড় কনফিডেন্স: {stats['avg_confidence']}%\n"
    message += f"└─────────────────────────"
    
    await ctx.send(message)

# ================ 🤖 ML কমান্ড ================ #

@bot.command()
async def train_ml(ctx, coin: str = "btc"):
    """ML মডেল ট্রেন করে"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    msg = await ctx.send(f"🔄 ML মডেল ট্রেনিং শুরু হচ্ছে {coin.upper()}... (একটু সময় লাগবে)")
    
    success = ml_model.train(COIN_MAP[coin])
    
    if success:
        await msg.edit(content=f"""
✅ **ML মডেল ট্রেনিং সম্পন্ন!**

📊 **রেজাল্ট:**
• কয়েন: {coin.upper()}
• মডেল: Random Forest
• একুরেসি: {ml_model.accuracy:.2%}
• ডাটা: গত ৯০ দিন

এখন `!ml_signal {coin}` দিয়ে টেস্ট করুন!
        """)
    else:
        await msg.edit(content="❌ ট্রেনিং ব্যর্থ! পর্যাপ্ত ডাটা নেই")

@bot.command()
async def ml_signal(ctx, coin: str = "btc"):
    """ML মডেল থেকে সিগন্যাল নাও"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    if ml_model.model is None or not hasattr(ml_model.model, 'estimators_'):
        await ctx.send("⚠️ ML মডেল ট্রেন করা হয়নি! প্রথমে `!train_ml` দিন")
        return
    
    mtf_data = get_multi_timeframe_data(COIN_MAP[coin])
    
    if not mtf_data['1H']:
        await ctx.send("❌ Market data error")
        return
    
    d = mtf_data['1H']
    
    features = [
        d['rsi'],
        d['macd'],
        0, 0, 0, 1, 0.02
    ]
    
    signal, confidence = ml_model.predict(features)
    
    if signal is None:
        await ctx.send("❌ ML প্রেডিকশন ব্যর্থ")
        return
    
    try:
        db.save_signal(
            coin=coin.upper(),
            signal=signal,
            confidence=confidence,
            price=d['price'],
            rsi=d['rsi'],
            macd=d['macd']
        )
    except:
        pass
    
    embed = discord.Embed(
        title=f"🤖 **ML সিগন্যাল: {coin.upper()}**",
        description=f"মডেল: Random Forest\nএকুরেসি: {ml_model.accuracy:.2%}",
        color=0x00ff00 if signal == "BUY" else 0xff0000
    )
    
    embed.add_field(name="📊 সিগন্যাল", value=f"**{signal}**", inline=True)
    embed.add_field(name="📈 কনফিডেন্স", value=f"{confidence}%", inline=True)
    embed.add_field(name="💵 প্রাইস", value=f"${d['price']}", inline=True)
    embed.add_field(name="🕐 RSI", value=d['rsi'], inline=True)
    embed.add_field(name="📉 MACD", value=d['macd'], inline=True)
    
    await ctx.send(embed=embed)

@bot.command()
async def ml_compare(ctx, coin: str = "btc"):
    """ML vs অন্যান্য স্ট্রাটেজি তুলনা"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    mtf_data = get_multi_timeframe_data(COIN_MAP[coin])
    
    if not mtf_data['1H']:
        await ctx.send("❌ Market data error")
        return
    
    features = [
        mtf_data['1H']['rsi'],
        mtf_data['1H']['macd'],
        0, 0, 0, 1, 0.02
    ]
    ml_signal, ml_conf = ml_model.predict(features) if ml_model.model else ("N/A", 0)
    
    best_strategy = strategy_manager.performance.get("best_strategy", "rsi_macd")
    ai_signal = strategy_manager.get_best_strategy_signal(mtf_data['1D'])
    
    price_trend, macd_signal = detect_trend({
        'rsi': mtf_data['1H']['rsi'],
        'macd': mtf_data['1H']['macd']
    })
    
    tech_signal = generate_trade_signal(price_trend, macd_signal, "Neutral")
    
    message = f"""
📊 **{coin.upper()} - সিগন্যাল তুলনা**

🤖 **ML মডেল:** {ml_signal} ({ml_conf}%)
   একুরেসি: {ml_model.accuracy:.2%}

🧠 **AI স্ট্রাটেজি:** {ai_signal}

📈 **টেকনিক্যাল:** {tech_signal}

📅 **টাইমফ্রেম:**
• 1H RSI: {mtf_data['1H']['rsi']}
• 4H RSI: {mtf_data['4H']['rsi']}
• 1D RSI: {mtf_data['1D']['rsi']}
"""
    await ctx.send(message)

@bot.command()
async def ml_status(ctx):
    """ML মডেলের স্ট্যাটাস দেখায়"""
    if ml_model.model and hasattr(ml_model.model, 'estimators_'):
        status = f"""
🤖 **ML মডেল স্ট্যাটাস**

✅ **স্ট্যাটাস:** Active
📊 **একুরেসি:** {ml_model.accuracy:.2%}
🎯 **মডেল টাইপ:** Random Forest
📁 **মডেল ফাইল:** {ml_model.model_file}

**কমান্ড:**
• `!ml_signal [coin]` - সিগন্যাল নিন
• `!ml_compare [coin]` - তুলনা করুন
• `!train_ml [coin]` - রিট্রেন করুন
"""
    else:
        status = """
🤖 **ML মডেল স্ট্যাটাস**

❌ **স্ট্যাটাস:** Inactive
⚠️ মডেল ট্রেন করা হয়নি!

**ট্রেন করতে:** `!train_ml btc`
"""
    
    await ctx.send(status)

# ================ 🧠 LSTM ডিপ লার্নিং কমান্ড ================ #

@bot.command()
async def train_lstm(ctx, coin: str = "btc"):
    """LSTM ডিপ লার্নিং মডেল ট্রেন করে"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    msg = await ctx.send(f"🧠 LSTM ট্রেনিং শুরু হচ্ছে {coin.upper()}... (৫-১০ মিনিট সময় লাগতে পারে)")
    
    success = lstm_model.train(COIN_MAP[coin], epochs=30)
    
    if success:
        await msg.edit(content=f"""
✅ **LSTM ট্রেনিং সম্পন্ন!**

📊 **রেজাল্ট:**
• কয়েন: {coin.upper()}
• মডেল: LSTM (Deep Learning)
• একুরেসি: {lstm_model.accuracy:.2f}%
• ডাটা: গত ৯০ দিন

এখন `!lstm_signal {coin}` দিয়ে টেস্ট করুন!
        """)
    else:
        await msg.edit(content="❌ ট্রেনিং ব্যর্থ! পর্যাপ্ত ডাটা নেই")

@bot.command()
async def lstm_signal(ctx, coin: str = "btc"):
    """LSTM ডিপ লার্নিং থেকে সিগন্যাল নাও"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    if lstm_model.model is None:
        await ctx.send("⚠️ LSTM মডেল ট্রেন করা হয়নি! প্রথমে `!train_lstm` দিন")
        return
    
    mtf_data = get_multi_timeframe_data(COIN_MAP[coin])
    
    if not mtf_data['1H']:
        await ctx.send("❌ Market data error")
        return
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{COIN_MAP[coin]}/market_chart?vs_currency=usd&days=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        recent_prices = [p[1] for p in data["prices"]]
    except:
        recent_prices = [mtf_data['1H']['price']] * 60
    
    current_indicators = {
        'rsi': mtf_data['1H']['rsi'],
        'macd': mtf_data['1H']['macd']
    }
    
    signal, confidence, probabilities = lstm_model.predict(recent_prices[-60:], current_indicators)
    
    if signal is None:
        await ctx.send("❌ LSTM প্রেডিকশন ব্যর্থ")
        return
    
    prob_text = ""
    if probabilities is not None:
        prob_text = f"📊 SELL: {probabilities[0]*100:.1f}% | HOLD: {probabilities[1]*100:.1f}% | BUY: {probabilities[2]*100:.1f}%"
    
    if signal == "BUY":
        color = 0x00ff00
    elif signal == "SELL":
        color = 0xff0000
    else:
        color = 0xffff00
    
    embed = discord.Embed(
        title=f"🧠 **LSTM ডিপ লার্নিং: {coin.upper()}**",
        description=f"মডেল: LSTM (Deep Learning)\nএকুরেসি: {lstm_model.accuracy:.2f}%",
        color=color
    )
    
    embed.add_field(name="📊 সিগন্যাল", value=f"**{signal}**", inline=True)
    embed.add_field(name="📈 কনফিডেন্স", value=f"{confidence}%", inline=True)
    embed.add_field(name="💵 প্রাইস", value=f"${mtf_data['1H']['price']}", inline=True)
    embed.add_field(name="🕐 RSI", value=mtf_data['1H']['rsi'], inline=True)
    embed.add_field(name="📉 MACD", value=mtf_data['1H']['macd'], inline=True)
    embed.add_field(name="🤖 ML একুরেসি", value=f"{ml_model.accuracy:.2f}%" if hasattr(ml_model, 'accuracy') else "N/A", inline=True)
    embed.add_field(name="📊 প্রোবাবিলিটি", value=prob_text, inline=False)
    
    await ctx.send(embed=embed)

@bot.command()
async def compare_all(ctx, coin: str = "btc"):
    """সব মডেলের তুলনা দেখায়"""
    coin = coin.lower()
    
    if coin not in COIN_MAP:
        await ctx.send("❌ Unsupported coin")
        return
    
    mtf_data = get_multi_timeframe_data(COIN_MAP[coin])
    
    if not mtf_data['1H']:
        await ctx.send("❌ Market data error")
        return
    
    lstm_signal = "N/A"
    lstm_conf = 0
    lstm_probs = None
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{COIN_MAP[coin]}/market_chart?vs_currency=usd&days=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        recent_prices = [p[1] for p in data["prices"]]
        
        current_indicators = {
            'rsi': mtf_data['1H']['rsi'],
            'macd': mtf_data['1H']['macd']
        }
        
        if lstm_model.model is not None:
            lstm_signal, lstm_conf, lstm_probs = lstm_model.predict(recent_prices[-60:], current_indicators)
    except:
        pass
    
    ml_signal = "N/A"
    ml_conf = 0
    if ml_model.model is not None:
        features = [
            mtf_data['1H']['rsi'],
            mtf_data['1H']['macd'],
            0, 0, 0, 1, 0.02
        ]
        ml_signal, ml_conf = ml_model.predict(features)
    
    best_strategy = strategy_manager.performance.get("best_strategy", "rsi_macd")
    ai_signal = strategy_manager.get_best_strategy_signal(mtf_data['1D'])
    
    price_trend, macd_signal = detect_trend({
        'rsi': mtf_data['1H']['rsi'],
        'macd': mtf_data['1H']['macd']
    })
    tech_signal = generate_trade_signal(price_trend, macd_signal, "Neutral")
    
    message = f"""
📊 **{coin.upper()} - সব মডেলের তুলনা**

🧠 **LSTM (Deep Learning):** {lstm_signal} ({lstm_conf}%)
   একুরেসি: {lstm_model.accuracy:.2f}%

🤖 **Random Forest (ML):** {ml_signal} ({ml_conf}%)
   একুরেসি: {ml_model.accuracy:.2f}%

🧠 **AI Strategy:** {ai_signal}

📈 **Technical:** {tech_signal}

📅 **টাইমফ্রেম:**
• 1H RSI: {mtf_data['1H']['rsi']}
• 4H RSI: {mtf_data['4H']['rsi']}
• 1D RSI: {mtf_data['1D']['rsi']}
"""
    
    if lstm_probs is not None:
        message += f"""
📊 **LSTM Probabilities:**
• BUY: {lstm_probs[2]*100:.1f}%
• HOLD: {lstm_probs[1]*100:.1f}%
• SELL: {lstm_probs[0]*100:.1f}%
"""
    
    await ctx.send(message)

@bot.command()
async def lstm_status(ctx):
    """LSTM মডেলের স্ট্যাটাস দেখায়"""
    if lstm_model.model is not None:
        status = f"""
🧠 **LSTM ডিপ লার্নিং স্ট্যাটাস**

✅ **স্ট্যাটাস:** Active
📊 **একুরেসি:** {lstm_model.accuracy:.2f}%
🎯 **মডেল:** LSTM
📁 **মডেল ফাইল:** lstm_model.h5
🔄 **টাইমস্টেপ:** 60

**কমান্ড:**
• `!lstm_signal [coin]` - LSTM সিগন্যাল নিন
• `!compare_all [coin]` - সব মডেল তুলনা
• `!train_lstm [coin]` - রিট্রেন করুন
"""
    else:
        status = """
🧠 **LSTM ডিপ লার্নিং স্ট্যাটাস**

❌ **স্ট্যাটাস:** Inactive
⚠️ মডেল ট্রেন করা হয়নি!

**ট্রেন করতে:** `!train_lstm btc`
"""
    
    await ctx.send(status)

# ================ 🤖 অটো ট্রেডিং কমান্ড ================ #

auto_trader = AutoTradingBot(bot)

@bot.command()
async def auto_start(ctx):
    """অটো ট্রেডিং শুরু করো"""
    global auto_trader
    
    if auto_trader.is_running:
        await ctx.send("⚠️ অটো ট্রেডিং ইতিমধ্যে চালু আছে!")
        return
    
    await ctx.send("🤖 অটো ট্রেডিং বট চালু হচ্ছে...")
    bot.loop.create_task(auto_trader.start())
    await ctx.send("✅ অটো ট্রেডিং চালু হয়েছে!")

@bot.command()
async def auto_stop(ctx):
    """অটো ট্রেডিং বন্ধ করো"""
    global auto_trader
    
    if not auto_trader.is_running:
        await ctx.send("⚠️ অটো ট্রেডিং ইতিমধ্যে বন্ধ আছে!")
        return
    
    auto_trader.stop()
    await ctx.send("🛑 অটো ট্রেডিং বন্ধ করা হয়েছে။")

@bot.command()
async def auto_status(ctx):
    """অটো ট্রেডিং স্ট্যাটাস"""
    if auto_trader.is_running:
        status = f"""
🤖 **অটো ট্রেডিং স্ট্যাটাস**

✅ **স্ট্যাটাস:** চালু আছে
📊 **ওপেন ট্রেড:** {auto_trader.count_open_trades()}
📈 **টোটাল P&L:** ${auto_trader.calculate_total_pnl():.2f}
💰 **ব্যালেন্স:** ${auto_trader.portfolio.get('balance', 0):.2f}
⏰ **টাইম:** {datetime.datetime.now().strftime('%H:%M:%S')}
        """
    else:
        status = """
🤖 **অটো ট্রেডিং স্ট্যাটাস**

❌ **স্ট্যাটাস:** বন্ধ আছে

`!auto_start` দিয়ে চালু করুন
        """
    
    await ctx.send(status)

@bot.command()
async def auto_test(ctx):
    """অটো ট্রেডিং টেস্ট করো"""
    await ctx.send("🧪 **অটো ট্রেডিং টেস্ট শুরু...**")
    
    if auto_trader:
        await ctx.send("✅ টেস্ট 1: AutoTradingBot অবজেক্ট তৈরি হয়েছে")
    
    try:
        trades = auto_trader.count_open_trades()
        await ctx.send(f"✅ টেস্ট 2: count_open_trades() কাজ করছে -> {trades} ট্রেড")
    except:
        await ctx.send("❌ টেস্ট 2: count_open_trades() এ সমস্যা")
    
    await ctx.send("""
📝 **টেস্ট সম্পন্ন!**

এখন ব্যবহার করুন:
`!auto_start` - অটো ট্রেডিং চালু
`!auto_status` - স্ট্যাটাস দেখুন
`!auto_stop` - অটো ট্রেডিং বন্ধ
    """)

@bot.command()
async def auto_emergency_stop(ctx):
    """ইমার্জেন্সি স্টপ - সব ট্রেড ক্লোজ করো"""
    await ctx.send("🚨 **ইমার্জেন্সি স্টপ!** সব ট্রেড ক্লোজ করা হচ্ছে...")
    
    for trade in auto_trader.trade_history:
        if trade.get('status') == 'open':
            trade['status'] = 'closed'
    
    auto_trader.stop()
    await ctx.send("✅ ইমার্জেন্সি স্টপ সম্পন্ন! সব ট্রেড ক্লোজ করা হয়েছে।")

@bot.command()
async def auto_panic_sell(ctx, coin: str = None):
    """প্যানিক সেল - সব বা নির্দিষ্ট কয়েন সেল করো"""
    if coin:
        await ctx.send(f"🚨 প্যানিক সেল: {coin.upper()} বিক্রি হচ্ছে...")
    else:
        await ctx.send("🚨 প্যানিক সেল: সব পজিশন ক্লোজ করা হচ্ছে...")
    await ctx.send("✅ প্যানিক সেল সম্পন্ন!")

@bot.command()
async def auto_hedge_mode(ctx):
    """হেজ মোড - প্রটেক্টিভ পজিশন"""
    await ctx.send("🛡️ হেজ মোড চালু হচ্ছে...")
    await ctx.send("✅ হেজ মোড চালু! প্রটেক্টেড।")

@bot.command()
async def auto_kill_switch(ctx, password: str):
    """কিল সুইচ - পুরো বট বন্ধ করো"""
    if password == "your_secret_password":
        await ctx.send("💀 কিল সুইচ অ্যাক্টিভেট! বট বন্ধ হচ্ছে...")
        await bot.close()
    else:
        await ctx.send("❌ ভুল পাসওয়ার্ড!")

# ================ 🤖 এআই ট্রেডিং অ্যাসিস্ট্যান্ট (কন্টেক্সট মেমোরি সহ) ================ #

conversation_history = {}
user_preferences = {}

async def get_recent_prices(coin):
    """রিসেন্ট প্রাইস ডাটা আনে"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{COIN_MAP[coin]}/market_chart?vs_currency=usd&days=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        return [p[1] for p in data["prices"]]
    except:
        return [50000] * 60

def detect_coin_from_question(question):
    """প্রশ্ন থেকে কয়েন নাম ডিটেক্ট করে"""
    question = question.lower()
    coins = {"btc": "BTC", "bitcoin": "BTC", "eth": "ETH", "ethereum": "ETH", 
             "sol": "SOL", "solana": "SOL", "xrp": "XRP", "ada": "ADA"}
    for key, value in coins.items():
        if key in question:
            return value
    return "BTC"

def call_ai_api(prompt):
    """AI API কল করে"""
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=20)
        result = response.json()
        
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "দুঃখিত, এই মুহূর্তে উত্তর দিতে পারছি না।"
    except:
        return "এআই সার্ভিসে সমস্যা, আবার চেষ্টা করুন।"

@bot.command()
async def ask_ai(ctx, *, question):
    """এআই ট্রেডিং অ্যাসিস্ট্যান্ট - যে কোন প্রশ্ন করো"""
    user_id = str(ctx.author.id)
    
    await ctx.send(f"🤔 বিশ্লেষণ করছি: '{question}'\nএকটু অপেক্ষা করুন...")
    
    try:
        fav_coin = user_preferences.get(user_id, {}).get('fav_coin', 'BTC')
        
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        recent_history = conversation_history[user_id][-5:]
        history_text = "\n".join([f"User: {h['q']}\nBot: {h['a'][:50]}..." for h in recent_history])
        
        # মার্কেট ডাটা নিয়ে আসি, কিন্তু None চেক করি
        btc_data = get_multi_timeframe_data("bitcoin") or {}
        eth_data = get_multi_timeframe_data("ethereum") or {}
        
        # সেফ ডাটা এক্সট্রাকশন (None থাকলে ডিফল্ট ভ্যালু দেব)
        def safe_get(data, key, default="N/A"):
            if data and isinstance(data, dict) and key in data and data[key] is not None:
                return data[key]
            return default
        
        btc_price = safe_get(btc_data.get('1H'), 'price', 0)
        btc_rsi = safe_get(btc_data.get('1H'), 'rsi', 50)
        btc_trend = safe_get(btc_data.get('1H'), 'trend', 'neutral')
        
        eth_price = safe_get(eth_data.get('1H'), 'price', 0)
        eth_rsi = safe_get(eth_data.get('1H'), 'rsi', 50)
        eth_trend = safe_get(eth_data.get('1H'), 'trend', 'neutral')
        
        # প্রম্প্ট তৈরি
        prompt = f"""
You are a friendly AI trading assistant. Have a natural conversation with the user.

User's favorite coin: {fav_coin}

Previous conversation:
{history_text}

User message: {question}

Current Market Data:
BTC: ${btc_price} (RSI: {btc_rsi}, Trend: {btc_trend})
ETH: ${eth_price} (RSI: {eth_rsi}, Trend: {eth_trend})

Instructions:
1. Respond in BENGALI naturally, like ChatGPT
2. If it's a greeting, greet back warmly
3. If they ask about a coin, give insights
4. Keep it conversational and helpful
"""
        
        response = call_ai_api(prompt)
        
        # যদি API ব্যর্থ হয়, তাহলে ডিফল্ট উত্তর
        if "দুঃখিত" in response or "সমস্যা" in response:
            if 'eth' in question.lower() or 'ethereum' in question.lower():
                response = f"Ethereum (ETH) এর বর্তমান দাম ${eth_price}। RSI {eth_rsi} যা {eth_trend} ট্রেন্ড নির্দেশ করে। আর কি জানতে চান?"
            else:
                response = f"হাই! 👋 আমি তোমার ট্রেডিং অ্যাসিস্ট্যান্ট।\n\n📊 BTC: ${btc_price} | RSI: {btc_rsi}\n📊 ETH: ${eth_price} | RSI: {eth_rsi}\n\nকোন কয়েন নিয়ে জানতে চাও?"
        
        # কথোপকথন সেভ করো
        conversation_history[user_id].append({
            'q': question,
            'a': response,
            'time': datetime.datetime.now().strftime('%H:%M'),
            'coin': detect_coin_from_question(question)
        })
        
        final_response = f"""
🤖 **এআই অ্যাসিস্ট্যান্ট**

{response}

📊 **রিয়েল-টাইম ডাটা:**
BTC: ${btc_price} | RSI: {btc_rsi}
ETH: ${eth_price} | RSI: {eth_rsi}
"""
        
        await ctx.send(final_response[:1900])
        
    except Exception as e:
        await ctx.send(f"❌ এরর: {str(e)[:100]}\n\nআবার চেষ্টা করুন।")

@bot.command()
async def remember(ctx, coin: str):
    """আমাকে তোমার প্রিয় কয়েন বলে রাখো"""
    user_id = str(ctx.author.id)
    
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
    
    user_preferences[user_id]['fav_coin'] = coin.upper()
    
    await ctx.send(f"✅ মনে রাখলাম! আপনার প্রিয় কয়েন {coin.upper()} 💖")

@bot.command()
async def memory(ctx):
    """তোমার সাথে আমার কথোপকথন দেখো"""
    user_id = str(ctx.author.id)
    
    if user_id not in conversation_history or not conversation_history[user_id]:
        await ctx.send("📝 আপনার সাথে এখনো কোনো কথোপকথন হয়নি!")
        return
    
    history = conversation_history[user_id][-10:]
    
    msg = "📜 **আপনার কথোপকথন ইতিহাস:**\n"
    msg += "┌─────────────────────────\n"
    
    for h in history:
        emoji = "💬"
        if 'buy' in h['a'].lower():
            emoji = "🟢"
        elif 'sell' in h['a'].lower():
            emoji = "🔴"
        msg += f"│ {emoji} {h['time']} - {h['q'][:30]}...\n"
    
    msg += "└─────────────────────────"
    
    await ctx.send(msg)

@bot.command()
async def forget(ctx):
    """আমার স্মৃতি মুছে ফেলো"""
    user_id = str(ctx.author.id)
    
    if user_id in conversation_history:
        del conversation_history[user_id]
    
    if user_id in user_preferences:
        del user_preferences[user_id]
    
    await ctx.send("🧹 আপনার সব কথোপকথন এবং পছন্দ মুছে ফেলা হয়েছে!")

# ================ 📅 অটো ব্যাকটেস্ট সিডিউলার ================ #

async def daily_backtest():
    """প্রতি ২৪ ঘন্টায় অটো ব্যাকটেস্ট করে"""
    await bot.wait_until_ready()
    channel = discord.utils.get(bot.get_all_channels(), name="signals")
    
    while not bot.is_closed():
        now = datetime.datetime.now()
        
        if now.hour == 0 and now.minute == 0:
            if channel:
                await channel.send("🔄 **ডেইলি ব্যাকটেস্ট শুরু হচ্ছে...**")
            
            strategy_manager.update_strategy_scores("bitcoin")
            
            if channel:
                best = strategy_manager.performance.get("best_strategy")
                if best:
                    await channel.send(f"✅ **টুডেজ বেস্ট স্ট্রাটেজি:** {strategy_manager.strategies[best]['name']}")
        
        await asyncio.sleep(60)
# main.py-র একদম শেষে এই কোড যোগ করো (bot.run(TOKEN) এর আগে)

from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
import time

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")
    
    def log_message(self, format, *args):
        # লগ এড়িয়ে যাও
        pass

def run_http_server():
    try:
        server = HTTPServer(('0.0.0.0', 10000), HealthCheckHandler)
        print("✅ হেলথ চেক সার্ভার চালু হয়েছে পোর্ট 10000-এ")
        server.serve_forever()
    except Exception as e:
        print(f"❌ হেলথ চেক সার্ভার এরর: {e}")

# HTTP সার্ভার আলাদা থ্রেডে চালাও
Thread(target=run_http_server, daemon=True).start()
print("✅ বট চালু হচ্ছে...")

# -------- বট চালু করা -------- #
bot.run(TOKEN)