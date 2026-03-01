# database.py
import sqlite3
import datetime
import json

class DatabaseManager:
    def __init__(self, db_name="trading_bot.db"):
        self.db_name = db_name
        self.create_tables()
    
    def create_tables(self):
        """সব টেবিল তৈরি করে"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # সিগন্যাল টেবিল
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                coin TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence INTEGER,
                price REAL,
                rsi REAL,
                macd REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ট্রেড টেবিল
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                coin TEXT NOT NULL,
                type TEXT,
                entry_price REAL,
                exit_price REAL,
                amount REAL,
                pnl REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # পারফরম্যান্স টেবিল
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                win_rate REAL,
                total_trades INTEGER,
                date DATE
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ ডাটাবেস তৈরি হয়েছে!")
    
    def save_signal(self, coin, signal, confidence, price, rsi, macd):
        """সিগন্যাল ডাটাবেসে সেভ করে"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (coin, signal, confidence, price, rsi, macd)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (coin, signal, confidence, price, rsi, macd))
        
        conn.commit()
        conn.close()
        print(f"✅ সিগন্যাল সেভ হয়েছে: {coin} - {signal}")
    
    def get_recent_signals(self, coin, limit=10):
        """সাম্প্রতিক সিগন্যাল দেখায়"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT signal, confidence, price, timestamp 
            FROM signals 
            WHERE coin = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (coin, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_statistics(self):
        """ডাটাবেস পরিসংখ্যান"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        stats = {}
        
        # মোট সিগন্যাল
        cursor.execute("SELECT COUNT(*) FROM signals")
        stats['total_signals'] = cursor.fetchone()[0]
        
        # BUY/SELL/HOLD কাউন্ট
        cursor.execute("SELECT signal, COUNT(*) FROM signals GROUP BY signal")
        stats['signal_counts'] = dict(cursor.fetchall())
        
        # এভারেজ কনফিডেন্স
        cursor.execute("SELECT AVG(confidence) FROM signals")
        stats['avg_confidence'] = round(cursor.fetchone()[0] or 0, 2)
        
        conn.close()
        return stats

# ডাটাবেস ইনিশিয়ালাইজ
db = DatabaseManager()