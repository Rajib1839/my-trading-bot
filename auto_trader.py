# auto_trader.py
import asyncio
import datetime
import discord
from discord.ext import commands

class AutoTradingBot:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.is_running = False
        self.portfolio = {'balance': 10000}
        self.trade_history = []
        self.risk_percent = 2
        self.max_concurrent_trades = 5
        
    async def start(self):
        """অটো ট্রেডিং শুরু করো"""
        self.is_running = True
        print("🤖 অটো ট্রেডিং বট চালু হয়েছে!")
        
        while self.is_running:
            try:
                now = datetime.datetime.now()
                
                # প্রতি মিনিটে স্ট্যাটাস দেখাও
                if now.second == 0:
                    print(f"🕐 অটো ট্রেডিং চালু আছে... {now.strftime('%H:%M:%S')}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in auto trading: {e}")
                await asyncio.sleep(5)
    
    def stop(self):
        """অটো ট্রেডিং বন্ধ করো"""
        self.is_running = False
        print("🛑 অটো ট্রেডিং বন্ধ হয়েছে")
    
    def count_open_trades(self):
        """ওপেন ট্রেডের সংখ্যা দেখাও"""
        return len([t for t in self.trade_history if t.get('status') == 'open'])
    
    def calculate_total_pnl(self):
        """মোট প্রফিট/লস ক্যালকুলেট করো"""
        total = 0
        for trade in self.trade_history:
            if trade.get('pnl'):
                total += trade['pnl']
        return total
    
    def next_scan_time(self):
        """পরবর্তী স্ক্যান টাইম দেখাও"""
        next_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
        return f"পরবর্তী স্ক্যান: {next_time.strftime('%H:%M:%S')}"