import sqlite3
import random
from datetime import datetime, timedelta

# 创建数据库连接（如果文件不存在，会自动创建）
conn = sqlite3.connect('water_quality.db')
cursor = conn.cursor()

# 1. 创建数据表
cursor.execute('''
CREATE TABLE IF NOT EXISTS monitoring_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_name TEXT NOT NULL,
    record_time TEXT NOT NULL,
    ph_value REAL,
    cod REAL,
    ammonia_nitrogen REAL,
    dissolved_oxygen REAL,
    water_temperature REAL
)
''')

# 2. 模拟数据生成函数
def generate_sample_data():
    stations = ['松原监测站', '扶余监测站', '前郭监测站']
    start_date = datetime(2026, 6, 1)
    end_date = datetime(2026, 9, 1)
    delta = end_date - start_date
    records = []
    
    for day in range(delta.days + 1):
        current_date = start_date + timedelta(days=day)
        for station in stations:
            # 随机生成符合真实水环境特征的数据
            records.append((
                station,
                current_date.strftime('%Y-%m-%d'),
                round(random.uniform(6.5, 8.5), 2),          # pH值 6.5-8.5
                round(random.uniform(10, 40), 2),           # COD 10-40 mg/L
                round(random.uniform(0.2, 2.0), 2),         # 氨氮 0.2-2.0 mg/L
                round(random.uniform(5.0, 9.0), 2),         # 溶解氧 5.0-9.0 mg/L
                round(random.uniform(10.0, 28.0), 1)        # 水温 10.0-28.0℃
            ))
    return records

# 3. 插入数据
data = generate_sample_data()
cursor.executemany('''
INSERT INTO monitoring_data 
(station_name, record_time, ph_value, cod, ammonia_nitrogen, dissolved_oxygen, water_temperature)
VALUES (?, ?, ?, ?, ?, ?, ?)
''', data)

# 4. 提交事务并关闭连接
conn.commit()
conn.close()

print(f"✅ 数据库初始化完成！共插入 {len(data)} 条模拟监测数据。")
print("📁 数据库文件已保存为: water_quality.db")