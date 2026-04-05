import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. ข้อมูลตัวอย่าง (สมมติ: ขนาด และ น้ำหนัก ของอุปกรณ์)
# [ความยาว, น้ำหนัก] -> ประเภท (0: สกรู, 1: ไขควง)
data = [[1, 5], [1.5, 7], [10, 50], [12, 60]]
labels = [0, 0, 1, 1] 

# 2. สร้างโมเดล AI
model = RandomForestClassifier()
model.fit(data, labels)

# 3. บันทึกโมเดลไว้ใช้งาน
joblib.dump(model, 'hardware_model.pkl')
print("สร้างโมเดล AI สำเร็จแล้ว!")
