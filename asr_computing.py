import pandas as pd

# 读取CSV文件
df = pd.read_csv('first_ord_stable_final_result.csv')

# 计算SAR
total_rows = len(df)  # 总行数
success_rows = len(df[df['is_succ'] == True])  # 成功的行数

ASR = success_rows / total_rows
print(f'Average Successful Rate (ASR): {ASR}')
