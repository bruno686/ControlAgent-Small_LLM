import pandas as pd

# 读取文件
df = pd.read_csv('first_order_stable_fast_data_final_result.csv', header=None)

# 计算第一列为 True 的比例
ASR = (df[0] == True).sum() / len(df)

print(f'Average Successful Rate (ASR): {ASR}')
