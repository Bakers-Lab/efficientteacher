import matplotlib.pyplot as plt
import numpy as np



# 假设你有三列数据
data1 = [17, 1203, 17, 6, 13, 41, 9, 1,
         0, 0, 0, 0, 0, 0, 0, 0]
data2 = [219, 0, 16, 1, 25, 2, 0, 3, 
         219, 0, 655, 0, 150, 0, 0, 0]
data3 = [17, 1203, 17, 6, 13, 41, 9, 1,
         0, 0, 0, 0, 150, 0, 0, 0]

# 类别标签
categories = ['CM', 'PASS', 'SB', 'FM', 'IS', 'PS', 'pass', 'IC',
              'PHONE\nCM', 'PHONE\nPASS', 'PHONE\nSB', 'PHONE\nFM', 'PHONE\nIS', 'PHONE\nPS', 'PHONE\npass', 'PHONE\nIC',]

# 设置柱状图的宽度
bar_width = 0.5

# 生成x轴的位置
x = np.arange(0, len(categories) * 2, 2)

# fig, ax = plt.subplots()

# 绘制横向分组柱状图
rects1 = plt.bar(x - bar_width, data1, bar_width, label='BASE')
rects2 = plt.bar(x, data2, bar_width, label='+PHONE')
rects3 = plt.bar(x + bar_width, data3, bar_width, label='+IS')

# 设置x轴标签和标题

custom_xticks = x
custom_xticklabels = categories

# plt.xticks(custom_xticks, custom_xticklabels, fontsize=10)

for rect in rects1 + rects2 + rects3:
    height = rect.get_height()
    plt.annotate(f'{height}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=4)



plt.xticks(custom_xticks, custom_xticklabels, fontsize=4)
# ax.set_xticklabels(categories)

# 添加图例
plt.legend()

# 显示图形

plt.savefig('bar_chart.png',dpi=600)
