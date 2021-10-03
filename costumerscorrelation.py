from efficient_apriori import apriori

# 设置数据集
data = []
'''
data = [['牛奶','面包','尿布'],
            ['可乐','面包', '尿布', '啤酒'],
            ['牛奶','尿布', '啤酒', '鸡蛋'],
            ['面包', '牛奶', '尿布', '啤酒'],
            ['面包', '牛奶', '尿布', '可乐']]

'''

with open(r'/Users/qinanyu/Desktop/test.txt', 'r', encoding='utf-8') as file_read:
    while True:
        lines = file_read.readline()
        # lines=20
        if not lines:
            break
        line_tuple = tuple(lines.strip('\n').split(','))

        data.append(line_tuple)
    print(data[0:4])

# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(data, min_support=0.1, min_confidence=0.2)
# print(data[0],data[1],data[2])
# print(itemsets)
new_rules = []
print(len(rules))
for i in rules:
    new_rules.append((i, i.confidence))

print(new_rules)