# 文件路径
file1_path = "D:\\cppProject\\hnswlib-master\\tests\\new\sift_label_filter.out"
file2_path = "D:\\cppProject\\hnswlib-master\\tests\\new\sift_gt.out"  # 另一个文件，用于比较

# 初始化变量
total_recall = 0
total_elements = 0

# 定义行的起始和结束范围 (例如: n=1 到 m=10000)
start_line = 1  # 起始行号 (包含)
end_line = 10000  # 结束行号 (包含)

# 定义每行取前多少个数据进行比较
num_elements_per_line = 10

# 逐行读取两个文件并比较指定范围内的行
with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
    for line_num, (line1, line2) in enumerate(zip(file1, file2), start=1):
        if line_num < start_line:
            continue  # 跳过指定范围之前的行
        if line_num > end_line:
            break  # 退出循环在指定范围之后

        # 将每行的数值解析为列表，并取前 num_elements_per_line 个数据
        list1 = list(map(int, line1.strip().split()))[:num_elements_per_line]
        list2 = list(map(int, line2.strip().split()))[:num_elements_per_line]
        
        # 将列表转换为集合
        set1 = set(list1)
        set2 = set(list2)
        
        # 计算相同的元素数量和基准集合的元素总数
        common_elements = set1.intersection(set2)
        total_recall += len(common_elements)
        total_elements += num_elements_per_line  # 每行固定取 num_elements_per_line 个数据

# 计算并输出总召回率
recall_rate = total_recall / total_elements if total_elements > 0 else 0
print(f"Total Recall Rate from line {start_line} to {end_line}: {recall_rate:.4f}")