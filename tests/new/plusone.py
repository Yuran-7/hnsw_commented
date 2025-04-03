# 读取文件并处理数据
input_file = "./tests/cpp/sift_knn.out"
output_file = "./tests/cpp/sift_knn.out"

# 打开文件，逐行读取并处理
with open(input_file, 'r') as infile:
    lines = infile.readlines()

# 将每个数值加 1，然后写回文件
with open(output_file, 'w') as outfile:
    for line in lines:
        # 将每行的值转换为整数，+1 后再转回字符串
        updated_line = " ".join(str(int(value) + 1) for value in line.split())
        outfile.write(updated_line + "\n")

print("所有值已成功加 1 并保存到 sift_iterator.out 文件中。")
