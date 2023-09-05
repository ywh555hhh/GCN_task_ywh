from datetime import datetime
import subprocess

# 设置循环次数
iterations = 500
total_score = 0

# 打开或创建txt文件来保存实验数据
output_file = "experiment_results.txt"
file_mode = "a"  # 使用追加模式打开文件

# 循环
for i in range(iterations):
    # 写入时间戳和程序名称，只在第一次运行时执行
    if i == 0:
        with open(output_file, file_mode) as file:
            file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Infomation: try to use complier opt  \n\n")

    # 运行程序并捕获输出
    result = subprocess.getoutput('./run.sh')

    # 按行分割结果
    lines = result.split('\n')

    # 获取程序的结果和得分
    program_result = float(lines[0])
    program_score = float(lines[1])

    # 检查程序的结果是否正确
    if program_result == -16.68968964:
        # 累加得分
        total_score += program_score
    else:
        # 如果结果错误，报错并跳出循环
        print("程序结果错误")
        break

# 计算平均得分
average_score = total_score / iterations

# 打开文件并写入平均得分
with open(output_file, file_mode) as file:
    # 写入平均得分
    file.write(f"Average Score: {average_score:.2f}\n\n")

# 打印保存数据的文件名
print(f"实验数据已保存到文件: {output_file}")

