# tensorflow-exercise

# 1. mnist_test.py 

程序列举了常用分类评价指标的计算方法

1. 混淆矩阵（Confusion Matrix，CM）
2. 每类分类精度
3. 平均分类精度（Average Accuracy，OA)
4. 总体分类精度（Overall Accuracy，OA）

# 2. svm_grid_search.py

程序使用SVM算法针对iris数据集做分类实验，使用GridSearchCV模块自动搜索超参数

1. 自动搜索超参数的GridSearchCV模块的基本用法
2. 获取不同超参数组合对应的模型性能，使用pandas.DataFrame模块快速保存至CSV文件。

# 3. evaluate.py

程序列举了高光谱图像重构评价指标及其Python实现

1. MSE
2. 从PSNR到MPSNR
3. 从SSIM到MSSIM
