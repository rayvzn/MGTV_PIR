## 第三届“马栏山杯”芒果tv算法大赛邀请赛商品意图识别赛道baseline
本代码运行方式：
1.将训练集按比例划分train.tsv和dev.tsv，放入dataset目录下
2.执行python train.py完成模型训练
3.将需要预测的数据集放入dataset，例如dev.csv，然后运行python predict.py，即可得到预测结果

本baseline在test_a数据集上，macro f1约为0.73

温馨提示：
1.在比赛官网提交结果时，顶上将有进度条，且提交成功后会有提示。接收到"提交成功"的提示前不要关掉页面哦。
2.评分失败时请检查是否按照正确提交格式提交，正确的提交应该是一个csv文件，里面包含id和label两列
