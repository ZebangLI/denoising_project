项目进展总结
============

1. 总体进展
-----------

+----+--------------------------+--------------------------------------------------------------------------+
| 序号 | 项目内容                 | 总结说明                                                                 |
+----+--------------------------+--------------------------------------------------------------------------+
| 1  | Synthetic 数据集准备     | 已下载 BSD500，并使用 train 集合作为 clean image 数据。                  |
| 2  | Synthetic 噪声生成       | 已实现 Gaussian noise，将 clean image 转换为 noisy image。               |
| 3  | Patch 裁剪               | 已对图像随机裁剪为 128x128 的固定大小 patch。                            |
| 4  | Synthetic 数据集流程     | 已构建返回 (noisy_patch, clean_patch) 的 dataset。                       |
| 5  | Synthetic DataLoader     | 已成功构建 batch，输出尺寸为 [4, 3, 128, 128]。                         |
| 6  | Baseline 模型            | 已实现一个简单的 CNN 去噪模型 Simple CNN denoiser。                      |
| 7  | Synthetic 模型训练       | 已完成 3 个 epoch 的训练，并且 loss 明显下降。                           |
| 8  | Synthetic 推理可视化     | 已显示 clean / noisy / denoised 三种图像对比结果。                      |
| 9  | Synthetic 定量评价       | 已计算单张测试 patch 的 PSNR，并完成 BSD500 val 平均 PSNR 评价。         |
| 10 | Synthetic 结果保存       | 已保存 5 组 BSD500 validation 的 clean / noisy / denoised 对比图。       |
| 11 | SIDD 数据集准备          | 已下载并读取 SIDD-Small sRGB 数据集，共 160 对 noisy / ground truth 图像。|
| 12 | SIDD 数据集流程          | 已实现 SIDDDataset，可返回 (noisy_patch, clean_patch)。                  |
| 13 | SIDD DataLoader          | 已成功构建 batch，输出尺寸为 [4, 3, 128, 128]。                         |
| 14 | GPU 环境配置             | 已将 PyTorch 切换为 CUDA 版本，成功识别 NVIDIA GeForce RTX 3060 Ti。     |
| 15 | SIDD 模型训练            | 已使用同一个 Simple CNN 在 SIDD 上完成 3 个 epoch 的 real-noise 训练。   |
| 16 | SIDD 推理可视化          | 已完成 SIDD patch 的 Ground Truth / Noisy / Denoised 推理显示。          |
| 17 | SIDD 定量评价            | 已计算单张 SIDD patch 的 PSNR。                                          |
| 18 | SIDD 结果保存            | 已保存 10 组 SIDD 的 Ground Truth / Noisy / Denoised 对比图。            |
+----+--------------------------+--------------------------------------------------------------------------+


2. Synthetic 训练结果（BSD500 + Gaussian Noise）
------------------------------------------------

+---------+-----------+
| Epoch   | Loss      |
+---------+-----------+
| 1       | 0.020709  |
| 2       | 0.003988  |
| 3       | 0.002266  |
+---------+-----------+


3. Synthetic PSNR 结果
----------------------

3.1 单张测试 patch 结果
+----------------------------+-----------+
| 指标                       | 数值      |
+----------------------------+-----------+
| Noisy PSNR                 | 20.31 dB  |
| Denoised PSNR              | 28.77 dB  |
+----------------------------+-----------+

3.2 BSD500 validation 平均结果
+----------------------------+-----------+
| 指标                       | 数值      |
+----------------------------+-----------+
| Validation sample 数量     | 100       |
| Average Noisy PSNR         | 20.51 dB  |
| Average Denoised PSNR      | 27.38 dB  |
+----------------------------+-----------+


4. SIDD 训练结果（Real-Noise Baseline）
--------------------------------------

+---------+-----------+
| Epoch   | Loss      |
+---------+-----------+
| 1       | 0.017450  |
| 2       | 0.003797  |
| 3       | 0.002140  |
+---------+-----------+


5. SIDD PSNR 结果
-----------------

5.1 单张 Patch PSNR 结果
+----------------------------+-----------+
| 指标                       | 数值      |
+----------------------------+-----------+
| Noisy PSNR                 | 29.23 dB  |
| Denoised PSNR              | 42.01 dB  |
+----------------------------+-----------+

5.2 SIDD 平均 PSNR 结果
+----------------------------+-----------+
| 指标                       | 数值      |
+----------------------------+-----------+
| SIDD sample 数量           | 160       |
| Average Noisy PSNR         | 27.93 dB  |
| Average Denoised PSNR      | 30.14 dB  |
+----------------------------+-----------+

说明：
单张 patch 的结果可能较高，因为随机裁剪到的局部区域可能较为平坦或较暗。
因此更正式、更有代表性的结果应以整个 SIDD 数据集上的平均 PSNR 为主。


6. 文件功能总结
---------------

+-------------------------------+------------------------------------------------------------------+
| 文件名                        | 功能说明                                                         |
+-------------------------------+------------------------------------------------------------------+
| datasets/synthetic_dataset.py | 读取 BSD500 图像，裁剪 patch，加噪声，并返回训练数据对。         |
| datasets/sidd_dataset.py      | 读取 SIDD noisy / ground-truth 图像，裁剪 patch，并返回数据对。  |
| utils/noise.py                | 定义 Gaussian noise 生成函数。                                   |
| utils/metrics.py              | 定义 PSNR 计算函数。                                             |
| models/simple_cnn.py          | 定义 baseline CNN 去噪模型。                                     |
| test_dataset.py               | 测试 BSD500 图像读取、patch 裁剪及 noisy/clean 图像对。          |
| test_dataloader.py            | 测试 synthetic DataLoader 的 batch 输出尺寸。                    |
| test_model.py                 | 测试模型输入输出的 shape 是否正确。                              |
| train.py                      | 训练 synthetic baseline 模型，并保存为 simple_denoiser.pth。     |
| test_inference.py             | 对 synthetic baseline 做推理可视化并计算单张 patch 的 PSNR。     |
| evaluate.py                   | 在 BSD500 validation 集上计算平均 Noisy/Denoised PSNR。          |
| save_val_results.py           | 保存 BSD500 validation 的结果图。                                |
| test_sidd_pair.py             | 测试读取单对 SIDD 的 noisy / ground-truth 图像。                 |
| test_sidd_dataset.py          | 测试 SIDD dataset 是否能正确返回 128x128 patch。                 |
| test_sidd_dataloader.py       | 测试 SIDD DataLoader 的 batch 输出尺寸。                         |
| train_sidd.py                 | 使用 SIDD 数据训练 real-noise baseline，并保存为                 |
|                               | simple_denoiser_sidd.pth。                                       |
| test_sidd_inference.py        | 对 SIDD 模型做单张 patch 推理可视化并计算 PSNR。                 |
| save_sidd_results.py          | 保存 SIDD 的结果图。                                             |
| check_cuda.py                 | 检查当前 PyTorch 是否识别 CUDA 和 RTX 3060 Ti。                  |
+-------------------------------+------------------------------------------------------------------+


7. 当前实验流程
---------------

7.1 Synthetic baseline
+-------------------+--------------------------------------------------+
| 步骤              | 说明                                             |
+-------------------+--------------------------------------------------+
| 输入数据          | BSD500 clean images                              |
| 数据处理          | 随机裁剪为 128x128 patch                         |
| 噪声模型          | Gaussian noise，sigma = 25                       |
| 训练数据对        | (noisy_patch, clean_patch)                       |
| 模型              | Simple CNN denoiser                              |
| 损失函数          | MSELoss                                          |
| 优化器            | Adam                                             |
| 输出              | 去噪后的 patch                                   |
| 评价方式          | 图像可视化 + 单张 PSNR + 平均 PSNR               |
+-------------------+--------------------------------------------------+

7.2 Real-noise baseline
+-------------------+--------------------------------------------------+
| 步骤              | 说明                                             |
+-------------------+--------------------------------------------------+
| 输入数据          | SIDD-Small sRGB noisy / ground-truth pairs       |
| 数据处理          | 从 noisy / GT 同一位置随机裁剪 128x128 patch     |
| 训练数据对        | (noisy_patch, clean_patch)                       |
| 模型              | Simple CNN denoiser                              |
| 损失函数          | MSELoss                                          |
| 优化器            | Adam                                             |
| 训练设备          | CUDA (NVIDIA GeForce RTX 3060 Ti)                |
| 输出              | 去噪后的 patch                                   |
| 评价方式          | 图像可视化 + 单张 PSNR                           |
+-------------------+--------------------------------------------------+


8. 当前可展示结果
----------------

+------------------------------+--------------------------------------------------------------+
| 项目                         | 当前结果                                                     |
+------------------------------+--------------------------------------------------------------+
| Synthetic train loss         | 已完成，loss 明显下降                                         |
| Synthetic validation PSNR    | 已完成，Average Noisy = 20.51 dB，Average Denoised = 27.38 dB|
| Synthetic 可视化结果         | 已保存 5 组 BSD500 validation 图像对比结果                   |
| SIDD train loss              | 已完成，loss 明显下降                                         |
| SIDD 平均 PSNR              | 已完成，Average Noisy = 27.93 dB，Average Denoised = 30.14 dB|
| SIDD 可视化结果              | 已保存 10 组 SIDD 图像对比结果                               |
+------------------------------+--------------------------------------------------------------+

9. 简短总结
-----------

目前已经完成了两条 baseline 实验线：

1. Synthetic noise baseline：
   使用 BSD500 作为 clean image 数据集，通过加入 Gaussian noise 构造 noisy input，
   并使用随机裁剪得到 128x128 patch 作为训练样本。随后实现了一个简单的 CNN 去噪模型，
   完成了训练、推理可视化、单张 patch PSNR 评价以及 BSD500 validation 平均 PSNR 评价。
   结果表明模型能够有效降低 synthetic noise，并将 validation 平均 PSNR 从
   20.51 dB 提升到 27.38 dB。

2. Real-noise baseline：
   使用 SIDD-Small sRGB 真实噪声图像对，构建了 SIDDDataset，并在 RTX 3060 Ti 上
   使用同一个 Simple CNN 完成 real-noise 训练。模型训练 loss 明显下降，并在
   整个 SIDD 数据集上将 Average PSNR 从 27.93 dB 提升到 30.14 dB，同时已保存
   多组 SIDD 的 Ground Truth / Noisy / Denoised 结果图用于后续展示。

10. 下一步计划
--------------

+----+-----------------------------------+--------------------------------------------------------------+
| 序号 | 下一步任务                        | 说明                                                         |
+----+-----------------------------------+--------------------------------------------------------------+
| 2  | 整理对比图                         | 从 BSD500 与 SIDD 结果图中挑选最适合展示的样例。             |
| 3  | 整理 comparison 表格              | 比较 synthetic 与 real-noise baseline 的 loss / PSNR / 图像。|
| 4  | 组织 PPT / 报告                   | 将实验流程、表格和可视化结果整理为 presentation materials。 |
+----+-----------------------------------+--------------------------------------------------------------+

11. Synthetic vs Real-Noise Comparison
--------------------------------------

+------------------------------+--------------------------------------+--------------------------------------+
| 项目                         | Synthetic Baseline                   | Real-Noise Baseline                  |
+------------------------------+--------------------------------------+--------------------------------------+
| 数据集                       | BSD500                               | SIDD-Small sRGB                      |
| 噪声来源                     | 人工添加 Gaussian noise              | 真实手机噪声                         |
| 训练数据形式                 | clean image + synthetic noise        | noisy / ground-truth image pairs     |
| patch size                   | 128 x 128                            | 128 x 128                            |
| batch size                   | 4                                    | 4                                    |
| epoch 数                     | 3                                    | 3                                    |
| 模型                         | Simple CNN denoiser                  | Simple CNN denoiser                  |
| 损失函数                     | MSELoss                              | MSELoss                              |
| 优化器                       | Adam                                 | Adam                                 |
| 训练设备                     | CPU                                  | CUDA (RTX 3060 Ti)                   |
| 平均 Noisy PSNR              | 20.51 dB                             | 27.93 dB                             |
| 平均 Denoised PSNR           | 27.38 dB                             | 30.14 dB                             |
| 平均 PSNR 提升               | 6.87 dB                              | 2.21 dB                              |
| 可视化结果                   | 已保存 5 组 validation 结果图        | 已保存 10 组 SIDD 结果图             |
+------------------------------+--------------------------------------+--------------------------------------+


12. Comparison Summary
----------------------

+----+-----------------------------------+------------------------------------------------------------------+
| 序号 | 对比点                            | 说明                                                             |
+----+-----------------------------------+------------------------------------------------------------------+
| 1  | 噪声类型不同                      | Synthetic 使用规则的 Gaussian noise，SIDD 使用真实复杂噪声。     |
| 2  | 数据生成方式不同                  | Synthetic 由 clean image 人工加噪得到，SIDD 直接提供 noisy/GT 对。|
| 3  | PSNR 提升幅度不同                 | Synthetic 提升更大，说明规则噪声更容易被模型学习和去除。          |
| 4  | 真实噪声更难处理                  | SIDD 的噪声更复杂，所以虽然有提升，但提升幅度小于 synthetic。     |
| 5  | 两种训练方式都有效                | 两个 baseline 都成功降低了噪声，并提高了图像质量。               |
+----+-----------------------------------+------------------------------------------------------------------+


13. 当前阶段结论
----------------

从当前实验结果来看，Simple CNN 在两种训练设置下都能够学习图像去噪任务。
对于 synthetic Gaussian noise，模型在 BSD500 validation 集上的平均 PSNR
从 20.51 dB 提升到 27.38 dB，提升了 6.87 dB；对于真实噪声数据 SIDD，
模型的平均 PSNR 从 27.93 dB 提升到 30.14 dB，提升了 2.21 dB。

这表明：
1. synthetic noise training 更容易获得较大的数值提升；
2. real-noise training 也有效，但真实噪声更复杂，去噪难度更高；
3. 当前实验结果支持项目主题：不同训练数据会显著影响 denoising model 的表现。
-----------------------------------------------------------------------------------------------------------------------
Improved Pipeline v2

                    Training Data
                         |
          +--------------+---------------+
          |                              |
   Synthetic Gaussian Noise         Real Noise (SIDD)
          |                              |
      Train Model A                  Train Model B
          |                              |
          +--------------+---------------+
                         |
                  Cross Evaluation
                         |
       +----------------+----------------+
       |                                 |
 Test on Synthetic Set             Test on Real SIDD Set
       |                                 |
 PSNR / SSIM / Visual             PSNR / SSIM / Visual
       +----------------+----------------+
                         |
                 Compare & Analyze
                         |
  Generalization / Robustness / Noise Distribution Effect


  Project Improvement List
  ========================

  A. 必做 / Must Do
  -----------------
  1. 统一训练设置 / Unify training settings
     - 两边使用相同的 epoch 数 / Use the same number of epochs
     - 两边使用相同的 learning rate / Use the same learning rate
     - 两边使用相同的 batch size / Use the same batch size
     - 两边使用相同的 patch size / Use the same patch size
     - 两边尽量使用相同的 device / Use the same device if possible

  2. 增加训练轮数 / Increase training epochs
     - 从 3 epochs 提高到 20-50 epochs / Increase from 3 epochs to 20-50 epochs
     - 保存每个 epoch 的 loss 和 PSNR / Save loss and PSNR for each epoch

  3. 加入 SSIM / Add SSIM
     - 不要只用 PSNR 评价 / Do not evaluate only with PSNR
     - 使用 PSNR + SSIM 一起评价 / Use both PSNR and SSIM

  4. 保存更多可视化结果 / Save more visualization results
     - 对比 noisy / denoised / clean 图像 / Compare noisy / denoised / clean images
     - 多保存一些代表性结果 / Save more representative examples

  B. 最重要的提升 / Most Important Improvements
  --------------------------------------------
  5. 做 cross-dataset evaluation / Perform cross-dataset evaluation
     - synthetic-trained model 测 synthetic data / Test synthetic-trained model on synthetic data
     - synthetic-trained model 测 SIDD data / Test synthetic-trained model on SIDD data
     - real-noise-trained model 测 synthetic data / Test real-noise-trained model on synthetic data
     - real-noise-trained model 测 SIDD data / Test real-noise-trained model on SIDD data

  6. 画训练曲线 / Plot training curves
     - training loss curve / 训练损失曲线
     - validation PSNR curve / 验证集 PSNR 曲线
     - validation SSIM curve / 验证集 SSIM 曲线

  7. 讨论比较的公平性 / Discuss fairness of comparison
     - 说明两个实验使用的数据集不同 / Explain that the datasets are different
     - 说明 noisy baseline 不同 / Explain that the noisy baselines are different
     - 说明当前结论属于现象性比较，不是完全公平比较 /
       State that the current conclusion is an empirical comparison,
       not a fully controlled comparison

  C. 让项目更完整 / Make the Project More Complete
  -----------------------------------------------
  8. 加局部放大图 / Add local zoom-in comparisons
     - 对比边缘、纹理和细节恢复 / Compare edges, textures, and fine details

  9. 加 difference map / Add difference maps
     - 显示 |denoised - ground truth|
     - 可视化剩余误差 / Visualize remaining reconstruction error

  10. 增加结果表 / Add more result tables
     - noisy PSNR
     - denoised PSNR
     - PSNR improvement
     - SSIM

  D. 让项目更有深度 / Make the Project Deeper
  ------------------------------------------
  11. 加入一个更强的模型 / Add a stronger model
     - 推荐 DnCNN / Recommended: DnCNN
     - 或者 small U-Net / Or a small U-Net

  12. 做 noise-level ablation / Perform noise-level ablation
     - sigma = 15
     - sigma = 25
     - sigma = 50

  13. 比较不同噪声强度下的表现 / Compare performance under different noise levels
     - 分析噪声越大是否更难恢复 / Check whether larger noise is harder to remove
     - 分析哪种训练方式更稳定 / Analyze which training strategy is more stable

  E. 可作为扩展 / Possible Extensions
  ----------------------------------
  14. 尝试 mixed training / Try mixed training
     - synthetic only
     - real only
     - synthetic pretrain + real fine-tune

  15. 分析泛化能力 / Analyze generalization
     - synthetic training 是否只对 synthetic noise 效果好 /
       Whether synthetic training only works well on synthetic noise
     - real-noise training 是否更适合真实图像 /
       Whether real-noise training is better for real images

  16. 清楚写出 limitation / Write the limitations clearly
     - 数据集不同 / Different datasets
     - 噪声分布不同 / Different noise distributions
     - 模型较简单 / Simple model
     - 训练轮数有限 / Limited training epochs

  Top 5 Recommended Next Steps / 最推荐的 5 个下一步
  -------------------------------------------------
  1. 统一训练设置 / Unify training settings
  2. 训练到 20-30 epochs / Train for 20-30 epochs
  3. 加入 SSIM / Add SSIM
  4. 做 cross-dataset evaluation / Perform cross-dataset evaluation
  5. 加一个 DnCNN baseline / Add a DnCNN baseline

  ---------2026/3/25----------------------------------------------------------------