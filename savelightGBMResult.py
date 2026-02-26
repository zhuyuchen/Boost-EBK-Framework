import pandas as pd
import lightgbm as lgb
import shap
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t
from eCharts_test import get_RadialColumnDiagram, sort_importances_with_features
from scipy.interpolate import CubicSpline

if __name__ == '__main__':
    pd.set_option('display.width', 600)
    pd.set_option('display.max_columns', 600)
    csv_path = "../file/csv/POI/"
    file_name = "SP1107.csv"
    file_path = csv_path + file_name
    # 读取包含经纬度信息的CSV文件
    data = pd.read_csv(file_path)
    # 选择CSV中所需要的变量，输入变量名即可
    # selected_columns = ['Pb', 'Cr', 'Hg', 'As_']
    # selected_columns = ['pH', 'Corg', 'DEM', 'RiverDista', 'RoadDistan', 'Slopetry', 'CropDistan', 'RainMax', 'RainAvg',
    #                     'LandCov', 'Geology', 'SoilType', 'Parent', 'KuangChanD', 'NDVI2016_30']
    # selected_columns = ['pH', 'DEM', 'RiverDist', 'RoadDist', 'Slopetry', 'CropDist', 'RainMax', 'RainAvg', 'LandCov',
    #                     'Geology', 'SoilType', 'Parent', 'MineDist', 'Corg', 'NDVI', 'PM25', 'PM10']
    selected_columns = ['pH', 'DEM', 'RiverDist', 'RoadDist', 'RainMax',
                        'Geology', 'SoilType', 'Parent', 'MineDist', 'Corg', 'PM10', 'NDVI']
    selected_columns_cat = ['Geology', 'SoilType', 'Parent']
    selected_y = 'Cd'
    selected_data_x = data[selected_columns]  # 不包括目标变量'As_'
    selected_data_y = data[selected_y]

    # 将分类特征的列名转换为列索引
    categorical_feature_indices = [selected_columns.index(col) for col in selected_columns_cat]
    # 正态分布剔除异常值
    y_mean = np.mean(selected_data_y)
    y_std = np.std(selected_data_y)
    # 定义异常值的界限
    lower_bound = y_mean - 3 * y_std
    upper_bound = y_mean + 3 * y_std


    def remove_Outliers(y):
        index = []
        # 同时遍历x和y，使用enumerate获取索引
        for i, value in enumerate(y):
            if value < lower_bound or value > upper_bound:
                index.append(i)
        return index


    normalData = data.drop(remove_Outliers(selected_data_y))
    x_filtered = normalData[selected_columns]
    y_filtered = normalData['Cd']
    print('y_filtered', len(y_filtered))
    print(y_filtered.describe())
    print("y_filtered std:", y_filtered.std())
    print("y_filtered min/max:", y_filtered.min(), y_filtered.max())
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(x_filtered, y_filtered, test_size=0.3, random_state=31)

    # 填充缺失值
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    # 是否设置 离散值特殊处理
    isSet = False
    if isSet:
        # 创建 LightGBM 数据集
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature_indices)
        # 创建 LightGBM 回归模型
        params = {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.02,
            'feature_fraction': 0.9,
            'n_estimators': 400,
            'verbose': 2
        }
        lgbm_reg = lgb.train(params, lgb_train)
        y_pred = lgbm_reg.predict(X_test)
        y_pred_train = lgbm_reg.predict(X_train)
    else:
        lgbm_reg = lgb.LGBMRegressor(objective='regression', num_leaves=6, learning_rate=0.02, n_estimators=200,
                                     verbosity=-1, random_state=31 ,lambda_l1=0.7,  # L1正则化
                                     lambda_l2=0.7,  # L2正则化
                                     feature_fraction=0.8, )  # 特征采样比例

        lgbm_reg.fit(X_train, y_train)
        # 训练模型后
        importance = lgbm_reg.feature_importances_
        # 打印特征重要性
        sorted_idx = np.argsort(importance)
        for i in sorted_idx:
            print(f"{selected_columns[i]}: {importance[i]:.4f}")
        # 进行预测
        y_pred = lgbm_reg.predict(X_test)
        y_pred_train = lgbm_reg.predict(X_train)
        isCharts = False
        if isCharts:
            # 使用函数对特征重要性进行排序，并与特征名称组成键值对
            sorted_importances_with_features = sort_importances_with_features(importance, selected_columns)
            get_RadialColumnDiagram(sorted_importances_with_features, 'LGBM', 18, '#EB7E60')
    # 评估模型
    x_values = np.arange(len(y_test))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_pred_train)
    rmse = np.sqrt(mse)  # 计算均方根误差（RMSE）
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("测试集:", r2_score(y_test, y_pred))
    print("训练集:", r2_score(y_train, y_pred_train))
    # 交叉验证
    reg = lgb.LGBMRegressor(iterations=200)  # 交叉验证中导入的是没有经过训练的模型，故不需要fit，只需要实例化
    print('CV', cross_val_score(reg, X_train, y_train, cv=5).mean())  # 五折交叉验证，返回R方
    # 创建数据框
    isScatter = False
    # 创建数据框
    if isScatter:
        # 创建数据框
        train_results_df = pd.DataFrame({
            'True Values': y_train,
            'Predicted Values': y_pred_train,
            'Dataset': 'Training Points'  # 标记为训练集
        })

        test_results_df = pd.DataFrame({
            'True Values': y_test,
            'Predicted Values': y_pred,
            'Dataset': 'Testing Points'  # 标记为测试集
        })

        # 合并数据框
        results_df = pd.concat([train_results_df, test_results_df])

        # 创建 JointGrid 对象
        plt.figure(figsize=(10, 6), dpi=1200)
        palette = {
            'Training Points': (1.0, 0.498, 0.314),  # RGB格式
            'Testing Points': (0.1216, 0.4667, 0.7059)  # RGB格式
        }
        g = sns.JointGrid(data=results_df, x="True Values", y="Predicted Values", hue="Dataset", height=6,
                          palette=palette)
        # 添加训练集的散点
        sns.scatterplot(data=train_results_df, x="True Values", y="Predicted Values",
                        ax=g.ax_joint,
                        label='Train Points',
                        marker='o', edgecolor=[255 / 255, 127 / 255, 80 / 255, .4],
                        facecolor=[255 / 255, 127 / 255, 80 / 255, .1], linewidth=0.5,
                        s=15,
                        )

        # 添加测试集的散点
        sns.scatterplot(data=test_results_df, x="True Values", y="Predicted Values",
                        ax=g.ax_joint,
                        label='Test Points',
                        marker='o', edgecolor=[0.1216, 0.4667, 0.7059, .4], facecolor=[0.1216, 0.4667, 0.7059, .1],
                        linewidth=0.5, s=15,
                        )
        # 计算完整范围
        full_range = np.linspace(results_df['True Values'].min(), results_df['True Values'].max(), 500)

        # 重新拟合直线
        fit_params = np.polyfit(test_results_df['True Values'], test_results_df['Predicted Values'], 1)
        fit_line = np.poly1d(fit_params)

        # 拟合线的斜率和截距
        a, b = fit_params

        # 预测值
        predicted_values = fit_line(full_range)

        # 手动计算置信区间
        confidence_level = 0.97
        alpha = 1 - confidence_level
        n = len(test_results_df)  # 样本数量
        t_value = t.ppf(1 - alpha / 2, df=n - 2)  # t分布临界值

        # 标准误差的计算
        x_mean = test_results_df['True Values'].mean()
        se = np.sqrt(np.sum((test_results_df['True Values'] - x_mean) ** 2))
        std_err = np.sqrt(
            np.mean((test_results_df['Predicted Values'] - fit_line(test_results_df['True Values'])) ** 2))

        # 置信区间宽度
        confidence_interval = t_value * std_err * np.sqrt(1 / n + (full_range - x_mean) ** 2 / se)

        # 绘制拟合线
        g.ax_joint.plot(full_range, predicted_values, label=f'Linear Fit: y = {a:.2f}x + {b:.2f}', color='#005EFF')

        # 绘制置信区间
        g.ax_joint.fill_between(full_range,
                                predicted_values - confidence_interval,
                                predicted_values + confidence_interval,
                                color='#005EFF', alpha=0.2, label='97% Confidence Interval')
        # 添加边缘的柱状图，多个数据集叠加显示
        sns.histplot(
            data=results_df,
            x="True Values",
            kde=False,
            ax=g.ax_marg_x,
            hue="Dataset",  # 使用 hue 分组
            palette=palette,  # 配置颜色
            alpha=0.35,
            legend=False,  # 禁用图例
            multiple="stack"  # 设置为叠加显示
        )

        sns.histplot(
            data=results_df,
            y="Predicted Values",
            kde=False,
            ax=g.ax_marg_y,
            hue="Dataset",  # 使用 hue 分组
            palette=palette,  # 配置颜色
            alpha=0.35,
            legend=False,  # 禁用图例
            multiple="stack"  # 设置为叠加显示
        )
        # 获取边缘柱状图中的每个柱子并设置其边框颜色
        for ax in [g.ax_marg_x, g.ax_marg_y]:
            for patch in ax.patches:
                patch.set_linewidth(0.2)  # 设置边框宽度
                patch.set_edgecolor((0, 0, 0))
        ax = g.ax_joint
        # 添加中心线
        ax.plot([results_df['True Values'].min(), results_df['True Values'].max()],
                [results_df['True Values'].min(), results_df['True Values'].max()], c="black", alpha=0.5,
                linestyle='--', label='Ideal Fit: y = x')
        # 创建一个假的线条对象来代表置信区间
        confidence_interval_patch = Rectangle((0, 0), 1, 1, linewidth=0, edgecolor='white', facecolor='#005EFF',
                                              alpha=0.2,
                                              label='97% Confidence Interval')

        # 获取当前轴对象
        ax = g.ax_joint  # 直接使用 g.ax_joint

        # 获取默认的图例项
        handles, labels = ax.get_legend_handles_labels()
        # 将最后一项移动到第三项
        if len(handles) > 2:  # 确保至少有3项
            handles.insert(2, handles.pop(-1))  # 把最后一项插入到索引2的位置
            labels.insert(2, labels.pop(-1))  # 同时调整对应的标签
        # 添加图例
        ax.legend(handles=handles, labels=labels, loc='upper left', frameon=False, facecolor='none')
        # 添加边界框
        for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
            if ax == g.ax_joint:
                # 设置边界框的颜色和线宽
                ax.spines['top'].set_visible(True)
                ax.spines['top'].set_color('black')
                ax.spines['top'].set_linewidth(1)

                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(1)

                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_color('black')
                ax.spines['left'].set_linewidth(1)

                ax.spines['right'].set_visible(True)
                ax.spines['right'].set_color('black')
                ax.spines['right'].set_linewidth(1)
            else:
                # 设置边界框的颜色和线宽
                ax.spines['bottom'].set_visible(True)
                ax.spines['bottom'].set_color('black')
                ax.spines['bottom'].set_linewidth(1)

                ax.spines['left'].set_visible(True)
                ax.spines['left'].set_color('black')
                ax.spines['left'].set_linewidth(1)
        # 确保边缘图的坐标轴数值可见
        g.ax_marg_x.set_ylabel("Frequency")
        g.ax_marg_y.set_xlabel("Frequency")
        g.ax_marg_x.yaxis.set_ticks_position('left')  # 显示y轴
        g.ax_marg_y.xaxis.set_ticks_position('bottom')  # 显示x轴
        g.ax_marg_x.yaxis.get_label().set_visible(True)
        g.ax_marg_y.xaxis.get_label().set_visible(True)
        # plt.show()
        # 保存影像 不能show show完就空白
        plt.savefig('scatter_LGBM.svg', dpi=1200, bbox_inches='tight', format='svg')  # 打印特征重要性

    sorted_idx = np.argsort(importance)
    for i in sorted_idx:
        print(f"{selected_columns[i]}: {importance[i]:.4f}")
    # 是否启用SHAP 可视化各要素重要性
    isShap = False
    if isShap:
        explainer = shap.Explainer(lgbm_reg)
        shap_values = explainer(x_filtered)
        shap.plots.waterfall(shap_values[0], 20)
        # print(shap_values)

    # 存储预测结果
    isSave = False
    if isSave:
        # 使用训练好的模型对全部数据进行预测（这里我们用selected_data_x作为例子）
        predictions = lgbm_reg.predict(selected_data_x)  # 对完整的特征集进行预测

        # 将原始dataframe和新列合并
        data['LGBM'] = predictions  # 'predictions'是新增加的列名

        # 输出包含'predictions'的新csv文件。假设original_file_path是你的原始csv路径。
        output_file_path = csv_path + file_name  # 设置输出文件路径
        data.to_csv(output_file_path, index=False)

        print(f"Prediction results have been saved to {output_file_path}")
    pre_name = "SP1201_50.csv"
    pre_file_path = csv_path + pre_name
    #  保存细化100的推理预测结果
    isSaveNewDataset = False
    if isSaveNewDataset:
        pre_data = pd.read_csv(pre_file_path)
        selected_predata_x = pre_data[selected_columns]
        predictions = lgbm_reg.predict(selected_predata_x)
        pre_data['LGBM'] = predictions
        output_file_path = pre_file_path
        pre_data.to_csv(output_file_path, index=False)
        print(f"New prediction results have been saved to {output_file_path}")
    # 显示对比折线图
    isLine = False
    isNone = False
    if isLine:

        if isNone:
            # 假设 y_test 和 y_pred 已经是你的真实值和预测值
            # 创建一个布尔索引，筛选出差值大于0.1或小于-0.1的点
            diff = y_pred - y_test
            mask = np.abs(diff) > 0.1

            # 使用CubicSpline生成平滑的曲线
            splines_actual = CubicSpline(x_values[mask], y_test[mask])
            splines_predicted = CubicSpline(x_values[mask], y_pred[mask])

            # 生成平滑曲线的x值
            x_smooth = np.linspace(min(x_values[mask]), max(x_values[mask]), 500)

            # 绘制所有的点，真实值和预测值
            plt.figure(figsize=(10, 5))

            # 绘制真实值和预测值的折线图
            plt.plot(x_values, y_test, label='Actual Values', color='#1E90FF', marker='o')
            plt.plot(x_values, y_pred, label='Predicted Values', color='#FF4500', linestyle='--', marker='x')

            # 绘制平滑的曲线
            plt.plot(x_smooth, splines_actual(x_smooth), label='Smoothed Actual Values', color='green', linewidth=2)
            plt.plot(x_smooth, splines_predicted(x_smooth), label='Smoothed Predicted Values', color='purple',
                     linewidth=2)

            # 添加图例
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

            # 添加标题和轴标签
            plt.title('LGBM ' + '(' + selected_y + ')')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.savefig('Line_LGBM.svg', dpi=1200, bbox_inches='tight', format='svg')
            # 显示图表
            plt.show()
        else:
            # 绘制实际值与预测值的折线图
            plt.figure(figsize=(10, 5))
            plt.plot(x_values, y_test, label='Actual Values', color='#4393c3', marker='o')
            plt.plot(x_values, y_pred, label='Predicted Values', color='#DC143C', linestyle='--', marker='x')

            # 添加统计量的文本
            # plt.text(0.04, 0.96, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}',
            #          verticalalignment='top', horizontalalignment='left',
            #          fontsize=12, color='black', transform=plt.gca().transAxes)
            # 添加图例、标题和轴标签，显示图表
            plt.legend()
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
            # 添加标题和轴标签
            plt.title('CatBoost' + '(' + selected_y + ')')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.savefig('Line_LGBM(None).svg', dpi=1200, bbox_inches='tight', format='svg')
