import streamlit as st
from PIL import Image

st.title("高频交易中的短期波动率预测")
st.subheader("""
简介
""")
st.write(r"""本文主要介绍使用某期货主力合约5个交易日的数据，建立波动率预测模型的特征选择和预测结果.
""")


st.subheader(r"""
特征工程和采样
""")
st.write(r"""
在这一小节中，我们将详细介绍项目对于波动率的定义和特征工程，以及具体生成样本的算法。
""")

st.write(r"""
对于有交易记录的时间点 $t$，记 $P_t$ 为此时刻的最新成交价，$BP_{t,i}, AP_{t,i}$ 为此时刻第 $i$ 档买卖价，
$BV_{t,i}, AV_{t,i}$ 为此时刻第 $i$ 档买卖量，$M_t$为上一记录时刻到此时刻的成交量变化，$N_t$ 为上一记录时刻到此时刻的成交金额变化。
    对于采样时间点$t_0$，记 $t_0<t_1<\dots<t_n$ 为未来一分钟时间段 $(t_0, t_0+60]$ 内所有的交易数据时间点，$t_{-m}<\dots<t_{-1}\leq t_0$为过去 $[t_0-l, t_0]$ 内所有的交易数据时间点，其中 $l=60$ 是选取的窗口长度，记$\Delta t_i=t_i-t_{i-1}$。
""")

st.markdown("#### 波动率的定义")
st.write("""对于时间点$t$，定义其**未来一分钟的波动率** $V_t$ 为""")
st.latex(r"""V_t=\log\left[\frac{1}{t_n-t_1}\sum_{i=1}^{n-1}(\log P_{t_{i+1}}-\log P_{t_i})^2\right]""")
st.markdown(r"""对应字段 rv. 这样定义是因为
- 波动描述交易价格的起伏程度，因此连续的数据之间，无论是交易价格增加或减少，都应当对波动产生正向贡献           
- 一般常用二次变差 $\int_{t_0}^{t_0+60}\sigma^2dt$ 描述函数在一定范围内的波动（它具有良好的数学性质），由于我们预测的是未来1分钟内的波动率，而原始数据的频率多数在1秒内，因此上述算法是二次变差的近似值
- 符合人们通过波动来决定不确定性风险的担忧，例如持续稳定的上涨和下跌不应当是波动考虑的范围（这也是为什么不考虑绝对值和），平方和更容易反映剧烈跳变
- 使用对数收益率是为了一方面为了可加性，另一方面符合人们日常生活非线性的感知习惯
- 外层额外再加一个对数是为了后续拟合模型时使目标变量更接近正态分布，因为平方和容易产生重尾
""")

st.markdown("#### 特征设计")
st.markdown("##### 1.过去窗口的波动率 ")
st.latex(r"""\log\left[\frac{1}{t_{-1}-t_{-m}}\sum_{i=-m+1}^{-1}(\log P_{t_i}-\log P_{t_{i-1}})^2\right]""")
st.markdown(r"""past_rv. 中高频预测时波动率可能具有惯性，过去一段时间的波动率有可能持续下去，能够感知更大范围的不稳定性. 这个值越高，未来的波动率也应当越高.
""")

st.markdown("##### 2.过去窗口的正收益时间加权占比")
st.latex(r"""\frac1{t_{-1}-t_{-m}}\sum_{i=-m+1}^{-1}\Delta t_i\mathbf{1}(P_i>P_{i-1})""")
st.markdown(r"""past_up_ratio. 过去时间窗口中持续上涨的时间越久，表明交易市场上买方更主动，从而存在单边压力使对手做出调整，持续的正收益或负收益也意味着未来很有可能突然反转. 这个值推测买方主动还是卖方主动的市场更容易造成波动.
""")

st.markdown("##### 3.过去窗口内买卖价差加权平均")
st.latex(r"""\frac1{t_{-1}-t_{-m}}\left|\sum_{i=-m+1}^{-1}\Delta t_i\sum_{j=1}^5BP_{t_i,j}BV_{t_i,j}-AP_{t_i,j}AV_{t_i,j}\right|""")
st.markdown(r"""past_weighted_bid_ask_price_diff. 这一项表示一段时间买方或卖方前五档的总支付意愿差距，当有一方挂单的总支付额度远超过另一方时，很容易一次就连续吃掉对手多档，造成波动. 这个值越高，未来的波动率也应当越高.
""")

st.markdown("##### 4.过去窗口内买卖量不平衡加权平均")
st.latex(r"""\frac1{t_{-1}-t_{-m}}\sum_{i=-m+1}^{-1}\Delta t_i\left|\frac{\sum_{j=1}^5BV_{t_i,j}-AV_{t_i,j}}{\sum_{j=1}^5BV_{t_i,j}+AV_{t_i,j}}\right|""")
st.markdown(r"""past_weighted_bid_ask_volume_diff. 这一项反应过去一段时间的买卖量不平衡，当不平衡严重时，说明需要更大幅度调整价格来吃掉挂单. 这个值越高，未来的波动率也应当越高.
""")

st.markdown("##### 5.过去窗口带驱动方向的总成交量")
st.latex(r"""\sum_{i=-m+1}^{-i}M_{t_i}\text{sgn}(P_i-P_{i-1})""")
st.markdown(r"""past_signed_volume. 这一项反映过去一段时间实际发生的交易属于买方主动还是卖方主动.
""")

st.markdown("##### 其他特征")
st.markdown(r"""
6.过去窗口的总对数收益率：past_ac_log_return.
            
7.过去窗口的成交价范围：past_price_range.
            
8.过去窗口内买卖价加权平均：past_weighted_bid_ask_price_sum.
            
9.过去窗口内买卖加权中价变化范围：past_weighted_mid_bid_ask_price_range.
            
10.过去窗口的总成交量：past_total_volume.
            
11.过去窗口的总交易额：past_total_value.
            
12.过去窗口带驱动方向的成交量波动率：past_signed_volume_var.
""")

st.markdown("#### 生成样本")
st.markdown(r"""每天的交易时间可以分为9:00-10:15，10:30-11:30，13:30-15:00和21:00-23:00四个时间段，且交易时间段内存在一些高频数据中断的情况，其中连续10秒没有记录共发生7次，连续5秒没有记录共发生249次。
将全部数据按照以交易时间段和数据中断划分为若干片段，对于每个片段，从片段开始1分钟后到距离结束1分钟前，每一整秒采样一次.
            
最终用于训练和测试的数据共有82651条，每一条数据包含采样时间和按照上述规则计算的未来1分钟波动率和特征.
""")

st.markdown("#### 特征与与目标波动率的统计关系")

st.markdown(r"""下图展示了生成样本后，波动率和特征的频率分布. 波动率的分布接近正态，除了过去窗口内买卖价加权平均，其他特征多数呈现正态或半正态分布，说明这些数据适合用于后续线性模型.
""")

with open('diagrams/histofvariable.html', 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=300)



st.markdown(r"""下图展示了所有特征和波动率之间的Pearson相关系数（最基本的IC值）从高到低排列和和单变量回归时的p值，基本符合在定义特征时对其与目标波动率的统计关系倾向:
""")
with open('diagrams/pearson.html', 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=300)


st.subheader("模型和预测结果")
st.markdown("#### 验证方案")
st.markdown(r"""在实际应用中，目标是实时预测未来该期货合约交易的波动率，在测试模型性能时和评估特征效果时，为了避免**前瞻性偏差**（即使用了尚未发生的信息，它会高估模型的实际效果，尤其是使用了信息高度重叠的数据），
            本项目使用了2025-09-22至2025-09-25四个交易日的数据作为训练集（共66612条），使用2025-09-26交易日的数据作为测试集（共16039条）进行预测，并且在交叉验证选择系数时，训练集的时间范围也在测试集前. 模型具体的参数设置可以在代码找到.
""")

st.markdown("#### 模型选择")
st.markdown("##### 基线模型")

st.markdown(r"""基线模型直接使用上1分钟的波动率作为未来1分钟波动率的预测，在本文中选取时间窗口长度$l=60$，因此和第一个特征基本一致.
""")

st.markdown("##### LASSO回归")

st.markdown(r"""
- LASSO回归的L1惩罚项能够控制参数绝对值大小，在拟合变量时具有稀疏性，能够帮助筛选特征，特别是在高维模型，模型解释能力强
- 构造的特征很有可能存在多重共线性问题，如果采用简单的回归模型，由于金融数据的噪声大，不稳定因素多，容易使特征拟合的系数不稳定
- 计算速度极快
- 实际目标更倾向于追求波动率预测的准确度，而非变量参数的可解释性，LASSO回归的预测能力可能稍低
- 数据高度依赖时序，自相关，可能高估拟合优度和p值，模型可能倾向发现趋势而非特征与目标的因果关系
""")
st.markdown("##### XGBoost")
st.markdown(r"""
- XGBoost是非参数模型，不需要对原数据做独立和正态性假设
- 模型有一定解释能力，树模型适合“价格和数量变化差超过阈值，就会调整交易行为”的交易习惯
- 可以通过设置树的最大分裂次数、惩罚项系数、部分抽样训练来避免过拟合，特别是在噪声较大的高频金融数据
- 计算速度较快，但是调参所需时间高
""")

st.markdown("#### 预测结果")
st.markdown(r"""下列图表依次展示了模型预测结果与实际值的对比随测试集时间的变化、三种模型的RMSE、MAE和平方拟合优度 $R^2$，可以观察到，加入新的特征对预测性能有小幅改进.
""")

with open('diagrams/predict.html', 'r', encoding='utf-8') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=300)

f = Image.open('diagrams/eval.png')
st.image(f, caption="模型预测评估", use_container_width=False)

st.markdown(r"""下列表展示了LASSO回归模型的拟合后每个特征的系数和XGBoost模型每个特征用其分裂时对降低损失函数下的贡献.
""")

col1, col2 = st.columns(2)
with col1:
    f1 = Image.open('diagrams/lassocoef.png')
    st.image(f1, caption="LASSO系数")
with col2:
    f2 = Image.open('diagrams/xgbgain.png')
    st.image(f2, caption="XGBoost Gain")

st.markdown(r"""结果显示，LASSO保留了四个变量和XGBoost模型的Gain顺序基本一致。除去过去一分钟的波动、总交易量和交易额外，XGBoost的模型指出特征2,3,4都能够提升预测性能.
            """)

st.subheader("改进方向")
st.markdown(r"""
- 采用不同的时间窗口长度、不同的多项式次数、不同的聚合方法等，或对已有的特征作变换、一阶差分等，批量构造更多特征并纳入模型训练，筛选出合适的特征.
- 扩展数据范围并持续进行滚动窗口测试，即持续以 $(t-t_m,t]$ 的数据作为训练集，$(t, t+t_n)$ 的数据作为测试集，观察随着 $t$ 变化各个特征的IC和模型总体性能的表现，筛选在长时间过程中保持影响力的特征.
- 对XGBoost模型进行更细致的超参数网格搜索，以求接近最佳拟合效果，并根据测试集结果对比
- 尝试更多模型，例如随机森林、RNN（和LSTM），高频预测性能可能因为噪声原因不会有太大改善
- 引入事件驱动的特征，例如成交量激增点、挂单价突变点，以事件形式预测未来波动率增加发生在多久之后
""")