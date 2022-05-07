# DocEE-Application
Leveraging DocEE corpus to implement an online news event extraction system


# DocEE部署说明书
## 环境安装
 ```python
    - conda create --name final python=3.8.11
    - python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    - python -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
    - python -m pip install transformers==4.12.5
    - python -m pip install nltk
    - python -m pip install datasets
    - python -m pip install sklearn
 ```   
## 事件抽取

```python
    - python predict_classify.txt
    - python predict_EE.py
```
## 数据格式
输入数据：test.pkl
```
['浙江高速隧道事故致5死31伤：送医伤员大多被熏黑', '甬台温高速猫狸岭隧道内发生车祸 有人员伤亡 \n\n据临海发布：8月27日19:20许，甬台温高速猫狸岭隧道杭州往临海方向发生一起交通事故，目前临海市正在全力抢救。据初步统计，共有36人送医院救治，其中已死亡5人。\n\n记者了解到，8月27日晚上18:54，浙江台州高速交警支队官方微博发布信息：G15甬台温高速往温州方向1607KM，猫狸岭隧道内，货车起火，吴岙实行卡口，三门进口、上三高速白鹤、天台、洋头进口关闭，往三门方向天台出口分流，因宁波方向烟雾影响，羊角山进洞口实行卡口，宁向临海北出口实行分流，临海北至温岭西各进口关闭。\n\n浙江一辆载满轮胎的大货车在隧道内起火 \n\n据悉，事故发生时间为18:27分许，一辆安徽牌照的半挂车因轮胎爆炸发生事故，引发车上货物起火。5分钟后台州高速交警到达现场，因火势过大无法灭火，火灾引发的烟雾造成过往车辆停于火灾现场无法行驶。\n\n\n昨天23:59，记者从临海市第一人民医院获悉，昨晚9点半开始陆陆续续有伤员送到这里，有救护车送来的，也有私家车送来的，共18人，大多全身熏黑，目前为止2人死亡。一名小朋友被转往台州医院救治。']
```
输出数据
```
[{'content': '甬台温高速猫狸岭隧道内发生车祸 有人员伤亡 \n\n据临海发布：8月27日19:20许，甬台温高速猫狸岭隧道杭州往临海方向发生一起交通事故，目前临海市正在全力抢救。据初步统计，共有36人送医院救治，其中已死亡5人。\n\n记者了解到，8月27日晚上18:54，浙江台州高速交警支队官方微博发布信息：G15甬台温高速往温州方向1607KM，猫狸岭隧道内，货车起火，吴岙实行卡口，三门进口、上三高速白鹤、天台、洋头进口关闭，往三门方向天台出口分流，因宁波方向烟雾影响，羊角山进洞口实行卡口，宁向临海北出口实行分流，临海北至温岭西各进口关闭。\n\n浙江一辆载满轮胎的大货车在隧道内起火 \n\n据悉，事故发生时间为18:27分许，一辆安徽牌照的半挂车因轮胎爆炸发生事故，引发车上货物起火。5分钟后台州高速交警到达现场，因火势过大无法灭火，火灾引发的烟雾造成过往车辆停于火灾现场无法行驶。\n\n\n昨天23:59，记者从临海市第一人民医院获悉，昨晚9点半开始陆陆续续有伤员送到这里，有救护车送来的，也有私家车送来的，共18人，大多全身熏黑，目前为止2人死亡。一名小朋友被转往台州医院救治。', 'annotations': {'日期': ['8月27日19:20许'], '地点': ['甬台温高速猫狸岭隧道杭州往临海方向'], '人员伤亡': ['']}, 'golden': [{'type': '日期'}, {'type': '地点'}, {'type': '人员伤亡'}]}]
```
