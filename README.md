<p align="center">
<h1 align="center">《动手学大模型》系列编程实践教程</h1>
</p>
<p align="center">
  	<a href="https://img.shields.io/badge/version-v0.1.0-blue">
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a >
       <img alt="Status-building" src="https://img.shields.io/badge/Status-building-blue" />
  	</a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
   	<a href="https://github.com/Lordog/dive-into-llms/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/Lordog/dive-into-llms" />
  	</a>
  	<a href="https://github.com/Lordog/dive-into-llms/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/Lordog/dive-into-llms?color=FF8000" />
  	</a>
    <a href="https://github.com/Lordog/dive-into-llms/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/Lordog/dive-into-llms?color=0088ff"/>
    </a>
    <br />
</p>


<div align="center">
<p align="center">
  <a href="#项目动机">项目动机</a>/
  <a href="#教程目录">教程目录</a>/
  <a href="#贡献者列表">贡献者列表</a>
</p>
</div>

## 💡 Updates

2025/04/28  感谢各位朋友们的关注和积极反馈！我们近期将从两个方面对本教程进行更新：

- [x] 上线国产化《大模型开发全流程》公益教程（含PPT、实验手册和视频），此处特别感谢华为昇腾社区的支持！
- [ ] 在原系列编程实践教程的基础上进行内容更新，并增加新的主题（数学推理、GUI Agent、大模型对齐等）。预计五一假期后完工，敬请期待！

## 🎯 项目动机

《动手学大模型》系列编程实践教程，由上海交通大学2024年春季《人工智能安全技术》课程（NIS3353）讲义拓展而来（教师：[张倬胜](https://bcmi.sjtu.edu.cn/home/zhangzs/)），旨在提供大模型相关的入门编程参考。本教程属公益性质、完全免费。通过简单实践，帮助同学们快速入门大模型，更好地开展课程设计或学术研究。

## 📚 教程目录

| 教程内容           | 简介                                                         | 地址                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 微调与部署         | 预训练模型微调与部署指南：想提升预训练模型在指定任务上的性能？让我们选择合适的预训练模型，在特定任务上进行微调，并将微调后的模型部署成方便使用的Demo！ | [[Slides](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1/dive-tuning.pdf)] [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter1/README.md)] |
| 提示学习与思维链   | 大模型的API调用与推理指南：“AI在线求鼓励？大模型对一些问题的回答令人大跌眼镜，但它可能只是想要一句「鼓励」” | [[Slides](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter2/dive-prompting.pdf)] [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter2/README.md)] |
| 知识编辑           | 语言模型的编辑方法和工具：想操控语言模型在对指定知识的记忆？让我们选择合适的编辑方法，对特定知识进行编辑，并将对编辑后的模型进行验证！ | [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter3/dive_edit_0410.pdf)] [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter3/README.md)] |
| 模型水印           | 语言模型的文本水印：在语言模型生成的内容中嵌入人类不可见的水印 | [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter4/watermark.pdf)] [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter4/README.md)] |
| 越狱攻击           | 想要得到更好的安全，要先从学会攻击开始。让我们了解越狱攻击如何撬开大模型的嘴！ | [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter5/dive-Jailbreak.pdf)]  [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter5/README.md)] |
| 多模态模型         | 作为能够更充分模拟真实世界的多模态大语言模型，其如何实现更强大的多模态理解和生成能力？多模态大语言模型是否能够帮助实现AGI？ | [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter6/mllms.pdf)]  [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter6/README.md)] |
| 大模型智能体与安全 | 大模型智能体迈向了未来操作系统之旅。然而，大模型在开放智能体场景中能意识到风险威胁吗？ | [[Slides](https://github.com/Lordog/dive-into-llms/blob/main/documents/chapter7/agent.pdf)] [[Tutorial](https://github.com/Lordog/dive-into-llms/tree/main/documents/chapter7/README.md)] |

## 🔥 新上线：国产化《大模型开发全流程》

- **✨ 我们联合华为昇腾推出的《大模型开发全流程》公益教程正式上线！前沿技术+代码实践，手把手带你玩转AI大模型 ✨**: 

  在《动手学大模型》原系列教程的基础上，我们联合华为开发了《大模型开发全流程》系列课程。本系列教程基于昇腾基础软硬件开发，覆盖PPT、实验手册、视频等教程形式。该教程分为初级、中级、高级系列，面向不同的大模型实践需求，旨在将前沿技术通过代码实践的方式，为相关研究者、开发者由浅入深地提供快速上手、应用昇腾已支持模型和全新模型迁移调优的全流程开发指南。
  
- **🚀 前往昇腾社区探索《大模型开发全流程》系列课程**： 
  
  👉《[大模型开发学习专区](https://www.hiascend.com/edu/growth/lm-development#classification-floor-1)》@ 昇腾社区 👈 
  
- **✨ 课程内容展示 ✨**

  <!-- <img src="./pics/icon/title.jpg" width="300"/>
  <img src="./pics/icon/cover.png" width="300"/>
  <img src="./pics/icon/team.png" width="300"/>
  <img src="./pics/icon/agent.png" width="300"/> -->

<p align = "center">
  <img src="./pics/icon/title.jpg" width="48%"/>
  <img src="./pics/icon/cover.png" width="48%"/>
  <img src="./pics/icon/team.png" width="48%"/>
  <img src="./pics/icon/agent.png" width="48%"/>
</p>

## 🙏 免责声明

本教程所有内容仅仅来自于贡献者的个人经验、互联网数据、日常科研工作中的相关积累。所有技巧仅供参考，不保证百分百正确。若有任何问题，欢迎提交 Issue 或 PR。另本项目所用徽章来自互联网，如侵犯了您的图片版权请联系我们删除，谢谢。

## 🤝 欢迎贡献

本教程目前是一个正在进行中的项目，如有疏漏在所难免，欢迎任何的PR及issue讨论。

## ❤️ 贡献者列表

感谢以下老师和同学对本项目的支持与贡献：

**《动手学大模型》系列教程开发团队**：

- 上海交通大学：[张倬胜](https://bcmi.sjtu.edu.cn/home/zhangzs/)、[袁童鑫](https://github.com/Lordog)、[马欣贝](https://scholar.google.com/citations?user=LpUi3EgAAAAJ&hl=zh-CN&oi=ao)、 [何志威](https://zwhe99.github.io)、[杜巍](https://scholar.google.com/citations?user=tFYUBLkAAAAJ&hl=en)、[赵皓东](https://dongdongzhaoup.github.io/)、[吴宗儒](https://zrw00.github.io/)； 

- 新加坡国立大学：[费豪](http://haofei.vip/)

**《大模型开发全流程》系列教程开发团队：**

- 上海交通大学：[张倬胜](https://bcmi.sjtu.edu.cn/home/zhangzs/)、[刘功申](https://infosec.sjtu.edu.cn/DirectoryDetail.aspx?id=75)、[陈星宇](https://scholar.google.com/citations?user=d-dNtjrMJ5YC&hl=en)、[程彭洲](https://scholar.google.com/citations?user=qxnwzDUAAAAJ&hl=en)、[董凌众](https://github.com/LZ-Dong)、 [何志威](https://zwhe99.github.io)、[鞠天杰](https://scholar.google.com/citations?user=f8PPcnoAAAAJ&hl=en)、[马欣贝](https://scholar.google.com/citations?user=LpUi3EgAAAAJ&hl=zh-CN&oi=ao)、 [吴铮](https://scholar.google.com/citations?hl=zh-CN&user=qBM1UbUAAAAJ&view_op=list_works&gmla=AIfU4H6PG9JyjRub6BYIIZ4isQE7MBAM3Eoec6OJfX4z_8-pOE8bI1Wgdo3XL5qOZWR3U-h-lIP2q0zXt5gzyFKMSg7MNnBBWLv5d1IVG30UANczTP0)、[吴宗儒](https://zrw00.github.io/)、[闫子赫](https://scholar.google.com/citations?user=O2YfSHoAAAAJ&hl=zh-CN)、[姚杳](https://scholar.google.com/citations?user=tLMP3IkAAAAJ)、[袁童鑫](https://github.com/Lordog)、[赵皓东](https://dongdongzhaoup.github.io/);

- 华为昇腾社区：ZOMI、谢乾、程黎明、楼梨华、焦泽昱

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Lordog/dive-into-llms&type=Date)](https://star-history.com/#Lordog/dive-into-llms&Date)
