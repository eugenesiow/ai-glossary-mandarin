<h1 align="center">AI Glossary in Mandarin</h1>

<div align="center">
  <strong>人工智能术语/词汇库</strong>
</div>
<div align="center">
  An <code>English</code> to <code>Mandarin</code> glossary of AI terminology grouped topically by areas (e.g. NLP) and tasks (e.g. NER).
</div>

<br />

<div align="center">
  <!-- Version -->
  <a href="https://github.com/eugenesiow/ai-glossary-mandarin">
    <img src="https://img.shields.io/badge/version-1.0-blue.svg?style=flat-square"
      alt="Version" />
  </a>
  <!-- License -->
  <a href="https://github.com/eugenesiow/ai-glossary-mandarin/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-CC--BY--SA-green.svg?style=flat-square"
      alt="License" />
  </a>
  <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
  <!-- ALL-CONTRIBUTORS-BADGE:END -->
</div>

<div align="center">
  <h3>
    <a href="https://news.machinelearning.sg/posts/an_english_to_mandarin_ai_glossary_of_state_of_the_art_topics">
      Article
    </a>
    <span> | </span>
    <a href="CONTRIBUTING.md">
      Contributing
    </a>
  </h3>
</div>

## Table of Contents
- [Introduction](#introduction)
- [Natural Language Processing (NLP) `自然语言处理`](#natural-language-processing-nlp-自然语言处理)
- [Computer Vision (CV) `计算机视觉`](#computer-vision-cv-计算机视觉)
- [Robotics `机器人技术`](#robotics-机器人技术)
- [Basic Theory `基础理论`](#basic-theory-基础理论)
- [Optimization `运筹优化`](#optimization-运筹优化)
- [Causal Inference `因果推理`](#causal-inference-因果推理)
- [Knowledge Engineering `知识工程`](#knowledge-engineering-知识工程)
- [Data Science `数据科学`](#data-science-数据科学)
- [Model Optimization `模型优化`](#model-optimization-模型优化)
- [Information Retrieval (IR) `信息检索`](#information-retrieval-ir-信息检索)
- [Recommender System `推荐系统`](#recommender-system-推荐系统)
- [Autonomous Driving `自动驾驶技术`](#autonomous-driving-自动驾驶技术)
- [Generative Adversarial Networks (GAN) `生成对抗网络`](#generative-adversarial-networks-gan-生成对抗网络)
- [Machine Learning (ML) `机器学习`](#machine-learning-ml-机器学习)
- [Deep Learning (DL) `深度学习`](#deep-learning-dl-深度学习)
- [Reinforcement Learning (RL) `强化学习`](#reinforcement-learning-rl-强化学习)
- [Multimodal Learning `多模态学习`](#multimodal-learning-多模态学习)
- [Transfer Learning `迁移学习`](#transfer-learning-迁移学习)
- [Meta Learning `元学习`](#meta-learning-元学习)
- [Self-Supervised Learning `自监督学习`](#self-supervised-learning-自监督学习)
- [Semi-Supervised Learning `半监督学习`](#semi-supervised-learning-半监督学习)
- [Unsupervised Learning `无监督学习`](#unsupervised-learning-无监督学习)
- [AI Safety `人工智能安全`](#ai-safety-人工智能安全)
- [Others `其他`](#others-其他)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)
- [License](#license)

## Introduction

This is an effort to create an **English** to **Mandarin** glossary of AI terminology, grouped topically by areas (e.g. NLP) and then tasks (e.g. NER). The areas are largely borrowed and link to the open data from [Papers With Code](https://paperswithcode.com/), which tracks the state-of-the-art across AI/ML tasks.

When possible official or common terminology is used, in the absence of existing terminology, literal translation is used. Do request to contribute if you would like to add or modify the glossary terminology/translations.

The sort order of the tables are not alphabetical but by the number of datasets listed (the find function would help find a specific topic).

## Natural Language Processing (NLP) `自然语言处理`

|                                                   English                                                    |         Mandarin         |Datasets|                                               Subtasks                                               |
|--------------------------------------------------------------------------------------------------------------|--------------------------|-------:|------------------------------------------------------------------------------------------------------|
|[Question Answering](https://paperswithcode.com/task/question-answering)                                      |问答系统                  |      48|[9](https://paperswithcode.com/area/natural-language-processing/question-answering)                   |
|[Machine Translation](https://paperswithcode.com/task/machine-translation)                                    |机器翻译                  |      46|[6](https://paperswithcode.com/area/natural-language-processing/machine-translation)                  |
|[Text Classification](https://paperswithcode.com/task/text-classification)                                    |该技术                    |      29|[10](https://paperswithcode.com/area/natural-language-processing/text-classification)                 |
|[Sentiment Analysis](https://paperswithcode.com/task/sentiment-analysis)                                      |情感分析                  |      28|[7](https://paperswithcode.com/area/natural-language-processing/sentiment-analysis)                   |
|[Named Entity Recognition](https://paperswithcode.com/task/named-entity-recognition)                          |命名实体识别/专名辨识     |      25|[7](https://paperswithcode.com/area/natural-language-processing/named-entity-recognition)             |
|[Document Classification](https://paperswithcode.com/task/document-classification)                            |文件密级：                |      18|[1](https://paperswithcode.com/area/natural-language-processing/document-classification)              |
|[Relation Extraction](https://paperswithcode.com/task/relation-extraction)                                    |关系抽取任务              |      17|[6](https://paperswithcode.com/area/natural-language-processing/relation-extraction)                  |
|[Natural Language Inference](https://paperswithcode.com/task/natural-language-inference)                      |自然语言推理              |      16|[1](https://paperswithcode.com/area/natural-language-processing/natural-language-inference)           |
|[Semantic Parsing](https://paperswithcode.com/task/semantic-parsing)                                          |语义分析                  |      16|[4](https://paperswithcode.com/area/natural-language-processing/semantic-parsing)                     |
|[Speech Recognition](https://paperswithcode.com/task/speech-recognition)                                      |语音识别                  |      13|[7](https://paperswithcode.com/area/speech/speech-recognition)                                        |
|[Text Summarization](https://paperswithcode.com/task/text-summarization)                                      |自动摘要                  |      13|[7](https://paperswithcode.com/area/natural-language-processing/text-summarization)                   |
|[Dependency Parsing](https://paperswithcode.com/task/dependency-parsing)                                      |依存句法分析              |      11|[3](https://paperswithcode.com/area/natural-language-processing/dependency-parsing)                   |
|[Word Sense Disambiguation](https://paperswithcode.com/task/word-sense-disambiguation)                        |词义消歧                  |      11|0                                                                                                     |
|[Cross-Lingual Document Classification](https://paperswithcode.com/task/cross-lingual-document-classification)|跨语言文档分类            |      10|[1](https://paperswithcode.com/area/natural-language-processing/cross-lingual-document-classification)|
|[Entity Linking](https://paperswithcode.com/task/entity-linking)                                              |实体链接                  |      10|0                                                                                                     |
|[Aspect-Based Sentiment Analysis](https://paperswithcode.com/task/aspect-based-sentiment-analysis)            |基于方面的情感分析        |       9|[3](https://paperswithcode.com/area/natural-language-processing/aspect-based-sentiment-analysis)      |
|[Text Generation](https://paperswithcode.com/task/text-generation)                                            |文本生成                  |       8|[16](https://paperswithcode.com/area/natural-language-processing/text-generation)                     |
|[Entity Disambiguation](https://paperswithcode.com/task/entity-disambiguation)                                |实体消除歧义              |       7|0                                                                                                     |
|[Speech Enhancement](https://paperswithcode.com/task/speech-enhancement)                                      |语音增强                  |       7|[1](https://paperswithcode.com/area/speech/speech-enhancement)                                        |
|[Word Alignment](https://paperswithcode.com/task/word-alignment)                                              |字对齐                    |       7|0                                                                                                     |
|[Coreference Resolution](https://paperswithcode.com/task/coreference-resolution)                              |核心分辨率                |       6|[1](https://paperswithcode.com/area/natural-language-processing/coreference-resolution)               |
|[Grammatical Error Correction](https://paperswithcode.com/task/grammatical-error-correction)                  |语法错误纠正              |       6|[1](https://paperswithcode.com/area/natural-language-processing/grammatical-error-correction)         |
|[Multi-Label Classification](https://paperswithcode.com/task/multi-label-classification)                      |多标签分类                |       6|0                                                                                                     |
|[Sarcasm Detection](https://paperswithcode.com/task/sarcasm-detection)                                        |讽刺检测                  |       6|0                                                                                                     |
|[Speech Separation](https://paperswithcode.com/task/speech-separation)                                        |语音分离                  |       6|[1](https://paperswithcode.com/area/speech/speech-separation)                                         |
|[Conversational Response Selection](https://paperswithcode.com/task/conversational-response-selection)        |对话响应选择              |       5|0                                                                                                     |
|[Phrase Grounding](https://paperswithcode.com/task/phrase-grounding)                                          |短语接地                  |       5|0                                                                                                     |
|[Recipe Generation](https://paperswithcode.com/task/recipe-generation)                                        |配方生成                  |       5|0                                                                                                     |
|[Chunking](https://paperswithcode.com/task/chunking)                                                          |块状                      |       4|0                                                                                                     |
|[Cover Song Identification](https://paperswithcode.com/task/cover-song-identification)                        |封面歌曲标识              |       4|0                                                                                                     |
|[Cross-Lingual Bitext Mining](https://paperswithcode.com/task/cross-lingual-bitext-mining)                    |跨语言双文本采矿          |       4|0                                                                                                     |
|[Entity Typing](https://paperswithcode.com/task/entity-typing)                                                |实体类型                  |       4|0                                                                                                     |
|[Hate Speech Detection](https://paperswithcode.com/task/hate-speech-detection)                                |仇恨言论检测              |       4|0                                                                                                     |
|[Paraphrase Identification](https://paperswithcode.com/task/paraphrase-identification)                        |复述检测                  |       4|0                                                                                                     |
|[Semantic Textual Similarity](https://paperswithcode.com/task/semantic-textual-similarity)                    |语义文本相似性            |       4|[1](https://paperswithcode.com/area/natural-language-processing/semantic-textual-similarity)          |
|[Audio Denoising](https://paperswithcode.com/task/audio-denoising)                                            |音频去诺化                |       3|0                                                                                                     |
|[Constituency Parsing](https://paperswithcode.com/task/constituency-parsing)                                  |选区解析                  |       3|[1](https://paperswithcode.com/area/natural-language-processing/constituency-parsing)                 |
|[Hypernym Discovery](https://paperswithcode.com/task/hypernym-discovery)                                      |超名发现                  |       3|0                                                                                                     |
|[Intent Detection](https://paperswithcode.com/task/intent-detection)                                          |意图检测                  |       3|0                                                                                                     |
|[Question Generation](https://paperswithcode.com/task/question-generation)                                    |问题生成                  |       3|0                                                                                                     |
|[Reading Comprehension](https://paperswithcode.com/task/reading-comprehension)                                |机器阅读理解              |       3|[3](https://paperswithcode.com/area/natural-language-processing/reading-comprehension)                |
|[Slot Filling](https://paperswithcode.com/task/slot-filling)                                                  |插槽填充                  |       3|0                                                                                                     |
|[Amr Parsing](https://paperswithcode.com/task/amr-parsing)                                                    |抽象语义表示              |       2|0                                                                                                     |
|[Bias Detection](https://paperswithcode.com/task/bias-detection)                                              |偏置检测                  |       2|[1](https://paperswithcode.com/area/natural-language-processing/bias-detection)                       |
|[Fake News Detection](https://paperswithcode.com/task/fake-news-detection)                                    |假新闻检测                |       2|0                                                                                                     |
|[Graph-to-Sequence](https://paperswithcode.com/task/graph-to-sequence)                                        |指将图的输入格式转换成序列|       2|[1](https://paperswithcode.com/area/natural-language-processing/graph-to-sequence)                    |
|[Linguistic Acceptability](https://paperswithcode.com/task/linguistic-acceptability)                          |语言可接受性              |       2|0                                                                                                     |
|[Passage Re-Ranking](https://paperswithcode.com/task/passage-re-ranking)                                      |通行重新排序              |       2|0                                                                                                     |
|[Semantic Role Labeling](https://paperswithcode.com/task/semantic-role-labeling)                              |语义角色标记              |       2|[3](https://paperswithcode.com/area/natural-language-processing/semantic-role-labeling)               |
|[Sign Language Translation](https://paperswithcode.com/task/sign-language-translation)                        |手语翻译                  |       2|0                                                                                                     |
|[Speech Synthesis](https://paperswithcode.com/task/speech-synthesis)                                          |语音合成                  |       2|[2](https://paperswithcode.com/area/speech/speech-synthesis)                                          |
|[Table-to-Text Generation](https://paperswithcode.com/task/table-to-text-generation)                          |表到文本的生成            |       2|[1](https://paperswithcode.com/area/natural-language-processing/table-to-text-generation)             |
|[Text Style Transfer](https://paperswithcode.com/task/text-style-transfer)                                    |文本样式传输              |       2|[1](https://paperswithcode.com/area/natural-language-processing/text-style-transfer)                  |
|[Text-To-Speech Synthesis](https://paperswithcode.com/task/text-to-speech-synthesis)                          |文本到语音合成            |       2|[1](https://paperswithcode.com/area/natural-language-processing/text-to-speech-synthesis)             |
|[Acoustic Novelty Detection](https://paperswithcode.com/task/acoustic-novelty-detection)                      |声学新颖性检测            |       1|0                                                                                                     |
|[Arabic Text Diacritization](https://paperswithcode.com/task/arabic-text-diacritization)                      |阿拉伯语文本分文化        |       1|0                                                                                                     |
|[Audio Generation](https://paperswithcode.com/task/audio-generation)                                          |音频生成                  |       1|[1](https://paperswithcode.com/area/audio/audio-generation)                                           |
|[CCG Supertagging](https://paperswithcode.com/task/ccg-supertagging)                                          |组合范畴语法超标注        |       1|0                                                                                                     |
|[Code Summarization](https://paperswithcode.com/task/code-summarization)                                      |代码汇总                  |       1|[1](https://paperswithcode.com/area/computer-code/code-summarization)                                 |
|[Counterspeech Detection](https://paperswithcode.com/task/counterspeech-detection)                            |反信号检测                |       1|0                                                                                                     |
|[Dialog Act Classification](https://paperswithcode.com/task/dialog-act-classification)                        |对话框行为分类            |       1|0                                                                                                     |
|[Direction of Arrival Estimation](https://paperswithcode.com/task/direction-of-arrival-estimation)            |到达方向估计              |       1|0                                                                                                     |
|[Entity Alignment](https://paperswithcode.com/task/entity-alignment)                                          |实体对齐                  |       1|0                                                                                                     |
|[Fact Verification](https://paperswithcode.com/task/fact-verification)                                        |事实验证                  |       1|0                                                                                                     |
|[Humor Detection](https://paperswithcode.com/task/humor-detection)                                            |幽默检测                  |       1|0                                                                                                     |
|[Language Acquisition](https://paperswithcode.com/task/language-acquisition)                                  |语言习得                  |       1|0                                                                                                     |
|[Lexical Normalization](https://paperswithcode.com/task/lexical-normalization)                                |词法规范化                |       1|0                                                                                                     |
|[Memex Question Answering](https://paperswithcode.com/task/memex-question-answering)                          |梅克斯问题解答            |       1|0                                                                                                     |
|[Query Wellformedness](https://paperswithcode.com/task/query-wellformedness)                                  |格式良好的自然语言问题    |       1|0                                                                                                     |
|[Question Similarity](https://paperswithcode.com/task/question-similarity)                                    |问题相似性                |       1|[1](https://paperswithcode.com/area/natural-language-processing/question-similarity)                  |
|[Representation Learning](https://paperswithcode.com/task/representation-learning)                            |表征学习                  |       1|[11](https://paperswithcode.com/area/methodology/representation-learning)                             |
|[SQL-to-Text](https://paperswithcode.com/task/sql-to-text)                                                    |SQL转换文本               |       1|0                                                                                                     |
|[Speaker Diarization](https://paperswithcode.com/task/speaker-diarization)                                    |扬声器二元化              |       1|0                                                                                                     |
|[Speaker Identification](https://paperswithcode.com/task/speaker-identification)                              |扬声器识别                |       1|0                                                                                                     |
|[Subjectivity Analysis](https://paperswithcode.com/task/subjectivity-analysis)                                |主观性分析                |       1|0                                                                                                     |
|[Text Clustering](https://paperswithcode.com/task/text-clustering)                                            |文本群集                  |       1|[1](https://paperswithcode.com/area/natural-language-processing/text-clustering)                      |
|[Text-To-Sql](https://paperswithcode.com/task/text-to-sql)                                                    |文本到 Sql                |       1|0                                                                                                     |
|[Voice Conversion](https://paperswithcode.com/task/voice-conversion)                                          |语音转换                  |       1|0                                                                                                     |
|Abstract Anaphora Resolution                                                                                  |摘要 阿纳波拉分辨率       |       0|                                                                                                     0|
|Accented Speech Recognition                                                                                   |强调语音识别              |       0|                                                                                                     0|
|Acoustic Unit Discovery                                                                                       |声学单元发现              |       0|                                                                                                     0|
|Answer Selection                                                                                              |答案选择                  |       0|                                                                                                     0|
|Arrhythmia Detection                                                                                          |心律失常检测              |       0|                                                                                                     0|
|Aspect Extraction                                                                                             |方面提取                  |       0|                                                                                                     0|
|Audio Super-Resolution                                                                                        |音频超分辨率              |       0|                                                                                                     0|
|Citation Intent Classification                                                                                |引文意图分类              |       0|                                                                                                     0|
|Conversation                                                                                                  |对话                      |       0|                                                                                                     0|
|Cross-Lingual NER                                                                                             |跨语言 NER                |       0|                                                                                                     0|
|Cross-Lingual Natural Language Inference                                                                      |跨语言自然语言推理        |       0|                                                                                                     0|
|Curved Text Detection                                                                                         |曲线文本检测              |       0|                                                                                                     0|
|Dependency Grammar Induction                                                                                  |依赖语法归纳              |       0|                                                                                                     0|
|Dialogue Act Classification                                                                                   |对话法分类                |       0|                                                                                                     0|
|Dialogue Generation                                                                                           |对话生成                  |       0|                                                                                                     0|
|Dialogue State Tracking                                                                                       |对话状态跟踪              |       0|                                                                                                     0|
|Distant Speech Recognition                                                                                    |远程语音识别              |       0|                                                                                                     0|
|Document Embedding                                                                                            |文档嵌入                  |       0|                                                                                                     0|
|Drug–drug Interaction Extraction                                                                              |药物与药物相互作用提取    |       0|                                                                                                     0|
|Emotion Cause Extraction                                                                                      |情感原因提取              |       0|                                                                                                     0|
|Emotion Classification                                                                                        |情感分类                  |       0|                                                                                                     0|
|Emotion Recognition in Conversation                                                                           |对话中的情感识别          |       0|                                                                                                     0|
|Emotion-Cause Pair Extraction                                                                                 |情感原因对提取            |       0|                                                                                                     0|
|Event Detection                                                                                               |事件检测                  |       0|                                                                                                     0|
|Fine-Grained Opinion Analysis                                                                                 |细粒意见分析              |       0|                                                                                                     0|
|Grammatical Error Detection                                                                                   |语法错误检测              |       0|                                                                                                     0|
|Heartbeat Classification                                                                                      |心跳分类                  |       0|                                                                                                     0|
|KB-to-Language Generation                                                                                     |知识库到语言的生成        |       0|                                                                                                     0|
|Knowledge Base Question Answering                                                                             |知识库问题解答            |       0|                                                                                                     0|
|Language Modeling                                                                                             |语言模型                  |       0|                                                                                                     0|
|Multilingual Machine Comprehension                                                                            |多语种机器理解            |       0|                                                                                                     0|
|Name Entity Recognition                                                                                       |命名实体识别              |       0|                                                                                                     0|
|Negation Scope Resolution                                                                                     |否定范围解析              |       0|                                                                                                     0|
|Nested Named Entity Recognition                                                                               |嵌套命名实体识别          |       0|                                                                                                     0|
|Noisy Speech Recognition                                                                                      |嘈杂语音识别              |       0|                                                                                                     0|
|Paper generation                                                                                              |纸张生成                  |       0|                                                                                                     0|
|Paraphrase Generation                                                                                         |转述生成                  |       0|                                                                                                     0|
|Pos Tagging                                                                                                   |词性标记                  |       0|                                                                                                     0|
|Predicate Detection                                                                                           |谓词检测                  |       0|                                                                                                     0|
|Prosody Prediction                                                                                            |普罗索迪预测              |       0|                                                                                                     0|
|Query-Based Extractive Summarization                                                                          |基于查询的提取汇总        |       0|                                                                                                     0|
|Question Quality Assessment                                                                                   |问题质量评估              |       0|                                                                                                     0|
|Reader-Aware Summarization                                                                                    |读者感知汇总              |       0|                                                                                                     0|
|Scientific Concept Extraction                                                                                 |科学概念提取              |       0|                                                                                                     0|
|Scientific Results Extraction                                                                                 |科学结果提取              |       0|                                                                                                     0|
|Semantic Equivalence                                                                                          |语义等价                  |       0|                                                                                                     0|
|Semantic Role Labeling (predicted predicates)                                                                 |语义角色标记（预测谓词）  |       0|                                                                                                     0|
|Sentence Classification                                                                                       |句子分类                  |       0|                                                                                                     0|
|Speculation Scope Resolution                                                                                  |投机范围分辨率            |       0|                                                                                                     0|
|Speech Dereverberation                                                                                        |演讲德弗伯                |       0|                                                                                                     0|
|Speech Emotion Recognition                                                                                    |语音情感识别              |       0|                                                                                                     0|
|String Transformation                                                                                         |字符串转换                |       0|                                                                                                     0|
|Text Compression                                                                                              |文本压缩                  |       0|                                                                                                     0|
|Thai Word Tokenization                                                                                        |泰语单词标记              |       0|                                                                                                     0|
|Timeline Summarization                                                                                        |时间轴汇总                |       0|                                                                                                     0|
|Timex Normalization                                                                                           |时间表达式规范化          |       0|                                                                                                     0|
|Unsupervised Semantic Segmentation                                                                            |无监督语义分割            |       0|                                                                                                     0|
|Unsupervised Video Summarization                                                                              |无监督视频汇总            |       0|                                                                                                     0|
|Visual Dialog                                                                                                 |视觉对话框                |       0|                                                                                                     0|
|Word Segmentation                                                                                             |词分割                    |       0|                                                                                                     0|
|[Entity Resolution](https://paperswithcode.com/task/entity-resolution)                                        |实体解决方案              |       0|0                                                                                                     |
|[Relation Classification](https://paperswithcode.com/task/relation-classification)                            |关系分类                  |       0|[2](https://paperswithcode.com/area/natural-language-processing/relation-classification)              |
|[Sentence Embeddings](https://paperswithcode.com/task/sentence-embeddings)                                    |句子嵌入                  |       0|[4](https://paperswithcode.com/area/natural-language-processing/sentence-embeddings)                  |


## Computer Vision (CV) `计算机视觉`

|                                                             English                                                              |           Mandarin            |Datasets|                                       Subtasks                                       |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------|-------:|--------------------------------------------------------------------------------------|
|[Image Super-Resolution](https://paperswithcode.com/task/image-super-resolution)                                                  |超高分辨率成像                 |      47|[4](https://paperswithcode.com/area/audio/image-super-resolution)                     |
|[Graph Classification](https://paperswithcode.com/task/graph-classification)                                                      |图分类                         |      46|0                                                                                     |
|[Image Generation](https://paperswithcode.com/task/image-generation)                                                              |图像生成/图像合成              |      43|[13](https://paperswithcode.com/area/computer-vision/image-generation)                |
|[Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)                                                    |语义分割                       |      35|[18](https://paperswithcode.com/area/medical/semantic-segmentation)                   |
|[Image Classification](https://paperswithcode.com/task/image-classification)                                                      |图像分类                       |      34|[18](https://paperswithcode.com/area/computer-vision/image-classification)            |
|[Visual Question Answering](https://paperswithcode.com/task/visual-question-answering)                                            |视觉问题答案                   |      34|[2](https://paperswithcode.com/area/computer-vision/visual-question-answering)        |
|[3D Object Detection](https://paperswithcode.com/task/3d-object-detection)                                                        |3D 对象检测                    |      29|[2](https://paperswithcode.com/area/computer-vision/3d-object-detection)              |
|[Image Clustering](https://paperswithcode.com/task/image-clustering)                                                              |图像聚类                       |      27|[4](https://paperswithcode.com/area/computer-vision/image-clustering)                 |
|[Image-To-Image Translation](https://paperswithcode.com/task/image-to-image-translation)                                          |图像到图像的转换               |      27|[10](https://paperswithcode.com/area/computer-vision/image-to-image-translation)      |
|[Object Detection](https://paperswithcode.com/task/object-detection)                                                              |对象检测                       |      27|[23](https://paperswithcode.com/area/computer-vision/object-detection)                |
|[Action Recognition](https://paperswithcode.com/task/action-recognition)                                                          |活动识别                       |      22|[4](https://paperswithcode.com/area/computer-vision/action-recognition)               |
|[Image Retrieval](https://paperswithcode.com/task/image-retrieval)                                                                |图像检索系统                   |      20|[8](https://paperswithcode.com/area/computer-vision/image-retrieval)                  |
|[Fine-Grained Image Classification](https://paperswithcode.com/task/fine-grained-image-classification)                            |细粒度图像分类                 |      19|[1](https://paperswithcode.com/area/computer-vision/fine-grained-image-classification)|
|[Face Verification](https://paperswithcode.com/task/face-verification)                                                            |面部验证                       |      16|[1](https://paperswithcode.com/area/computer-vision/face-verification)                |
|[Hand Gesture Recognition](https://paperswithcode.com/task/hand-gesture-recognition)                                              |手势识别                       |      16|[1](https://paperswithcode.com/area/computer-vision/hand-gesture-recognition)         |
|[Facial Expression Recognition](https://paperswithcode.com/task/facial-expression-recognition)                                    |面部表情识别                   |      15|[4](https://paperswithcode.com/area/computer-vision/facial-expression-recognition)    |
|[Trajectory Prediction](https://paperswithcode.com/task/trajectory-prediction)                                                    |轨迹预测                       |      15|[2](https://paperswithcode.com/area/time-series/trajectory-prediction)                |
|[Object Localization](https://paperswithcode.com/task/object-localization)                                                        |目标定位任务                   |      14|[5](https://paperswithcode.com/area/computer-vision/object-localization)              |
|[Person Re-Identification](https://paperswithcode.com/task/person-re-identification)                                              |人物重新识别                   |      14|[7](https://paperswithcode.com/area/computer-vision/person-re-identification)         |
|[Person Re-identification](https://paperswithcode.com/task/person-re-identification)                                              |人员重新识别                   |      14|[7](https://paperswithcode.com/area/computer-vision/person-re-identification)         |
|[Density Estimation](https://paperswithcode.com/task/density-estimation)                                                          |密度估算                       |      13|0                                                                                     |
|[Pose Estimation](https://paperswithcode.com/task/pose-estimation)                                                                |姿势估计                       |      13|[14](https://paperswithcode.com/area/computer-vision/pose-estimation)                 |
|[Community Detection](https://paperswithcode.com/task/community-detection)                                                        |社区检测                       |      12|[3](https://paperswithcode.com/area/graphs/community-detection)                       |
|[Gesture Recognition](https://paperswithcode.com/task/gesture-recognition)                                                        |手势识别                       |      12|[4](https://paperswithcode.com/area/time-series/gesture-recognition)                  |
|[Medical Image Segmentation](https://paperswithcode.com/task/medical-image-segmentation)                                          |医学图像分割                   |      12|[28](https://paperswithcode.com/area/medical/medical-image-segmentation)              |
|[3D Human Pose Estimation](https://paperswithcode.com/task/3d-human-pose-estimation)                                              |3D 人类姿势估计                |      11|[4](https://paperswithcode.com/area/computer-vision/3d-human-pose-estimation)         |
|[Action Classification](https://paperswithcode.com/task/action-classification)                                                    |操作分类                       |      11|[1](https://paperswithcode.com/area/computer-vision/action-classification)            |
|[Interactive Segmentation](https://paperswithcode.com/task/interactive-segmentation)                                              |交互式分段                     |      10|0                                                                                     |
|[Anomaly Detection](https://paperswithcode.com/task/anomaly-detection)                                                            |异常检测                       |       9|[6](https://paperswithcode.com/area/computer-code/anomaly-detection)                  |
|[Scene Text Detection](https://paperswithcode.com/task/scene-text-detection)                                                      |场景文本检测                   |       9|[2](https://paperswithcode.com/area/computer-vision/scene-text-detection)             |
|[Optical Flow Estimation](https://paperswithcode.com/task/optical-flow-estimation)                                                |光流估计                       |       8|0                                                                                     |
|[Video Retrieval](https://paperswithcode.com/task/video-retrieval)                                                                |视频检索                       |       8|[1](https://paperswithcode.com/area/computer-vision/video-retrieval)                  |
|[Graph Regression](https://paperswithcode.com/task/graph-regression)                                                              |图形回归                       |       7|0                                                                                     |
|[Out-of-Distribution Detection](https://paperswithcode.com/task/out-of-distribution-detection)                                    |分布外检测                     |       7|0                                                                                     |
|[Talking Head Generation](https://paperswithcode.com/task/talking-head-generation)                                                |会说话的头一代                 |       7|[1](https://paperswithcode.com/area/computer-vision/talking-head-generation)          |
|[Temporal Action Localization](https://paperswithcode.com/task/temporal-action-localization)                                      |时态操作本地化                 |       7|[5](https://paperswithcode.com/area/computer-vision/temporal-action-localization)     |
|[Video Generation](https://paperswithcode.com/task/video-generation)                                                              |视频生成                       |       7|[1](https://paperswithcode.com/area/computer-vision/video-generation)                 |
|[Depth Estimation](https://paperswithcode.com/task/depth-estimation)                                                              |深度估计                       |       6|[7](https://paperswithcode.com/area/computer-vision/depth-estimation)                 |
|[Face Detection](https://paperswithcode.com/task/face-detection)                                                                  |人脸检测                       |       6|[1](https://paperswithcode.com/area/computer-vision/face-detection)                   |
|[Face Recognition](https://paperswithcode.com/task/face-recognition)                                                              |人脸识别                       |       6|[2](https://paperswithcode.com/area/computer-vision/face-recognition)                 |
|[Facial Landmark Detection](https://paperswithcode.com/task/facial-landmark-detection)                                            |面部地标检测                   |       6|[2](https://paperswithcode.com/area/computer-vision/facial-landmark-detection)        |
|[Hand Pose Estimation](https://paperswithcode.com/task/hand-pose-estimation)                                                      |手姿势估计                     |       6|[1](https://paperswithcode.com/area/computer-vision/hand-pose-estimation)             |
|[Human Interaction Recognition](https://paperswithcode.com/task/human-interaction-recognition)                                    |人际交往识别                   |       6|[2](https://paperswithcode.com/area/computer-vision/human-interaction-recognition)    |
|[Saliency Detection](https://paperswithcode.com/task/saliency-detection)                                                          |盐度检测                       |       6|[3](https://paperswithcode.com/area/computer-vision/saliency-detection)               |
|[Video Classification](https://paperswithcode.com/task/video-classification)                                                      |视频分类                       |       6|[1](https://paperswithcode.com/area/computer-vision/video-classification)             |
|[Video Prediction](https://paperswithcode.com/task/video-prediction)                                                              |视频预测                       |       6|[1](https://paperswithcode.com/area/computer-vision/video-prediction)                 |
|[3D Face Reconstruction](https://paperswithcode.com/task/3d-face-reconstruction)                                                  |3D 面部重建                    |       5|[1](https://paperswithcode.com/area/computer-vision/3d-face-reconstruction)           |
|[Compressive Sensing](https://paperswithcode.com/task/compressive-sensing)                                                        |压缩传感                       |       5|0                                                                                     |
|[Edge Detection](https://paperswithcode.com/task/edge-detection)                                                                  |边缘检测                       |       5|0                                                                                     |
|[Gaze Estimation](https://paperswithcode.com/task/gaze-estimation)                                                                |凝视估计                       |       5|0                                                                                     |
|[Image Dehazing](https://paperswithcode.com/task/image-dehazing)                                                                  |图像除哈辛                     |       5|0                                                                                     |
|[Image Matting](https://paperswithcode.com/task/image-matting)                                                                    |图像遮罩                       |       5|0                                                                                     |
|[Instance Segmentation](https://paperswithcode.com/task/instance-segmentation)                                                    |实例分割                       |       5|[5](https://paperswithcode.com/area/computer-vision/instance-segmentation)            |
|[Lipreading](https://paperswithcode.com/task/lipreading)                                                                          |唇读                           |       5|0                                                                                     |
|[Video Captioning](https://paperswithcode.com/task/video-captioning)                                                              |视频字幕                       |       5|[3](https://paperswithcode.com/area/computer-vision/video-captioning)                 |
|[3D Multi-Person Pose Estimation](https://paperswithcode.com/task/3d-multi-person-pose-estimation)                                |3D 多人姿势估计                |       4|[2](https://paperswithcode.com/area/computer-vision/3d-multi-person-pose-estimation)  |
|[3D Reconstruction](https://paperswithcode.com/task/3d-reconstruction)                                                            |3D 重建                        |       4|[1](https://paperswithcode.com/area/computer-vision/3d-reconstruction)                |
|[Action Quality Assessment](https://paperswithcode.com/task/action-quality-assessment)                                            |行动质量评估                   |       4|0                                                                                     |
|[Depth Completion](https://paperswithcode.com/task/depth-completion)                                                              |深度完成                       |       4|0                                                                                     |
|[Document Image Classification](https://paperswithcode.com/task/document-image-classification)                                    |文档图像分类                   |       4|0                                                                                     |
|[Horizon Line Estimation](https://paperswithcode.com/task/horizon-line-estimation)                                                |地平线线估计                   |       4|0                                                                                     |
|[Image Captioning](https://paperswithcode.com/task/image-captioning)                                                              |图像字幕                       |       4|[1](https://paperswithcode.com/area/computer-vision/image-captioning)                 |
|[Image Inpainting](https://paperswithcode.com/task/image-inpainting)                                                              |图像画                         |       4|[4](https://paperswithcode.com/area/computer-vision/image-inpainting)                 |
|[Motion Segmentation](https://paperswithcode.com/task/motion-segmentation)                                                        |运动分割                       |       4|0                                                                                     |
|[Novel View Synthesis](https://paperswithcode.com/task/novel-view-synthesis)                                                      |新视图合成                     |       4|0                                                                                     |
|[Surface Normals Estimation](https://paperswithcode.com/task/surface-normals-estimation)                                          |曲面法线估计                   |       4|0                                                                                     |
|[Weakly Supervised Action Localization](https://paperswithcode.com/task/weakly-supervised-action-localization)                    |弱监督的操作本地化             |       4|0                                                                                     |
|[3D Canonical Hand Pose Estimation](https://paperswithcode.com/task/3d-canonical-hand-pose-estimation)                            |3D 规范手姿势估计              |       3|0                                                                                     |
|[3D Point Cloud Classification](https://paperswithcode.com/task/3d-point-cloud-classification)                                    |3D 点云分类                    |       3|0                                                                                     |
|[6D Pose Estimation](https://paperswithcode.com/task/6d-pose-estimation)                                                          |6D 姿势估计                    |       3|[1](https://paperswithcode.com/area/computer-vision/6d-pose-estimation)               |
|[Blink Estimation](https://paperswithcode.com/task/blink-estimation)                                                              |闪烁估计                       |       3|0                                                                                     |
|[Graph Clustering](https://paperswithcode.com/task/graph-clustering)                                                              |图形聚类                       |       3|0                                                                                     |
|[Human Part Segmentation](https://paperswithcode.com/task/human-part-segmentation)                                                |人体部分分割                   |       3|0                                                                                     |
|[Human-Object Interaction Detection](https://paperswithcode.com/task/human-object-interaction-detection)                          |人-对象交互检测                |       3|0                                                                                     |
|[Image Reconstruction](https://paperswithcode.com/task/image-reconstruction)                                                      |图像重建                       |       3|[1](https://paperswithcode.com/area/computer-vision/image-reconstruction)             |
|[Object Classification](https://paperswithcode.com/task/object-classification)                                                    |对象分类                       |       3|0                                                                                     |
|[Object Counting](https://paperswithcode.com/task/object-counting)                                                                |对象计数                       |       3|0                                                                                     |
|[Point Cloud Generation](https://paperswithcode.com/task/point-cloud-generation)                                                  |点云生成                       |       3|[1](https://paperswithcode.com/area/computer-vision/point-cloud-generation)           |
|[Pose Tracking](https://paperswithcode.com/task/pose-tracking)                                                                    |姿势跟踪                       |       3|0                                                                                     |
|[Sketch-Based Image Retrieval](https://paperswithcode.com/task/sketch-based-image-retrieval)                                      |基于草图的图像检索             |       3|[1](https://paperswithcode.com/area/computer-vision/sketch-based-image-retrieval)     |
|[Stance Detection](https://paperswithcode.com/task/stance-detection)                                                              |姿态检测                       |       3|0                                                                                     |
|[Video Object Segmentation](https://paperswithcode.com/task/video-object-segmentation)                                            |视频对象分割                   |       3|[4](https://paperswithcode.com/area/computer-vision/video-object-segmentation)        |
|[3D Absolute Human Pose Estimation](https://paperswithcode.com/task/3d-absolute-human-pose-estimation)                            |3D 绝对人类姿势估计            |       2|[1](https://paperswithcode.com/area/medical/3d-absolute-human-pose-estimation)        |
|[3D Object Classification](https://paperswithcode.com/task/3d-object-classification)                                              |3D 对象分类                    |       2|[1](https://paperswithcode.com/area/computer-vision/3d-object-classification)         |
|[3D Object Reconstruction](https://paperswithcode.com/task/3d-object-reconstruction)                                              |3D 对象重建                    |       2|[2](https://paperswithcode.com/area/computer-vision/3d-object-reconstruction)         |
|[3D Semantic Segmentation](https://paperswithcode.com/task/3d-semantic-segmentation)                                              |3D 语义分割                    |       2|[2](https://paperswithcode.com/area/computer-vision/3d-semantic-segmentation)         |
|[3D Shape Modeling](https://paperswithcode.com/task/3d-shape-modeling)                                                            |3D 形状建模                    |       2|0                                                                                     |
|[Emotion Recognition](https://paperswithcode.com/task/emotion-recognition)                                                        |情感识别                       |       2|[7](https://paperswithcode.com/area/natural-language-processing/emotion-recognition)  |
|[Human Action Generation](https://paperswithcode.com/task/human-action-generation)                                                |人类行动生成                   |       2|[1](https://paperswithcode.com/area/computer-vision/human-action-generation)          |
|[Line Segment Detection](https://paperswithcode.com/task/line-segment-detection)                                                  |线段检测                       |       2|0                                                                                     |
|[Pneumonia Detection](https://paperswithcode.com/task/pneumonia-detection)                                                        |肺炎检测                       |       2|0                                                                                     |
|[Pose Retrieval](https://paperswithcode.com/task/pose-retrieval)                                                                  |姿势检索                       |       2|0                                                                                     |
|[Scanpath Prediction](https://paperswithcode.com/task/scanpath-prediction)                                                        |扫描路径预测                   |       2|0                                                                                     |
|[Sign Language Recognition](https://paperswithcode.com/task/sign-language-recognition)                                            |手语识别                       |       2|0                                                                                     |
|[Video Summarization](https://paperswithcode.com/task/video-summarization)                                                        |视频总结                       |       2|[2](https://paperswithcode.com/area/computer-vision/video-summarization)              |
|[3D Hand Pose Estimation](https://paperswithcode.com/task/3d-hand-pose-estimation)                                                |3D 手姿势估计                  |       1|[2](https://paperswithcode.com/area/computer-vision/3d-hand-pose-estimation)          |
|[3D Multi-person Pose Estimation (absolute)](https://paperswithcode.com/task/3d-multi-person-pose-estimation-(absolute))          |3D 多人姿势估计（绝对）        |       1|0                                                                                     |
|[3D Multi-person Pose Estimation (root-relative)](https://paperswithcode.com/task/3d-multi-person-pose-estimation-(root-relative))|3D 多人姿势估计（相对根）      |       1|0                                                                                     |
|[3D Shape Reconstruction](https://paperswithcode.com/task/3d-shape-reconstruction)                                                |3D 形状重建                    |       1|[1](https://paperswithcode.com/area/computer-vision/3d-shape-reconstruction)          |
|[Activity Prediction](https://paperswithcode.com/task/activity-prediction)                                                        |活动预测                       |       1|[2](https://paperswithcode.com/area/time-series/activity-prediction)                  |
|[Activity Recognition In Videos](https://paperswithcode.com/task/activity-recognition-in-videos)                                  |视频中的活动识别               |       1|[1](https://paperswithcode.com/area/computer-vision/activity-recognition-in-videos)   |
|[Animal Pose Estimation](https://paperswithcode.com/task/animal-pose-estimation)                                                  |动物姿势估计                   |       1|0                                                                                     |
|[Breast Tumour Classification](https://paperswithcode.com/task/breast-tumour-classification)                                      |乳房肿瘤分类                   |       1|0                                                                                     |
|[Camera Localization](https://paperswithcode.com/task/camera-localization)                                                        |摄像机本地化                   |       1|[1](https://paperswithcode.com/area/computer-vision/camera-localization)              |
|[DeepFake Detection](https://paperswithcode.com/task/deepfake-detection)                                                          |深度故障检测                   |       1|0                                                                                     |
|[Defocus Estimation](https://paperswithcode.com/task/defocus-estimation)                                                          |散焦估计                       |       1|0                                                                                     |
|[Dense Pixel Correspondence Estimation](https://paperswithcode.com/task/dense-pixel-correspondence-estimation)                    |寻找密集像素图像之间的对应关系 |       1|0                                                                                     |
|[Document Layout Analysis](https://paperswithcode.com/task/document-layout-analysis)                                              |文档布局分析                   |       1|[1](https://paperswithcode.com/area/computer-vision/document-layout-analysis)         |
|[Historical Color Image Dating](https://paperswithcode.com/task/historical-color-image-dating)                                    |历史彩色图像约会               |       1|0                                                                                     |
|[Image Compression](https://paperswithcode.com/task/image-compression)                                                            |图像压缩                       |       1|[3](https://paperswithcode.com/area/computer-vision/image-compression)                |
|[Image Cropping](https://paperswithcode.com/task/image-cropping)                                                                  |图像裁剪                       |       1|0                                                                                     |
|[Image Enhancement](https://paperswithcode.com/task/image-enhancement)                                                            |图像增强                       |       1|[3](https://paperswithcode.com/area/computer-vision/image-enhancement)                |
|[Image Recognition](https://paperswithcode.com/task/image-recognition)                                                            |图像识别                       |       1|[2](https://paperswithcode.com/area/computer-vision/image-recognition)                |
|[Image Registration](https://paperswithcode.com/task/image-registration)                                                          |图像注册                       |       1|0                                                                                     |
|[Lip to Speech Synthesis](https://paperswithcode.com/task/lip-to-speech-synthesis)                                                |唇到语合成                     |       1|[1](https://paperswithcode.com/area/computer-vision/lip-to-speech-synthesis)          |
|[Multi-tissue Nucleus Segmentation](https://paperswithcode.com/task/multi-tissue-nucleus-segmentation)                            |多组织核分割                   |       1|0                                                                                     |
|[Optical Character Recognition](https://paperswithcode.com/task/optical-character-recognition)                                    |光学字符识别                   |       1|[7](https://paperswithcode.com/area/computer-vision/optical-character-recognition)    |
|[Person Identification](https://paperswithcode.com/task/person-identification)                                                    |人员识别                       |       1|0                                                                                     |
|[Person Retrieval](https://paperswithcode.com/task/person-retrieval)                                                              |人员检索                       |       1|0                                                                                     |
|[Point Cloud Super Resolution](https://paperswithcode.com/task/point-cloud-super-resolution)                                      |点云超分辨率                   |       1|0                                                                                     |
|[Rain Removal](https://paperswithcode.com/task/rain-removal)                                                                      |清除雨水                       |       1|[1](https://paperswithcode.com/area/computer-vision/rain-removal)                     |
|[Robotic Grasping](https://paperswithcode.com/task/robotic-grasping)                                                              |机器人抓握                     |       1|[1](https://paperswithcode.com/area/robots/robotic-grasping)                          |
|[Rotated MNIST](https://paperswithcode.com/task/rotated-mnist)                                                                    |旋转的 MNIST                   |       1|0                                                                                     |
|[Safety Perception Recognition](https://paperswithcode.com/task/safety-perception-recognition)                                    |安全感知辨识                   |       1|0                                                                                     |
|[Scene Parsing](https://paperswithcode.com/task/scene-parsing)                                                                    |场景分析                       |       1|[8](https://paperswithcode.com/area/computer-vision/scene-parsing)                    |
|[Scene Understanding](https://paperswithcode.com/task/scene-understanding)                                                        |场景了解                       |       1|[2](https://paperswithcode.com/area/computer-vision/scene-understanding)              |
|[Seizure Detection](https://paperswithcode.com/task/seizure-detection)                                                            |缉获检测                       |       1|0                                                                                     |
|[Shadow Detection](https://paperswithcode.com/task/shadow-detection)                                                              |阴影检测                       |       1|[1](https://paperswithcode.com/area/computer-vision/shadow-detection)                 |
|[Table Detection](https://paperswithcode.com/task/table-detection)                                                                |表检测                         |       1|0                                                                                     |
|[Talking Face Generation](https://paperswithcode.com/task/talking-face-generation)                                                |会说话的脸生成                 |       1|[1](https://paperswithcode.com/area/computer-vision/talking-face-generation)          |
|[Traffic Sign Detection](https://paperswithcode.com/task/traffic-sign-detection)                                                  |交通标志检测                   |       1|0                                                                                     |
|[Video Reconstruction](https://paperswithcode.com/task/video-reconstruction)                                                      |视频重建                       |       1|0                                                                                     |
|[Vision-Language Navigation](https://paperswithcode.com/task/vision-language-navigation)                                          |视觉语言导航                   |       1|0                                                                                     |
|3D Facial Expression Recognition                                                                                                  |3D 面部表情识别                |       0|                                                                                     0|
|3D Facial Landmark Localization                                                                                                   |3D 面部地标本地化              |       0|                                                                                     0|
|3D Instance Segmentation                                                                                                          |3D 实例分割                    |       0|                                                                                     0|
|3D Medical Imaging Segmentation                                                                                                   |3D 医疗成像分割                |       0|                                                                                     0|
|3D Multi-Object Tracking                                                                                                          |3D 多对象跟踪                  |       0|                                                                                     0|
|3D Object Recognition                                                                                                             |3D 对象识别                    |       0|                                                                                     0|
|3D Object Reconstruction From A Single Image                                                                                      |从单个图像重建 3D 对象         |       0|                                                                                     0|
|3D Part Segmentation                                                                                                              |3D 零件分割                    |       0|                                                                                     0|
|3D Pose Estimation                                                                                                                |3D 姿势估计                    |       0|                                                                                     0|
|3D Room Layouts From A Single RGB Panorama                                                                                        |来自单个 RGB 全景的 3D 房间布局|       0|                                                                                     0|
|3D Room Layouts From A Single Rgb Panorama                                                                                        |来自单个 Rgb 全景的 3D 房间布局|       0|                                                                                     0|
|3D Semantic Instance Segmentation                                                                                                 |3D 语义实例分割                |       0|                                                                                     0|
|3D Shape Classification                                                                                                           |3D 形状分类                    |       0|                                                                                     0|
|3D Shape Retrieval                                                                                                                |3D 形状检索                    |       0|                                                                                     0|
|6D Pose Estimation using RGB                                                                                                      |使用 RGB 进行 6D 姿势估计      |       0|                                                                                     0|
|Abnormal Event Detection In Video                                                                                                 |视频中异常事件检测             |       0|                                                                                     0|
|Action Recognition In Videos                                                                                                      |视频中的操作识别               |       0|                                                                                     0|
|Action Segmentation                                                                                                               |操作分割                       |       0|                                                                                     0|
|Action Unit Detection                                                                                                             |操作单元检测                   |       0|                                                                                     0|
|Aesthetics Quality Assessment                                                                                                     |美学质量评估                   |       0|                                                                                     0|
|Age And Gender Classification                                                                                                     |年龄和性别分类                 |       0|                                                                                     0|
|Age Estimation                                                                                                                    |年龄估算                       |       0|                                                                                     0|
|Age-Invariant Face Recognition                                                                                                    |年龄不变人脸识别               |       0|                                                                                     0|
|Anomaly Detection in Edge Streams                                                                                                 |边缘流中的异常检测             |       0|                                                                                     0|
|Blood pressure estimation                                                                                                         |血压估计                       |       0|                                                                                     0|
|Brain Image Segmentation                                                                                                          |脑图像分割                     |       0|                                                                                     0|
|Building Change Detection                                                                                                         |生成更改检测                   |       0|                                                                                     0|
|Cell Segmentation                                                                                                                 |细胞分割                       |       0|                                                                                     0|
|Classification Consistency                                                                                                        |分类一致性                     |       0|                                                                                     0|
|Colorectal Gland Segmentation                                                                                                     |结肠直肠分段                   |       0|                                                                                     0|
|Cross-View Image-to-Image Translation                                                                                             |交叉查看图像到图像的转换       |       0|                                                                                     0|
|Crowd Counting                                                                                                                    |人群计数                       |       0|                                                                                     0|
|Dense Object Detection                                                                                                            |密集对象检测                   |       0|                                                                                     0|
|Depiction Invariant Object Recognition                                                                                            |描述不变对象识别               |       0|                                                                                     0|
|Disease Detection                                                                                                                 |疾病检测                       |       0|                                                                                     0|
|Disguised Face Verification                                                                                                       |伪装面部验证                   |       0|                                                                                     0|
|Displaced people recognition                                                                                                      |流离失所者得到承认             |       0|                                                                                     0|
|Document Image Dewarping                                                                                                          |文档图像去扭曲                 |       0|                                                                                     0|
|Document Image Unwarping                                                                                                          |文档图像解扭曲                 |       0|                                                                                     0|
|Drone Navigation                                                                                                                  |无人机导航                     |       0|                                                                                     0|
|Drone-view Target Localization                                                                                                    |无人机视图目标本地化           |       0|                                                                                     0|
|Egocentric Activity Recognition                                                                                                   |以自我为中心的活动识别         |       0|                                                                                     0|
|Face Alignment                                                                                                                    |面对齐                         |       0|                                                                                     0|
|Face Anti-Spoofing                                                                                                                |面防欺骗                       |       0|                                                                                     0|
|Face Hallucination                                                                                                                |面部幻觉                       |       0|                                                                                     0|
|Face Presentation Attack Detection                                                                                                |人脸演示攻击检测               |       0|                                                                                     0|
|Face Sketch Synthesis                                                                                                             |面部草图合成                   |       0|                                                                                     0|
|Facial Action Unit Detection                                                                                                      |面部动作单元检测               |       0|                                                                                     0|
|Facial Beauty Prediction                                                                                                          |面部美容预测                   |       0|                                                                                     0|
|Facial Inpainting                                                                                                                 |面部修复/面部完成              |       0|                                                                                     0|
|Factual Visual Question Answering                                                                                                 |事实视觉问题解答               |       0|                                                                                     0|
|Gender Prediction                                                                                                                 |性别预测                       |       0|                                                                                     0|
|Generating 3D Point Clouds                                                                                                        |生成 3D 点云                   |       0|                                                                                     0|
|Group Activity Recognition                                                                                                        |组活动识别                     |       0|                                                                                     0|
|Hand Gesture-to-Gesture Translation                                                                                               |手手势到手势翻译               |       0|                                                                                     0|
|Handwritten Digit Recognition                                                                                                     |手写数字识别                   |       0|                                                                                     0|
|Head Detection                                                                                                                    |头部检测                       |       0|                                                                                     0|
|Head Pose Estimation                                                                                                              |头部姿势估计                   |       0|                                                                                     0|
|Homography Estimation                                                                                                             |造影估计                       |       0|                                                                                     0|
|Human Grasp Contact Prediction                                                                                                    |人类抓地力接触预测             |       0|                                                                                     0|
|Human Instance Segmentation                                                                                                       |人类实例分割                   |       0|                                                                                     0|
|Human Pose Estimation                                                                                                             |人体姿态估计                   |       0|                                                                                     0|
|Hyperspectral Image Classification                                                                                                |超光谱图像分类                 |       0|                                                                                     0|
|Image Denoising                                                                                                                   |图像去噪                       |       0|                                                                                     0|
|Image Outpainting                                                                                                                 |图像外漆                       |       0|                                                                                     0|
|Image Segmentation                                                                                                                |图像分割                       |       0|                                                                                     0|
|Interactive Video Object Segmentation                                                                                             |交互式视频对象分割             |       0|                                                                                     0|
|Keypoint Detection                                                                                                                |关键点检测                     |       0|                                                                                     0|
|Layout-to-Image Generation                                                                                                        |布局到图像的生成               |       0|                                                                                     0|
|Lesion Segmentation                                                                                                               |病变分割                       |       0|                                                                                     0|
|License Plate Recognition                                                                                                         |牌照识别                       |       0|                                                                                     0|
|Liver Segmentation                                                                                                                |肝脏分割                       |       0|                                                                                     0|
|Low-Light Image Enhancement                                                                                                       |低光图像增强                   |       0|                                                                                     0|
|Lung Nodule Segmentation                                                                                                          |肺结核分割                     |       0|                                                                                     0|
|Medical Object Detection                                                                                                          |医疗对象检测                   |       0|                                                                                     0|
|Monocular Depth Estimation                                                                                                        |单目深度估计                   |       0|                                                                                     0|
|Multi-Human Parsing                                                                                                               |多人解析                       |       0|                                                                                     0|
|Multi-Person Pose Estimation                                                                                                      |多人姿势估计                   |       0|                                                                                     0|
|Multi-view Subspace Clustering                                                                                                    |多视图子空间群集               |       0|                                                                                     0|
|Multimodal Unsupervised Image-To-Image Translation                                                                                |多式联运无监督图像到图像转换   |       0|                                                                                     0|
|Multiple Object Forecasting                                                                                                       |多个对象预测                   |       0|                                                                                     0|
|Multiple Object Tracking                                                                                                          |多个对象跟踪                   |       0|                                                                                     0|
|Nuclear Segmentation                                                                                                              |核分段                         |       0|                                                                                     0|
|Object Proposal Generation                                                                                                        |对象建议生成                   |       0|                                                                                     0|
|Object Skeleton Detection                                                                                                         |对象骨架检测                   |       0|                                                                                     0|
|Occluded Face Detection                                                                                                           |遮挡人脸检测                   |       0|                                                                                     0|
|One-Shot 3D Action Recognition                                                                                                    |单次三拍动作识别               |       0|                                                                                     0|
|One-Shot Object Detection                                                                                                         |单次对象检测                   |       0|                                                                                     0|
|Online Multi-Object Tracking                                                                                                      |在线多对象跟踪                 |       0|                                                                                     0|
|Pancreas Segmentation                                                                                                             |胰腺分割                       |       0|                                                                                     0|
|Panoptic Segmentation                                                                                                             |泛视分割                       |       0|                                                                                     0|
|People Recognition                                                                                                                |人员认可                       |       0|                                                                                     0|
|Photo Geolocation Estimation                                                                                                      |照片地理位置估算               |       0|                                                                                     0|
|Pixel-By-Pixel Classification                                                                                                     |在像素层面上进行分类           |       0|                                                                                     0|
|Point Cloud Completion                                                                                                            |点云完成                       |       0|                                                                                     0|
|Pose Transfer                                                                                                                     |姿势转移                       |       0|                                                                                     0|
|RF-based Pose Estimation                                                                                                          |基于射频的姿势估计             |       0|                                                                                     0|
|Real-Time Object Detection                                                                                                        |实时对象检测                   |       0|                                                                                     0|
|Real-time Instance Segmentation                                                                                                   |实时实例细分                   |       0|                                                                                     0|
|Retinal Vessel Segmentation                                                                                                       |视网膜血管分割                 |       0|                                                                                     0|
|Robust Object Detection                                                                                                           |强大的对象检测                 |       0|                                                                                     0|
|Scene Graph Generation                                                                                                            |场景图形生成                   |       0|                                                                                     0|
|Scene Recognition                                                                                                                 |场景识别                       |       0|                                                                                     0|
|Scene Segmentation                                                                                                                |场景分割                       |       0|                                                                                     0|
|Self-Supervised Action Recognition                                                                                                |自我监督的操作识别             |       0|                                                                                     0|
|Semantic Part Detection                                                                                                           |语义零件检测                   |       0|                                                                                     0|
|Semi-Supervised Image Classification                                                                                              |半监督图像分类                 |       0|                                                                                     0|
|Sequential Image Classification                                                                                                   |顺序图像分类                   |       0|                                                                                     0|
|Single-Image-Based Hdr Reconstruction                                                                                             |基于单图像的 Hdr 重建          |       0|                                                                                     0|
|Skeleton Based Action Recognition                                                                                                 |基于骨架的操作识别             |       0|                                                                                     0|
|Skin Cancer Segmentation                                                                                                          |皮肤癌分割                     |       0|                                                                                     0|
|Smile Recognition                                                                                                                 |微笑识别                       |       0|                                                                                     0|
|Speaker-Specific Lip to Speech Synthesis                                                                                          |扬声器特定唇部语音合成         |       0|                                                                                     0|
|Steering Control                                                                                                                  |转向控制                       |       0|                                                                                     0|
|Stereo Depth Estimation                                                                                                           |立体深度估计                   |       0|                                                                                     0|
|Synthetic-to-Real Translation                                                                                                     |合成到真实翻译                 |       0|                                                                                     0|
|Temporal Action Proposal Generation                                                                                               |临时操作建议生成               |       0|                                                                                     0|
|Text-To-Image Generation                                                                                                          |文本到图像生成                 |       0|                                                                                     0|
|Text-to-Image Generation                                                                                                          |文本到图像的生成               |       0|                                                                                     0|
|Trajectory Forecasting                                                                                                            |轨迹预测                       |       0|                                                                                     0|
|Unsupervised Facial Landmark Detection                                                                                            |无监督的面部地标检测           |       0|                                                                                     0|
|Unsupervised Person Re-Identification                                                                                             |无监督的人重新识别             |       0|                                                                                     0|
|Unsupervised image classification                                                                                                 |无监督图像分类                 |       0|                                                                                     0|
|Vector Graphics Animation                                                                                                         |矢量图形动画                   |       0|                                                                                     0|
|Vehicle Re-Identification                                                                                                         |车辆重新识别                   |       0|                                                                                     0|
|Video Deinterlacing                                                                                                               |视频去隔行扫描                 |       0|                                                                                     0|
|Video Denoising                                                                                                                   |视频去诺化                     |       0|                                                                                     0|
|Video Frame Interpolation                                                                                                         |视频帧插值                     |       0|                                                                                     0|
|Video Image Super-Resolution                                                                                                      |视频超分辨率                   |       0|                                                                                     0|
|Video Object Detection                                                                                                            |视频对象检测                   |       0|                                                                                     0|
|Video Story QA                                                                                                                    |视频故事 QA                    |       0|                                                                                     0|
|Video-to-Video Synthesis                                                                                                          |视频到视频合成                 |       0|                                                                                     0|
|Visual Object Tracking                                                                                                            |视觉对象跟踪                   |       0|                                                                                     0|
|Visual Storytelling                                                                                                               |视觉讲故事                     |       0|                                                                                     0|
|Volumetric Medical Image Segmentation                                                                                             |体积医学图像分割               |       0|                                                                                     0|
|Weakly Supervised Object Detection                                                                                                |弱监督对象检测                 |       0|                                                                                     0|
|Weakly-Supervised Semantic Segmentation                                                                                           |弱监督语义分割                 |       0|                                                                                     0|
|Weakly-supervised Temporal Action Localization                                                                                    |监督不力的时态操作本地化       |       0|                                                                                     0|
|Zero-Shot Action Recognition                                                                                                      |零射击动作识别                 |       0|                                                                                     0|
|Zero-Shot Object Detection                                                                                                        |零射对象检测                   |       0|                                                                                     0|
|[Face Generation](https://paperswithcode.com/task/face-generation)                                                                |人脸生成                       |       0|[3](https://paperswithcode.com/area/computer-vision/face-generation)                  |
|[Graph Generation](https://paperswithcode.com/task/graph-generation)                                                              |图表生成                       |       0|                                                                                     0|
|[Image Matching](https://paperswithcode.com/task/image-matching)                                                                  |图像匹配                       |       0|[2](https://paperswithcode.com/area/computer-vision/image-matching)                   |
|[Object Segmentation](https://paperswithcode.com/task/object-segmentation)                                                        |对象分割                       |       0|[1](https://paperswithcode.com/area/computer-vision/object-segmentation)              |
|[Visual Tracking](https://paperswithcode.com/task/visual-tracking)                                                                |视觉跟踪                       |       0|[3](https://paperswithcode.com/area/computer-vision/visual-tracking)                  |


## Robotics `机器人技术`

|                               English                                | Mandarin |Datasets|                          Subtasks                          |
|----------------------------------------------------------------------|----------|-------:|------------------------------------------------------------|
|[Visual Navigation](https://paperswithcode.com/task/visual-navigation)|视觉导航  |       4|                                                           0|
|[Robotic Grasping](https://paperswithcode.com/task/robotic-grasping)  |机器人抓握|       1|[1](https://paperswithcode.com/area/robots/robotic-grasping)|
|PointGoal Navigation                                                  |点目标导航|       0|                                                           0|
|[Visual Odometry](https://paperswithcode.com/task/visual-odometry)    |视觉测光仪|       0|[1](https://paperswithcode.com/area/robots/visual-odometry) |


## Basic Theory `基础理论`

|                                       English                                        |        Mandarin         |Datasets|                               Subtasks                                |
|--------------------------------------------------------------------------------------|-------------------------|-------:|-----------------------------------------------------------------------|
|[Node Classification](https://paperswithcode.com/task/node-classification)            |节点分类                 |      60|[1](https://paperswithcode.com/area/graphs/node-classification)        |
|[Continuous Control](https://paperswithcode.com/task/continuous-control)              |连续控制代               |      49|[2](https://paperswithcode.com/area/computer-vision/continuous-control)|
|[Quantization](https://paperswithcode.com/task/quantization)                          |量化                     |       2|[1](https://paperswithcode.com/area/computer-vision/quantization)      |
|[Multi-Armed Bandits](https://paperswithcode.com/task/multi-armed-bandits)            |多武装强盗               |       1|0                                                                      |
|[Pain Intensity Regression](https://paperswithcode.com/task/pain-intensity-regression)|疼痛强度回归预测         |       1|0                                                                      |
|Arithmetic                                                                            |算术                     |       0|                                                                      0|
|Physical Object Perception                                                            |物理对象感知/物理对象识别|       0|                                                                      0|


## Optimization `运筹优化`

|                                        English                                         |      Mandarin       |Datasets|                              Subtasks                              |
|----------------------------------------------------------------------------------------|---------------------|-------:|--------------------------------------------------------------------|
|[Atari Games](https://paperswithcode.com/task/atari-games)                              |雅达利电子游戏       |      62|[1](https://paperswithcode.com/area/playing-games/atari-games)      |
|[Neural Architecture Search](https://paperswithcode.com/task/neural-architecture-search)|神经结构搜索         |      13|0                                                                   |
|[Automated Theorem Proving](https://paperswithcode.com/task/automated-theorem-proving)  |自动定理证明         |       6|0                                                                   |
|[Network Pruning](https://paperswithcode.com/task/network-pruning)                      |网络修剪             |       4|0                                                                   |
|[Program Synthesis](https://paperswithcode.com/task/program-synthesis)                  |程序合成/自动构建程序|       2|[3](https://paperswithcode.com/area/computer-code/program-synthesis)|
|[Game of Go](https://paperswithcode.com/task/game-of-go)                                |围棋                 |       1|0                                                                   |
|[Game of Shogi](https://paperswithcode.com/task/game-of-shogi)                          |将棋                 |       1|0                                                                   |
|[Game of Suduko](https://paperswithcode.com/task/game-of-suduko)                        |苏杜科游戏           |       1|0                                                                   |
|Abstract-Strategy Game                                                                  |抽象策略游戏         |       0|                                                                   0|
|Game of Doom                                                                            |末日游戏             |       0|                                                                   0|
|NetHack Game                                                                            |网哈克游戏           |       0|                                                                   0|
|SNES Games                                                                              |斯内斯游戏           |       0|                                                                   0|
|Starcraft II                                                                            |星际争霸II           |       0|                                                                   0|


## Causal Inference `因果推理`

|                                    English                                     |  Mandarin  |Datasets|                             Subtasks                              |
|--------------------------------------------------------------------------------|------------|-------:|-------------------------------------------------------------------|
|[Common Sense Reasoning](https://paperswithcode.com/task/common-sense-reasoning)|常识推理    |      12|                                                                  0|
|[Visual Reasoning](https://paperswithcode.com/task/visual-reasoning)            |视觉推理    |       3|[1](https://paperswithcode.com/area/reasoning/visual-reasoning)    |
|[Causal Inference](https://paperswithcode.com/task/causal-inference)            |因果推论    |       1|[1](https://paperswithcode.com/area/miscellaneous/causal-inference)|
|[Relational Reasoning](https://paperswithcode.com/task/relational-reasoning)    |关系推理    |       1|0                                                                  |
|Commonsense Inference                                                           |常识推理    |       0|                                                                  0|
|Constituency Grammar Induction                                                  |选区语法归纳|       0|                                                                  0|


## Knowledge Engineering `知识工程`

|                                         English                                          |     Mandarin     |Datasets|                                   Subtasks                                   |
|------------------------------------------------------------------------------------------|------------------|-------:|------------------------------------------------------------------------------|
|[Knowledge Graph Completion](https://paperswithcode.com/task/knowledge-graph-completion)  |知识图完成        |       4|[1](https://paperswithcode.com/area/knowledge-base/knowledge-graph-completion)|
|[Entity Alignment](https://paperswithcode.com/task/entity-alignment)                      |实体对齐          |       1|0                                                                             |
|[Medical Relation Extraction](https://paperswithcode.com/task/medical-relation-extraction)|医疗关系提取      |       1|0                                                                             |
|Joint Entity and Relation Extraction                                                      |联合实体和关系提取|       0|                                                                             0|
|Knowledge Graph Embeddings                                                                |知识图嵌入        |       0|                                                                             0|
|Open Knowledge Graph Canonicalization                                                     |开放式知识图规范化|       0|                                                                             0|


## Data Science `数据科学`

|                                           English                                            |     Mandarin     |Datasets|                                  Subtasks                                  |
|----------------------------------------------------------------------------------------------|------------------|-------:|----------------------------------------------------------------------------|
|[Time Series Classification](https://paperswithcode.com/task/time-series-classification)      |时序列分类        |      24|                                                                           0|
|[Click-Through Rate Prediction](https://paperswithcode.com/task/click-through-rate-prediction)|点击率预测        |      17|                                                                           0|
|[Drug Discovery](https://paperswithcode.com/task/drug-discovery)                              |药物发现          |      14|                                                                           0|
|[Outlier Detection](https://paperswithcode.com/task/outlier-detection)                        |异常值检测        |      10|[2](https://paperswithcode.com/area/methodology/outlier-detection)          |
|[Traffic Prediction](https://paperswithcode.com/task/traffic-prediction)                      |交通流量预测      |       4|[1](https://paperswithcode.com/area/time-series/traffic-prediction)         |
|[ECG Classification](https://paperswithcode.com/task/ecg-classification)                      |ECG 分类          |       1|[1](https://paperswithcode.com/area/medical/ecg-classification)             |
|[Length-of-Stay prediction](https://paperswithcode.com/task/length-of-stay-prediction)        |停留时间预测      |       1|                                                                           0|
|[Mortality Prediction](https://paperswithcode.com/task/mortality-prediction)                  |死亡率预测        |       1|                                                                           0|
|[Seizure Prediction](https://paperswithcode.com/task/seizure-prediction)                      |缉获预测          |       1|                                                                           0|
|[Stock Market Prediction](https://paperswithcode.com/task/stock-market-prediction)            |股票市场预测      |       1|[3](https://paperswithcode.com/area/time-series/stock-market-prediction)    |
|[Synthetic Data Generation](https://paperswithcode.com/task/synthetic-data-generation)        |合成数据生成      |       1|                                                                           0|
|[Time Series Clustering](https://paperswithcode.com/task/time-series-clustering)              |时序列聚类        |       1|                                                                           0|
|Attention Score Prediction                                                                    |注意分数预测      |       0|                                                                           0|
|ECG Denoising                                                                                 |ECG 去诺辛        |       0|                                                                           0|
|EEG Artifact Removal                                                                          |EEG 项目删除      |       0|                                                                           0|
|EEG emotion recognition                                                                       |EEG 情绪识别      |       0|                                                                           0|
|Electron Microscopy Image Segmentation                                                        |电子显微镜图像分割|       0|                                                                           0|
|Heart Rate Estimation                                                                         |心率估计          |       0|                                                                           0|
|K-complex detection                                                                           |K 复杂检测        |       0|                                                                           0|
|Linear Dynamical Systems Identification                                                       |线性动力系统识别  |       0|                                                                           0|
|Materials Screening                                                                           |材料筛选          |       0|                                                                           0|
|Multivariate Time Series Imputation                                                           |多变量时序列计算  |       0|                                                                           0|
|Semanticity Prediction                                                                        |语义预测          |       0|                                                                           0|
|Stock Trend Prediction                                                                        |股票趋势预测      |       0|                                                                           0|
|Total Magnetization                                                                           |总磁化            |       0|                                                                           0|
|[Data Mining](https://paperswithcode.com/task/data-mining)                                    |数据挖掘          |       0|[4](https://paperswithcode.com/area/natural-language-processing/data-mining)|
|[Time Series Forecasting](https://paperswithcode.com/task/time-series-forecasting)            |时间序列预测      |       0|[4](https://paperswithcode.com/area/time-series/time-series-forecasting)    |
|[Time Series Prediction](https://paperswithcode.com/task/time-series-prediction)              |时间序列预测      |       0|                                                                           0|


## Model Optimization `模型优化`

|         English          |  Mandarin  |Datasets|Subtasks|
|--------------------------|------------|-------:|-------:|
|Neural Network Compression|神经网络压缩|       0|       0|


## Information Retrieval (IR) `信息检索`

|                                    English                                     |       Mandarin       |Datasets|                                       Subtasks                                        |
|--------------------------------------------------------------------------------|----------------------|-------:|---------------------------------------------------------------------------------------|
|Drug–drug Interaction Extraction                                                |药物与药物相互作用提取|       0|                                                                                      0|
|Scientific Results Extraction                                                   |科学结果提取          |       0|                                                                                      0|
|[Information Extraction](https://paperswithcode.com/task/information-extraction)|信息提取              |       0|[6](https://paperswithcode.com/area/natural-language-processing/information-extraction)|


## Recommender System `推荐系统`

|        English        |Mandarin|Datasets|Subtasks|
|-----------------------|--------|-------:|-------:|
|Collaborative Filtering|协同过滤|       0|       0|


## Autonomous Driving `自动驾驶技术`

|                                             English                                              |      Mandarin      |Datasets|Subtasks|
|--------------------------------------------------------------------------------------------------|--------------------|-------:|-------:|
|[Birds Eye View Object Detection](https://paperswithcode.com/task/birds-eye-view-object-detection)|正投影视图目标检测  |      18|       0|
|Autonomous Flight (Dense Forest)                                                                  |自主飞行（密集森林）|       0|       0|
|Lane Detection                                                                                    |车道检测            |       0|       0|
|Pedestrian Attribute Recognition                                                                  |行人属性识别        |       0|       0|
|Pedestrian Detection                                                                              |行人检测            |       0|       0|
|Traffic Sign Recognition                                                                          |交通标志识别        |       0|       0|


## Generative Adversarial Networks (GAN) `生成对抗网络`

|                                 English                                  |       Mandarin       |Datasets|                               Subtasks                                |
|--------------------------------------------------------------------------|----------------------|-------:|-----------------------------------------------------------------------|
|[Adversarial Defense](https://paperswithcode.com/task/adversarial-defense)|对抗防御              |       8|[1](https://paperswithcode.com/area/adversarial/adversarial-defense)   |
|[Video Generation](https://paperswithcode.com/task/video-generation)      |视频生成              |       7|[1](https://paperswithcode.com/area/computer-vision/video-generation)  |
|Representation Learning In Generative Models                              |使用表征学习的生成模型|       0|                                                                      0|
|[Adversarial Attack](https://paperswithcode.com/task/adversarial-attack)  |对抗攻击              |       0|[1](https://paperswithcode.com/area/computer-vision/adversarial-attack)|


## Machine Learning (ML) `机器学习`

|                                        English                                         |     Mandarin     |Datasets|                               Subtasks                               |
|----------------------------------------------------------------------------------------|------------------|-------:|----------------------------------------------------------------------|
|[Feature Selection](https://paperswithcode.com/task/feature-selection)                  |特征选择          |      26|                                                                     0|
|[Incremental Learning](https://paperswithcode.com/task/incremental-learning)            |增量学习          |      10|                                                                     0|
|[Continual Learning](https://paperswithcode.com/task/continual-learning)                |持续学习          |       9|                                                                     0|
|[Multi-Label Classification](https://paperswithcode.com/task/multi-label-classification)|多标签分类        |       6|                                                                     0|
|[Metric Learning](https://paperswithcode.com/task/metric-learning)                      |度量学习          |       5|                                                                     0|
|[Intrusion Detection](https://paperswithcode.com/task/intrusion-detection)              |入侵检测          |       4|[1](https://paperswithcode.com/area/miscellaneous/intrusion-detection)|
|[Sparse Learning](https://paperswithcode.com/task/sparse-learning)                      |稀疏学习          |       3|                                                                     0|
|[Music Modeling](https://paperswithcode.com/task/music-modeling)                        |音乐建模          |       2|                                                                     0|
|[Surgical Skills Evaluation](https://paperswithcode.com/task/surgical-skills-evaluation)|手术技术评估      |       2|                                                                     0|
|[Type Prediction](https://paperswithcode.com/task/type-prediction)                      |类型预测          |       2|                                                                     0|
|[Event Data Classification](https://paperswithcode.com/task/event-data-classification)  |事件数据分类      |       1|                                                                     0|
|[Feature Engineering](https://paperswithcode.com/task/feature-engineering)              |功能工程          |       1|                                                                     0|
|[Multi-target Regression](https://paperswithcode.com/task/multi-target-regression)      |多目标回归        |       1|                                                                     0|
|[Music Source Separation](https://paperswithcode.com/task/music-source-separation)      |音乐源分离        |       1|                                                                     0|
|[Twitter Bot Detection](https://paperswithcode.com/task/twitter-bot-detection)          |推特机器人检测    |       1|                                                                     0|
|[Value Prediction](https://paperswithcode.com/task/value-prediction)                    |价值预测          |       1|                                                                     0|
|EMG Signal Prediction                                                                   |EMG 信号预测      |       0|                                                                     0|
|LWR Classification                                                                      |LWR 分类          |       0|                                                                     0|
|Malware Detection                                                                       |恶意软件检测      |       0|                                                                     0|
|Medial knee JRF Prediction                                                              |中年膝盖 JRF 预测 |       0|                                                                     0|
|Muscle Force Prediction                                                                 |肌肉力预测        |       0|                                                                     0|
|Myocardial infarction detection                                                         |心肌梗塞检测      |       0|                                                                     0|
|Noise Level Prediction                                                                  |噪声级预测        |       0|                                                                     0|
|Predicting Patient Outcomes                                                             |预测患者结果      |       0|                                                                     0|
|QRS Complex Detection                                                                   |QRS 复杂检测      |       0|                                                                     0|
|Sleep Apnea Detection                                                                   |睡眠呼吸暂停检测  |       0|                                                                     0|
|Sleep Arousal Detection                                                                 |睡眠唤醒检测      |       0|                                                                     0|
|Sleep Quality Prediction                                                                |睡眠质量预测      |       0|                                                                     0|
|Sleep Stage Detection                                                                   |睡眠阶段检测      |       0|                                                                     0|
|Sparse Representation-based Classification                                              |基于稀疏表示的分类|       0|                                                                     0|
|Spindle Detection                                                                       |主轴检测          |       0|                                                                     0|
|Stroke Classification                                                                   |笔划分类          |       0|                                                                     0|
|Subspace Clustering                                                                     |子空间聚类        |       0|                                                                     0|


## Deep Learning (DL) `深度学习`

|                                                    English                                                     |     Mandarin     |Datasets|                                        Subtasks                                        |
|----------------------------------------------------------------------------------------------------------------|------------------|-------:|----------------------------------------------------------------------------------------|
|[Link Prediction](https://paperswithcode.com/task/link-prediction)                                              |链接预测          |      52|[3](https://paperswithcode.com/area/graphs/link-prediction)                             |
|[Data-to-Text Generation](https://paperswithcode.com/task/data-to-text-generation)                              |数据到文本的生成  |      15|[1](https://paperswithcode.com/area/natural-language-processing/data-to-text-generation)|
|[Neural Architecture Search](https://paperswithcode.com/task/neural-architecture-search)                        |神经结构搜索      |      13|0                                                                                       |
|[Protein Secondary Structure Prediction](https://paperswithcode.com/task/protein-secondary-structure-prediction)|蛋白质二级结构预测|       7|0                                                                                       |
|[Code Generation](https://paperswithcode.com/task/code-generation)                                              |代码生成          |       4|0                                                                                       |
|[Link Sign Prediction](https://paperswithcode.com/task/link-sign-prediction)                                    |链接符号预测      |       4|0                                                                                       |
|[Gene Interaction Prediction](https://paperswithcode.com/task/gene-interaction-prediction)                      |基因相互作用预测  |       2|0                                                                                       |
|[Quantization](https://paperswithcode.com/task/quantization)                                                    |量化              |       2|[1](https://paperswithcode.com/area/computer-vision/quantization)                       |
|[Lung Nodule Classification](https://paperswithcode.com/task/lung-nodule-classification)                        |肺结核分类        |       1|0                                                                                       |
|[Lung Nodule Detection](https://paperswithcode.com/task/lung-nodule-detection)                                  |肺结核检测        |       1|0                                                                                       |
|[Music Genre Recognition](https://paperswithcode.com/task/music-genre-recognition)                              |音乐类型识别      |       1|0                                                                                       |
|[Pulmonary Embolism Detection](https://paperswithcode.com/task/pulmonary-embolism-detection)                    |肺栓塞检测        |       1|0                                                                                       |
|ALS Detection                                                                                                   |ALS 检测          |       0|                                                                                       0|
|Atrial Fibrillation Detection                                                                                   |心房颤动检测      |       0|                                                                                       0|
|Conditional Program Generation                                                                                  |条件程序生成      |       0|                                                                                       0|
|Congestive Heart Failure detection                                                                              |充血性心力衰竭检测|       0|                                                                                       0|
|Disease Trajectory Forecasting                                                                                  |疾病轨迹预测      |       0|                                                                                       0|
|Neural Network Compression                                                                                      |神经网络压缩      |       0|                                                                                       0|
|Outdoor Light Source Estimation                                                                                 |室外光源估计      |       0|                                                                                       0|
|Pancreas Segmentation                                                                                           |胰腺分割          |       0|                                                                                       0|
|Pulmonary Artery–Vein Classification                                                                            |肺动脉+静脉分类   |       0|                                                                                       0|
|[Music Classification](https://paperswithcode.com/task/music-classification)                                    |音乐分类          |       0|0                                                                                       |


## Reinforcement Learning (RL) `强化学习`

|                                                English                                                 |   Mandarin   |Datasets|                                     Subtasks                                      |
|--------------------------------------------------------------------------------------------------------|--------------|-------:|-----------------------------------------------------------------------------------|
|[General Reinforcement Learning](https://paperswithcode.com/task/general-reinforcement-learning)        |一般强化学习  |       6|[3](https://paperswithcode.com/area/robots/general-reinforcement-learning)         |
|[Multi-agent Reinforcement Learning](https://paperswithcode.com/task/multi-agent-reinforcement-learning)|多代理强化学习|       1|[1](https://paperswithcode.com/area/methodology/multi-agent-reinforcement-learning)|


## Multimodal Learning `多模态学习`

|                     English                      |          Mandarin          |Datasets|Subtasks|
|--------------------------------------------------|----------------------------|-------:|-------:|
|Cross-Modal Retrieval                             |交叉模式检索                |       0|       0|
|Multimodal Activity Recognition                   |多式联运活动识别            |       0|       0|
|Multimodal Emotion Recognition                    |多式联运情感识别            |       0|       0|
|Multimodal Metaphor Recognition                   |多式联运隐喻识别            |       0|       0|
|Multimodal Sentiment Analysis                     |多式联运情绪分析            |       0|       0|
|Multimodal Sleep Stage Detection                  |多式联运睡眠阶段检测        |       0|       0|
|Multimodal Unsupervised Image-To-Image Translation|多式联运无监督图像到图像转换|       0|       0|


## Transfer Learning `迁移学习`

|                                   English                                    |    Mandarin     |Datasets|                                 Subtasks                                 |
|------------------------------------------------------------------------------|-----------------|-------:|--------------------------------------------------------------------------|
|[Domain Adaptation](https://paperswithcode.com/task/domain-adaptation)        |域适应           |      24|[4](https://paperswithcode.com/area/computer-vision/domain-adaptation)    |
|[Transfer Learning](https://paperswithcode.com/task/transfer-learning)        |迁移学习/转移学习|       6|[3](https://paperswithcode.com/area/methodology/transfer-learning)        |
|[Cross-Domain Few-Shot](https://paperswithcode.com/task/cross-domain-few-shot)|跨域很少拍摄     |       1|[1](https://paperswithcode.com/area/computer-vision/cross-domain-few-shot)|
|Cross-Domain Named Entity Recognition                                         |跨域命名实体识别 |       0|                                                                         0|
|Domain Generalization                                                         |域泛化           |       0|                                                                         0|
|Multi-Task Learning                                                           |多任务学习       |       0|                                                                         0|
|Partial Domain Adaptation                                                     |部分域适应       |       0|                                                                         0|
|Unsupervised Domain Adaptation                                                |无监督域适应     |       0|                                                                         0|


## Meta Learning `元学习`

|                                English                                 |     Mandarin     |Datasets|                               Subtasks                                |
|------------------------------------------------------------------------|------------------|-------:|-----------------------------------------------------------------------|
|[Few-Shot Learning](https://paperswithcode.com/task/few-shot-learning)  |少量样本就快速学习|       2|[6](https://paperswithcode.com/area/computer-vision/few-shot-learning) |
|[Meta-Learning](https://paperswithcode.com/task/meta-learning)          |元学习            |       2|[2](https://paperswithcode.com/area/methodology/meta-learning)         |
|[Zero-Shot Learning](https://paperswithcode.com/task/zero-shot-learning)|零次学习          |       2|[2](https://paperswithcode.com/area/computer-vision/zero-shot-learning)|
|[One-Shot Learning](https://paperswithcode.com/task/one-shot-learning)  |一次学习          |       1|0                                                                      |
|Compositional Zero-Shot Learning                                        |作文零射学习      |       0|                                                                      0|
|Few-Shot Object Detection                                               |很少拍摄的对象检测|       0|                                                                      0|
|One-Shot Instance Segmentation                                          |单次实例分割      |       0|                                                                      0|
|One-Shot Object Detection                                               |单次对象检测      |       0|                                                                      0|
|One-Shot Segmentation                                                   |单次分割          |       0|                                                                      0|
|Zero-Shot Action Recognition                                            |零射击动作识别    |       0|                                                                      0|
|Zero-Shot Object Detection                                              |零射对象检测      |       0|                                                                      0|


## Self-Supervised Learning `自监督学习`

|                                                   English                                                    |     Mandarin     |Datasets|Subtasks|
|--------------------------------------------------------------------------------------------------------------|------------------|-------:|-------:|
|[Weakly Supervised Action Localization](https://paperswithcode.com/task/weakly-supervised-action-localization)|弱监督的操作本地化|       4|       0|
|Self-Supervised Action Recognition                                                                            |自我监督的操作识别|       0|       0|
|Weakly Supervised Object Detection                                                                            |弱监督对象检测    |       0|       0|


## Semi-Supervised Learning `半监督学习`

|                                                       English                                                        |     Mandarin     |Datasets|                                           Subtasks                                           |
|----------------------------------------------------------------------------------------------------------------------|------------------|-------:|----------------------------------------------------------------------------------------------|
|[Semi-Supervised Video Object Segmentation](https://paperswithcode.com/task/semi-supervised-video-object-segmentation)|半监督视频对象分割|       5|[1](https://paperswithcode.com/area/computer-vision/semi-supervised-video-object-segmentation)|


## Unsupervised Learning `无监督学习`

|                     English                      |          Mandarin          |Datasets|Subtasks|
|--------------------------------------------------|----------------------------|-------:|-------:|
|Multimodal Unsupervised Image-To-Image Translation|多式联运无监督图像到图像转换|       0|       0|
|Unsupervised Anomaly Detection                    |无监督异常检测              |       0|       0|
|Unsupervised Domain Adaptation                    |无监督域适应                |       0|       0|
|Unsupervised Facial Landmark Detection            |无监督的面部地标检测        |       0|       0|
|Unsupervised Person Re-Identification             |无监督的人重新识别          |       0|       0|
|Unsupervised Semantic Segmentation                |无监督语义分割              |       0|       0|
|Unsupervised Video Object Segmentation            |无监督视频对象分割          |       0|       0|
|Unsupervised Video Summarization                  |无监督视频汇总              |       0|       0|
|Unsupervised image classification                 |无监督图像分类              |       0|       0|
|Video Frame Interpolation                         |视频帧插值                  |       0|       0|


## AI Safety `人工智能安全`

|                             English                              |  Mandarin  |Datasets|Subtasks|
|------------------------------------------------------------------|------------|-------:|-------:|
|[Fraud Detection](https://paperswithcode.com/task/fraud-detection)|欺诈检测    |       1|       0|
|Network Intrusion Detection                                       |网络入侵检测|       0|       0|


## Others `其他`

|                                                                    English                                                                     |       Mandarin       |Datasets|Subtasks|
|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-------:|-------:|
|[Participant Intervention Comparison Outcome Extraction](https://paperswithcode.com/task/participant-intervention-comparison-outcome-extraction)|参与者干预比较结果提取|       1|       0|
|Instrumentals Detection                                                                                                                         |仪器检测              |       0|       0|


## Acknowledgements
- [paperswithcode/paperswithcode-data](https://github.com/paperswithcode/paperswithcode-data) - paperswithcode.com evaluation-tables dataset for metrics and links.
- [jiqizhixin/Artificial-Intelligence-Terminology](https://github.com/jiqizhixin/Artificial-Intelligence-Terminology) - English-Chinese paired terminologies

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://allcontributors.org) specification.
Contributions of any kind are welcome!

## License
[CC-BY-SA](LICENSE)

---

<div align="center">
  <sub>Built with ❤︎ by the  
  <a href="https://machinelearning.sg">ML community in Singapore</a>.
</div>