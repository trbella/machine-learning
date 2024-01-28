# Towards an Early Warning System for Cardiovascular Disease Deaths Based on Temperature and Air Quality Variables

# Authors and [CRediT statement](https://www.elsevier.com/authors/policies-and-guidelines/credit-author-statement)
* [Thiago Ribas Bella](https://www.linkedin.com/in/thiago-ribas-bella-016380149/)<sup>a</sup> - Conceptualization, Methodology, Software, Formal analysis, Investigation, Data Curation, Writing - Original Draft, Writing - Reviewing and Editing
* [Júlia Perassolli De Lázari](https://github.com/juliaplazari)<sup>a</sup> - Data Curation
* [Liriam Samejima Teixeira](https://www.linkedin.com/in/liriam-samejima-teixeira-944996140/)<sup>b</sup> - Data Curation
* [Eliana Cotta de Faria](http://lattes.cnpq.br/9697459943018307)<sup>b</sup> - Supervision and Project administration
* [Ana Maria Heuminski de Avila](http://lattes.cnpq.br/7582420301693448)<sup>c</sup> - Supervision and Project administration
* [Paula Dornhofer Paro Costa](http://lattes.cnpq.br/4518009815956207)<sup>a</sup> - Conceptualization, Investigation, Resources, Supervision, Project administration, Writing-Original Draft, Writing-Reviewing and Editing

<sup>a</sup> *Department of Computer Engineering and Industrial Automation - FEEC/UNICAMP, Brasil*

<sup>b</sup> *Department of clinical pathology - FCM/UNICAMP, Brasil*

<sup>c</sup> *Center for Meteorological and Climatic Research Applied to Agriculture - UNICAMP, Brasil*


# **Abstract**

Cardiovascular diseases (CVDs) are the world-leading cause of death, with several studies showing that they are closely related to environmental variables,
especially temperature and air quality. Early warning systems aim to mitigate
the increased risk of death due to extreme environmental conditions. A key challenge is that the relationship between environmental variables and CVD deaths
varies significantly among different regions and populations worldwide. Existing
approaches have focused on deriving warning thresholds or intervals of attention for studied regions. However, they are based on non-standard parameters,
resulting in methods that vary among approaches, increasing the difficulty of
reproducing them. We propose the adoption of data-driven, machine learningbased models that would be able to analyze current environmental conditions
and predict the increase in the number of CVD deaths in advance. The best-inclass model can be trained and replicated by accessing similar input data for any
world region. In this work, we implemented and evaluated the results of linear
regression models with SARIMAX errors (LR-SARIMAX) and LSTM neural networks. As source data, we used daily occurrences of CVDs deaths, minimum
temperature, carbon monoxide, and particulate matter data from Campinas
city (Brazil) from 2001 to 2018. We also explored models derived from aggregated weekly and monthly frequencies. The models captured the seasonal variations with narrow confidence intervals, and the minimum temperature showed
to be the most relevant prediction variable. The models predicting CVD deaths
one month in advance presented the lowest prediction errors. LR-SARIMAX
monthly model, with just one lag of all the predictor variables, resulted in an
error of 6.4% in deaths that vary from 139 to 286 monthly. With weather forecasting, data-driven models can predict events that will require attention and
preparedness from health care services and the population in advance.


***Keywords***: Cardiovascular death, Time series prediction, Ambient
temperature, Pollution impact, Early warning system
