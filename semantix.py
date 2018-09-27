import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# PRE-PROCESSAMENTO: codificar os dados categóricos
def codificaCat():
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat = list(set(cols) - set(num_cols))
    for i in range(cat.__len__()):
        ratingList = df[cat[i]].tolist()
        labelEncoderRating = LabelEncoder()
        labelEncoderRating.fit(ratingList)
        labelsRating = labelEncoderRating.transform(ratingList)
        df[cat[i]] = pd.Series(labelsRating)

# PREPARAÇÃO: SEPARA OS DADOS DOS RÓTULOS
def preparaDados(dataframe, nomeLabel):
    # PREPARANDO
    headers = list(dataframe)
    label_index = headers.index(nomeLabel)
    headers.remove(nomeLabel)
    dataset = np.asmatrix(dataframe)
    # SEPARANDO DADOS DOS RÓTULOS
    dataNoLabels = np.delete(dataset, label_index, 1)  # obter os dados sem os rótulos
    labels = dataset[:, label_index]  # obter apenas os rótulos
    return [dataNoLabels, labels, headers]

df = pd.read_csv('bank-full.csv', sep=';', header=0).replace(np.NaN, 0) #lê o arquivo csv
file = open("resultados.txt","w");

#(Q1)------------------------------------------------------------------------------------------------------#
teste1 = df[['job','housing']]
hous = teste1[teste1["housing"]=="yes"].groupby("job").count().sort_values('housing', ascending = False)
teste2 = df[['job','loan']]
loan = teste2[teste2["loan"]=="yes"].groupby("job").count().sort_values('loan', ascending = False)
if hous["housing"].max() >= loan["loan"].max():
    file.write("QUESTÃO 1.\nProfissao e tipo de emprestimo:\n" + str(hous[:1]))
else:
    file.write("QUESTÃO 1.\nProfissao e tipo de emprestimo:\n" + str(loan[:1]))
#(Q2)e(Q3)------------------------------------------------------------------------------------------------------#
succ = df.groupby("y")["campaign"].sum()
succMean = df.groupby("y")["campaign"].mean()
succMax = df.groupby("y")["campaign"].max()
if succ.loc["yes"] >= succ.loc["no"]:
    file.write("\n\nQUESTÃO 2.\nEm uma relação entre número de contatos e sucesso da campanha. \n"
          "Quanto mais contatos nesta campanha, mais chances de sucesso."
          "\n\nQUESTÃO 3.\nPara otimizar a adesão é indicado um número médio de %0.1f ligações, \n"
          "e um número máximo de %0.1f ligações" % (round(succMean.loc["yes"]), round(succMax.loc["yes"])))
else:
    file.write("\n\nQUESTÃO 2.\nEm uma relação entre número de contatos e sucesso da campanha. \n"
          "Quanto menos contatos nesta campanha, mais chances de sucesso."
          "\n\nQUESTÃO 3.\nPara otimizar a adesão é indicado um número médio de %0.1f ligações, \n"
          "e um número máximo de %0.1f ligações" % (round(succMean.loc["yes"]), round(succMax.loc["yes"])))
#(Q4)------------------------------------------------------------------------------------------------------#
x = np.array(df.loc[:,"campaign"])
y = np.array(df.loc[:,"previous"])
mat = abs(np.corrcoef(x, y))
corr = mat[0,1]
if corr >= 0.9:
    file.write("\n\nQUESTÃO 4.\nOs resultados da campanha anterior têm MUITA relevância nos resultados da campanha atual")
elif corr >= 0.7:
    file.write("\n\nQUESTÃO 4.\nOs resultados da campanha anterior têm relevância nos resultados da campanha atual")
elif corr >= 0.4:
    file.write("\n\nQUESTÃO 4.\nOs resultados da campanha anterior têm relevância MODERADA nos resultados da campanha atual")
elif corr >= 0.2:
    file.write("\n\nQUESTÃO 4.\nOs resultados da campanha anterior têm POUCA relevância nos resultados da campanha atual")
else:
    file.write("\n\nQUESTÃO 4.\nOs resultados da campanha anterior NÃO têm relevância nos resultados da campanha atual")
#(Q5)------------------------------------------------------------------------------------------------------#
codificaCat(); #codifica dados categóricos
data, labels, features = preparaDados(df,'default');
model = LogisticRegression()
rfe = RFE(model, 1)
rfe = rfe.fit(data, np.ravel(labels,order='C'))
bestF = rfe.ranking_
proFeat = []
for i in range(bestF.__len__()):
    if bestF[i] == 1:
        proFeat.append(features[i])
file.write("\n\nQUESTÃO 5.\nO fator determinante para o banco exigir um seguro de crédito\n"
    "é estar em débito com o banco (default) que está primariamente relacionado a: " + str(proFeat));
#(Q6)------------------------------------------------------------------------------------------------------#
data, labels, features = preparaDados(df,'housing');
model = LogisticRegression()
rfe = RFE(model, 3)
rfe = rfe.fit(data, np.ravel(labels,order='C'))
bestF = rfe.ranking_
proFeat = []
for i in range(bestF.__len__()):
    if bestF[i] == 1:
        proFeat.append(features[i])
file.write("\n\nQUESTÃO 6.\nAs características mais proeminentes de quem possui um empréstimo\n"
    "imobiliário são " + str(proFeat));

file.close();
