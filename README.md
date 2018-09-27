# Semantix_Desafio_DataScience

README

Semantix: Desafio Data Science;
Autor: Felipe Azevedo Pinagé;
Dataset: Bank Marketing;

- Bibliotecas usadas:
Pandas, Numpy, Sklearn

- Funções criadas:
codificaCat(): usada para codificar os dados categóricos
preparaDados(): usada para separar a base entre dados e rótulos

- Questão 1:
Os dados são agrupados por profissão e contando o número de empréstimos imobiliários (housing) e pessoais (loan) cada profissão tem.
Como resultado, é exibida a profissão com mais tendência a fazer empréstimo e qual seu tipo.

- Questão 2 e Questão 3:
Assume-se que o sucesso da campanha se dá com os cliente que fizeram a adesão do plano. Logo os dados são agrupados pela adesão (y) onde o número de contatos (campaign) nesta campanha foram somados, verificado a média e valores máximos.

- Questão 4:
Foi feita a correlação dos dados de contato da campanha atual (campaign) e da campanha anterior (previous). Segundo essa correlação é apresentado o grau de relevância da campanha anterior nos resultados da atual.

- Questão 5 e Questão 6:
Foi utilizada a Regressão Logística nas variáveis categóricas binárias 'default' (questão 5) e 'housing' (questão 6) para estimar a probabilidade associada à tais ocorrências. 
Em seguida foi utilizada uma abordagem de seleção de características, chamada Recursive Feature Elimination (RFE) para fazer o 'ranking' das características mais relevantes, no caso, 1 característica para a questão 5, e 3 características para a questão 6.

- Resultados:
Os resultados são escritos no arquivo 'resultados.txt' gerado após a execução do programa 'semantix.py'. 
