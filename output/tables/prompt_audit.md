| item                                           | status  | detail                                                              |
| ---------------------------------------------- | ------- | ------------------------------------------------------------------- |
| Painel anual 1995-2024                         | ok      | ate 2024                                                            |
| Cinco metricas                                 | ok      | Domar, Wicksell, EMA, ISF, IDEC                                     |
| ICRA                                           | ok      | indice composto exportado                                           |
| VAR por pais                                   | ok      | lags por AIC/BIC e dummy 2008-2009                                  |
| Monte Carlo 10.000 trajetorias                 | ok      | fan charts e probabilidade em 5 anos                                |
| Cenarios A/B/C                                 | ok      | baseline, choque de juros, recessao                                 |
| Tabela diagnostica base 2024                   | ok      | csv e markdown                                                      |
| Series temporais das metricas                  | ok      | csv + graficos                                                      |
| Agregado global                                | ok      | ponderacao PPP                                                      |
| Painel de paises/agregados                     | ok      | 22 unidades calculadas; o prompt enumera 22, apesar de mencionar 23 |
| Trilha de fontes                               | ok      | source_manifest.json, data_quality.csv e source_trace_2024.csv      |
| Limitacoes explicitas                          | ok      | limitations.md exportado                                            |
| Resumo estocastico                             | ok      | stochastic_summary.csv                                              |
| Cobertura de dados oficiais                    | ok      | 22 unidades no painel bruto                                         |
| Series sem fonte monetaria direta em 2024      | atencao | 7 unidades no ano-base; ver source_trace_2024.csv                   |
| Series sem fonte longa direta de juros em 2024 | ok      | 0 unidades no ano-base; proxies observados estao rastreados         |