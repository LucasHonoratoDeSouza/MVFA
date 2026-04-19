# MVFA

Implementacao em Python do Modelo de Viabilidade Fiscal Austriaca descrito em [prompt.txt](/home/lucas/dados%20escola%20austriaca/prompt.txt) e documentado pela base bibliografica em [livros.txt](/home/lucas/dados%20escola%20austriaca/livros.txt).

## Estrutura

- `data/raw/` contem os dados brutos. O arquivo esperado pelo pipeline e `data/raw/panel.csv`.
- `data/processed/` recebe a base limpa e padronizada.
- `models/metrics.py` calcula as cinco metricas, o ICRA e a classificacao preliminar.
- `models/var_model.py` estima um VAR por pais com OLS em forma reduzida.
- `models/monte_carlo.py` gera fan charts, cenarios deterministas e risco estocastico.
- `output.py` exporta tabelas, graficos e apendice bibliografico.

## Como usar

1. Rode `python3 download_data.py --refresh` para baixar o painel bruto a partir de fontes oficiais.
2. Rode `python3 main.py --refresh-data` para reconstruir a base, limpar, calcular metricas e gerar os outputs.
3. Consulte `output/tables/` e `output/charts/`.

## Schema minimo

As colunas principais usadas pelo pipeline sao:

- `country`, `year`
- `debt_gdp`, `debt_nominal`, `avg_debt_nominal`
- `interest_paid_nominal`, `primary_balance_gdp`
- `gdp_real_growth`, `gdp_deflator_inflation`
- `policy_rate_nominal`, `yield_10y_nominal`, `inflation_forward_12m`
- `m2_nominal`, `gdp_nominal`, `gdp_real`, `inflation_target`
- `tax_revenue_nominal`
- `investment_sensitive_nominal`, `investment_total_nominal`, `private_savings_gdp`
- `housing_credit_share`, `fx_real_change`
- `natural_rate_hlw`, `natural_rate_lm`, `natural_rate_favar`

## Observacoes

- O pipeline constroi `data/raw/panel.csv` diretamente de APIs/downloads oficiais acessiveis no ambiente.
- Quando uma serie de `yield_10y_nominal` nao existe em fonte oficial continua, o codigo usa apenas proxies observados e reais, nunca um premio sinteticamente inventado.
- O VAR foi implementado com `numpy`, garantindo total ausencia de dependencia de `statsmodels` ou pacotes externos de regressao (o que tambem elimina warnings de indexacao temporal do Pandas). Isso vale tanto para o modelo principal quanto para as hipoteses estruturais (FAVAR em `rstar.py`).
- A classificacao final combina violacoes observadas, ICRA e risco estocastico em horizonte de 5 anos.
- **Risco e Ilusao Nominal**: Cenarios de hiperinflacao (como da Argentina) costumam "corrigir matematimaticamente" a relacao Div./PIB, pois a taxa de juros real fica extremamente negativa. O componente estocastico foca nesse crescimento da divida e pode emitir sinal tolerante ("Sustentavel" na frente estocastica). No entanto, a visao qualitativa das 5 Metricas, em especial a Expansao Monetaria Austriaca (EMA), aponta a real fragilidade desse modelo de expansao sem poupanca. O status final pondera efetivamente esses conflitos e alerta "Insustentavel".
