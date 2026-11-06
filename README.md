# jev-study

Estudo técnico do Jev, primeiro modelo System One da [TypeSafe](https://typesafe.ai/), contra um baseline de LLM com saída estruturada. O repositório é de Gabriel, desenvolvedor de IA e automação. Há o aparato de execução (contrato tipado, clientes, runner e registro cru) e o gabarito sintético v1 em `data/eval/decisions.v1.jsonl`. Não há resultado de modelo aqui.

Jev não é um LLM de chat. O cliente manda um estado não estruturado e perguntas tipadas e recebe decisões com probabilidade. A chamada publicada é `POST /v1/systemone` (`jev-latest` ou um id versionado, hoje `jev-1.14.0`). Referências: [System One](https://docs.typesafe.ai/concepts/system-one), [API](https://docs.typesafe.ai/api/system-one) e [Models](https://docs.typesafe.ai/models).

## Objetivo

Medir, nas mesmas tarefas rotuladas, três coisas:

- se a resposta obedece ao schema tipado (Choice, Score, Noul, Box);
- se a decisão concorda com o gabarito;
- se a confiança declarada é calibrada: confiança mais alta deveria acompanhar acerto mais alto, se a promessa de calibração se sustentar nos itens deste estudo.

Os dois sistemas veem o mesmo `state`, as mesmas `questions` e o mesmo gabarito. A comparação é pareada.

## Hipótese

A documentação da TypeSafe descreve o System One como um modelo de decisão calibrada: as probabilidades são treinadas para refletir incerteza, e a calibração vale para grupos de previsões, não como garantia de um caso. Neste repositório isso é hipótese do study-PR3, quando houver decisões reais. Não é premissa e não é resultado. Alegação de marketing da TypeSafe não entra como número deste estudo.

Velocidade, custo relativo e "zero alucinações" só entram num relatório depois que um experimento daqui medir esses pontos. Até lá ficam fora da metodologia. A tarifa de lista do Jev 1.14 (USD 0,042 por milhão de tokens de entrada; saída publicada como gratuita) aparece só como parâmetro para estimar custo a partir de `usage`. A fonte é a página Models. Não é medição deste estudo.

## Metodologia planejada

Unidade de análise: uma pergunta sobre um estado. Cada linha do registro é uma pergunta. Várias perguntas na mesma request compartilham latência e custo; o campo `cost_scope` vale `request` para a soma não contar a mesma chamada duas vezes.

Primitivas, no formato da API:

| Tipo | Decisão | O que o registro guarda |
| --- | --- | --- |
| Choice | uma chave de `criteria` | `confidence` da API e a probabilidade da chave escolhida |
| Score | esperança na rubrica, do 0 ao último nível | `confidence` da API e a probabilidade do nível inteiro mais próximo (empate `.5` sobe) |
| Noul | P(sim), de 0 a 1 | a própria probabilidade; não há campo `confidence`, e o código não inventa um |
| Box | arestas normalizadas | `confidence` se a API mandar; sem isso, os dois campos ficam nulos |

Concordância, a implementar no study-PR3: Choice por igualdade da chave; Score pelo nível inteiro mais próximo; Noul com limiar 0,5 (probabilidade maior ou igual a 0,5 conta como sim). Calibração, no mesmo PR: confiança declarada contra acerto, para os tipos que publicam `confidence`. Para Noul a curva usa a probabilidade, porque é o número que o modelo declara.

Validade de schema: o JSON parseia no modelo compartilhado, os números caem nas faixas publicadas e o tipo da resposta é o tipo da pergunta. O cliente Jev ainda exige `usage`, porque a API o marca como obrigatório. Campo extra na resposta não invalida o contrato.

As tarefas são sintéticas, escritas no repositório, em português, com chaves em inglês. Não há dado de cliente. Box depende de imagem. O placeholder de quinze itens guarda uma pergunta Box só para o runner registrar o skip, sem asset e sem o header `TypeSafe-Danger-Zone-Images`. O gabarito v1 não inclui Box.

O baseline de LLM recebe o mesmo JSON de request no user message e devolve o mesmo schema. O prompt de sistema não contém gabarito. Ele existe porque um LLM de chat precisa de instrução; o Jev não recebe esse prompt. Essa diferença fica explícita para o relatório não tratá-la como detalhe.

## O que o repositório entrega

- Projeto Python 3.11+, `uv`, Ruff e pytest. A CI roda lint e teste sem chave da TypeSafe e sem chave de LLM.
- `JevHttpBackend`: `POST /v1/systemone` pela stdlib. Não importa o SDK da TypeSafe. Redirect é recusado para o header `Authorization` não seguir para outro host.
- `LlmStructuredBackend`: mesma resposta tipada. O pacote `openai` só é importado na chamada ao vivo; os testes injetam um completer.
- `FakeBackend` e um eco de gabarito para a CI e para o comando local. O eco usa confiança 1 e não preenche `usage`. Não é previsão.
- Quinze tarefas sintéticas de placeholder: roteamento Choice, tom, Score, Noul e uma Box pulada. O comando local roda este conjunto, não o v1.
- Gabarito v1 em `data/eval/decisions.v1.jsonl`: 45 tarefas sintéticas, 15 de cada tipo Choice, Score e Noul. O loader recusa linha malformada. Seis itens trazem `low_confidence` como desenho do texto, não como saída de modelo.
- Runner que grava backend, modelo pedido, modelo que respondeu, decisão, confiança, probabilidade, latência, tokens e custo estimado quando a tarifa e o `usage` existem.

## Dataset de avaliação

O arquivo `data/eval/decisions.v1.jsonl` é o gabarito do study-PR2. São 45 linhas: 15 Choice, 15 Score e 15 Noul. Os estados são textos sintéticos em português. As chaves das opções ficam em inglês, no formato do `POST /v1/systemone`. Não há pessoa real, dado de cliente, conteúdo de negócio privado nem imagem. Box fica de fora deste arquivo.

Cada linha traz:

| Campo | Papel |
| --- | --- |
| `schema_version` | `1`. Outra versão o loader recusa. |
| `id` | Identificador estável da tarefa. |
| `category` | `choice`, `score` ou `noul`. As perguntas da linha são desse tipo. |
| `state` | Texto de entrada, o mesmo material que os dois backends veriam. |
| `questions` | Perguntas no schema do adapter. |
| `expected` | Gabarito. Choice guarda a chave; Score guarda o índice inteiro da rubrica (`score_level`); Noul guarda sim ou não, nunca uma probabilidade. |
| `notes` | Justificativa do gabarito. No schema é opcional; neste arquivo toda linha tem nota. |
| `expectation` | `clear` ou `low_confidence`. |

`low_confidence` marca um texto escrito para admitir outra leitura. Há dois itens desses em cada tipo. O valor em `expected` é a leitura preferida, e a nota diz qual é a leitura concorrente. O campo foi preenchido junto com o item. Não é confiança declarada por um modelo, não é probabilidade, e o request que sai para o adapter não leva esse campo.

O loader é `load_eval_decisions`, em `jev_study.dataset`. Ele recusa JSON inválido, campo fora do schema, categoria diferente do tipo da pergunta, gabarito incompatível com a rubrica, id repetido, estado repetido com as mesmas perguntas e linha vazia.

Inventário, sem chamar modelo:

```bash
uv run python - <<'PY'
from jev_study.dataset import load_eval_decisions

rows = load_eval_decisions()
print(f"{len(rows)} itens")
for row in rows:
    print(f"{row.id}  {row.category}  {row.expectation}")
PY
```

`uv run jev-study` continua no placeholder de quinze itens. Esse comando não lê o v1, e o JSON que imprime é eco de gabarito, não avaliação.

Nada no arquivo confirma alegação pública de calibração, velocidade, custo ou ausência de alucinação. Esses pontos seguem como hipótese do study-PR3 e matéria do study-PR4.

## Como rodar

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run jev-study
```

`uv run jev-study` imprime o registro do eco de gabarito do placeholder de quinze itens. Confiança 1 e custo nulo nesse JSON são o placeholder, não uma avaliação do Jev e não são números do gabarito v1.

Chamada ao vivo, fora da CI, com a chave só no ambiente:

```bash
export TYPESAFE_API_KEY="..."
uv run jev-study --backend jev
```

O baseline pede o extra e as duas variáveis:

```bash
uv sync --extra llm
export LLM_API_KEY="..."
export LLM_MODEL="..."
uv run jev-study --backend llm
```

O processo não lê `.env` sozinho. Há um `.env.example`. `.env` está no `.gitignore`. Não commitar chave.

O modelo default é `jev-1.14.0`. A página Models descreve `jev-latest` como alias móvel; por isso o default é o id versionado. `JEV_MODEL=jev-latest` volta ao alias.

## Variáveis de ambiente

| Variável | Papel |
| --- | --- |
| `TYPESAFE_API_KEY` | Chave do `POST /v1/systemone`. Ganha de `JEV_API_KEY`. |
| `JEV_API_KEY` | Mesma chave, se a primeira estiver vazia. |
| `TYPESAFE_BASE_URL` | Origem. Default `https://api.typesafe.ai`. |
| `JEV_MODEL` | Id ou alias. Default `jev-1.14.0`. |
| `JEV_INPUT_USD_PER_MTOK` | Tarifa de entrada para a estimativa. Default 0,042. |
| `JEV_OUTPUT_USD_PER_MTOK` | Tarifa de saída. Default 0, como a documentação de lista. |
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | Baseline. Sem model e sem chave o comando sai com código 2. |
| `LLM_INPUT_USD_PER_MTOK`, `LLM_OUTPUT_USD_PER_MTOK` | Tarifas do baseline. Vazias significam custo indisponível. |

Se a tarifa de saída não está definida e a resposta traz tokens de saída, o custo fica nulo em vez de estimar só a entrada.

## Registro

Cada linha é um objeto JSON com: `task_id`, `source`, `backend`, `requested_model`, `model`, `question`, `question_type`, `decision`, `declared_confidence`, `declared_probability`, `latency_ms`, `estimated_cost_usd`, `cost_scope`, tarifas, `input_tokens`, `output_tokens`, `skipped`, `skip_reason`, `error`.

`model` é o id que a resposta declara. No Jev ele pode diferir do alias pedido. Sem resposta, fica nulo. `error` guarda falha de rede, HTTP, schema ou tipo trocado, e a decisão ainda é copiada quando o JSON era válido mas não batia com a pergunta.

## Roadmap

- study-PR3: harness de avaliação e métricas de calibração. Validade de schema, concordância com o gabarito e calibração da confiança declarada. A marca `low_confidence` entra como desenho do item; a curva em si não é calculada aqui.
- study-PR4: experimentos ao vivo nos dois backends e o relatório. Números de qualidade, latência e custo só depois dessa rodada, com a configuração e o id do modelo gravados.

## Ainda fora

Não há acurácia, curva de calibração, latência de produção nem custo medido. O eco local não substitui essas rodadas. Imagem e o header de visão ficam de fora do gabarito v1, embora o cliente HTTP envie o header se alguém passar `assets` ou uma pergunta Box.
