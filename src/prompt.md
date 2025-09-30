# Prompt 1 - Gerar relatório

## Tarefa

Resolver a tarefa de **iteração de valor** descrita no `@tasks.md` usando `@01_value_iteration.py`.

## Requisitos

- Células de markdown devem ser bem descritivas
- Os outputs do código devem ser bem descritivos (usar prints explicando o que está sendo feito)
- Ao invés de só printar o resultado a esmo, explicar cada etapa

---

# Prompt 2 - Converter .py para notebook

## Tarefa

Converter arquivo `.py` (formato jupytext percent) para notebook `.ipynb`.

```bash
poetry run jupytext --from py:percent --to ipynb src/value_iteration.py
```

Executa e converte notebook `.ipynb` para HTML mostrando apenas resultados e markdown (sem código).

```bash
poetry run jupyter nbconvert --to html --execute--no-input value_iteration.ipynb --output value_iteration_done.html
```