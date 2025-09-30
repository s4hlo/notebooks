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

## Comando

```bash
poetry run jupytext --from py:percent --to ipynb src/value_iteration.py
```

## Resultado

- Gera `src/01_value_iteration.ipynb` a partir do arquivo `.py`
- Preserva todas as células markdown e código
- Mantém formatação jupytext

---

# Prompt 3 - Converter notebook para HTML

## Tarefa

Converter notebook `.ipynb` para HTML mostrando apenas resultados e markdown (sem código).

## Comando

before we must run the notebook to generate the results

```bash
poetry run jupyter  
```

convert the executed notebook to html

```bash
poetry run jupyter nbconvert --to html --execute--no-input value_iteration.ipynb --output value_iteration_done.html
```


## Resultado

- Gera `src/01_value_iteration.html` 
- Mostra apenas outputs das células de código
- Inclui todas as células markdown
- Oculta o código fonte (`--no-input`)
- Pronto para conversão para PDF via navegador
