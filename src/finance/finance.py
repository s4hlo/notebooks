import pandas as pd
import numpy as np
from pathlib import Path


class NubankAnalyzer:
    def __init__(self, data_folder: str = "secret"):
        self.data_folder = data_folder
        self.df = None
        self.df_profits = None
        self.df_expenses = None
        self.load_data()
        self.separate_data()

    def add_section(self, text: str, insights: list[str]):
        string = "\n> === " + text + " ===\n"
        insights.append(string)


    def load_data(self):
        """Carrega todos os arquivos CSV do Nubank"""
        csv_files = list(Path(self.data_folder).glob("Nubank_*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"Nenhum arquivo CSV encontrado em {self.data_folder}"
            )

        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # Converter data e amount
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["amount"] = pd.to_numeric(self.df["amount"])

        # Ordenar por data
        self.df = self.df.sort_values("date")

        print(f"Carregados {len(self.df)} transações de {len(csv_files)} arquivos")

    def separate_data(self):
        """Separa os dados em gastos e recebimentos"""
        self.df_expenses = self.df[self.df["amount"] > 0].copy()
        self.df_profits = self.df[self.df["amount"] < 0].copy()

        # Converter valores negativos para positivos nos recebimentos
        self.df_profits["amount"] = abs(self.df_profits["amount"])

        print(f"Gastos: {len(self.df_expenses)} transações")
        print(f"Recebimentos: {len(self.df_profits)} transações")


    def generate_expenses_analysis(
        self,
        no_include: list[str] = [],
    ) -> str:
        """Gera análise detalhada dos gastos"""
        insights = []
        pre_expenses = self.df_expenses
        
        # Filtrar estabelecimentos que contenham as palavras
        if no_include:
            mask = pre_expenses["title"].str.lower().str.contains('|'.join(no_include), na=False)
            expenses = pre_expenses[~mask]
        else:
            expenses = pre_expenses

        if len(expenses) == 0:
            return "Nenhum gasto encontrado no período."

        start_date = expenses["date"].min()
        end_date = expenses["date"].max()
        total_expenses = expenses["amount"].sum()

        # Usar o método add_section para organizar melhor
        self.add_section("ANÁLISE DETALHADA DE GASTOS", insights)
        insights.append(
            f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}"
        )
        insights.append(f"Total de gastos: {len(expenses):,} transações")
        insights.append(f"Valor total gasto: R$ {total_expenses:,.2f}")
        insights.append(
            f"Gasto médio diário: R$ {total_expenses / ((end_date - start_date).days + 1):.2f}"
        )
        insights.append(f"Valor médio por gasto: R$ {expenses['amount'].mean():.2f}")
        insights.append(f"Mediana dos gastos: R$ {expenses['amount'].median():.2f}")
        insights.append(f"Maior gasto: R$ {expenses['amount'].max():.2f}")
        insights.append(f"Menor gasto: R$ {expenses['amount'].min():.2f}")
        insights.append("")

        # Análise de percentis
        self.add_section("ANÁLISE DE PERCENTIS", insights)
        percentis = [10, 25, 50, 75, 90, 95, 99]
        for p in percentis:
            valor = expenses["amount"].quantile(p / 100)
            insights.append(f"Percentil {p}%: R$ {valor:.2f}")
        insights.append("")

        # Top estabelecimentos (análise principal)
        self.add_section("TOP 10 ESTABELECIMENTOS POR MESES ATIVOS (MESES ATIVOS > 1)", insights)

        # Adicionar coluna de mês
        expenses["month"] = expenses["date"].dt.to_period("M")

        # Calcular transações por estabelecimento por mês
        monthly_transactions = (
            expenses.groupby(["title", "month"])
            .size()
            .reset_index(name="transacoes_mes")
        )

        # Calcular média de transações por mês para cada estabelecimento
        avg_monthly_transactions = (
            monthly_transactions.groupby("title")
            .agg({"transacoes_mes": ["mean", "std", "count"]})
            .round(2)
        )

        avg_monthly_transactions.columns = [
            "Media_Mensal",
            "Desvio_Mensal",
            "Meses_Ativos",
        ]

        # Combinar com dados gerais
        top_merchants = (
            expenses.groupby("title")
            .agg(
                {"amount": ["sum", "count", "mean", "median", "std"], "date": "nunique"}
            )
            .round(2)
        )

        top_merchants.columns = [
            "Total",
            "Qtd_Transacoes",
            "Media",
            "Mediana",
            "Desvio_Padrao",
            "Dias_Com_Gasto",
        ]

        # Combinar os dados
        top_merchants = top_merchants.join(avg_monthly_transactions)

        # Filtrar apenas estabelecimentos com mais de 1 mês ativo
        top_merchants = top_merchants[top_merchants["Meses_Ativos"] > 1]

        # Ordenar por meses ativos (mais meses primeiro)
        top_merchants = top_merchants.sort_values("Meses_Ativos", ascending=False).head(
            10
        )

        for i, (merchant, row) in enumerate(top_merchants.iterrows(), 1):
            percentage = (row["Total"] / total_expenses) * 100

            # Calcular percentis para este estabelecimento
            merchant_data = expenses[expenses["title"] == merchant]["amount"]
            p75 = merchant_data.quantile(0.75)
            p90 = merchant_data.quantile(0.90)
            p95 = merchant_data.quantile(0.95)

            insights.append(f"\n{i:2d}. {merchant}:")
            insights.append(f"     Total: R$ {row['Total']:,.2f} ({percentage:.1f}%)")
            insights.append(f"     Transações totais: {row['Qtd_Transacoes']:.0f}")
            insights.append(
                f"     Média mensal: {row['Media_Mensal']:.1f} transações/mês"
            )
            insights.append(f"     Meses ativos: {row['Meses_Ativos']:.0f}")
            insights.append(
                f"     Valor médio: R$ {row['Media']:.2f} ({row['Desvio_Padrao']:.0f})"
            )
            insights.append(f"     Mediana: R$ {row['Mediana']:.2f}")
            insights.append(
                f"     P75: R$ {p75:.2f} | P90: R$ {p90:.2f} | P95: R$ {p95:.2f}"
            )
            insights.append(f"     Dias com gasto: {row['Dias_Com_Gasto']:.0f}")

        # Análise temporal
        self.add_section("ANÁLISE TEMPORAL DE GASTOS", insights)

        # Por mês
        monthly_expenses = (
            expenses.groupby(expenses["date"].dt.to_period("M"))
            .agg({"amount": ["sum", "count", "mean"], "date": "nunique"})
            .round(2)
        )

        insights.append("\nGastos por mês:")
        for month, row in monthly_expenses.iterrows():
            insights.append(
                f"  {month}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})"
            )

        # Por dia da semana
        expenses["day_of_week"] = expenses["date"].dt.day_name()
        daily_analysis = (
            expenses.groupby("day_of_week")
            .agg({"amount": ["sum", "count", "mean"]})
            .round(2)
        )

        insights.append("\nGastos por dia da semana:")
        for day, row in daily_analysis.iterrows():
            insights.append(
                f"  {day}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})"
            )

        # Por hora (se disponível)
        if "time" in expenses.columns:
            expenses["hour"] = pd.to_datetime(expenses["time"]).dt.hour
            hourly_analysis = (
                expenses.groupby("hour")
                .agg({"amount": ["sum", "count", "mean"]})
                .round(2)
            )

            insights.append("\nGastos por hora do dia:")
            for hour, row in hourly_analysis.iterrows():
                insights.append(
                    f"  {hour:02d}h: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações)"
                )

        # Análise de frequência
        self.add_section("ANÁLISE DE FREQUÊNCIA", insights)
        expense_frequency = expenses.groupby(expenses["date"].dt.date).size()
        insights.append(
            f"Dias com gastos: {len(expense_frequency)} de {(end_date - start_date).days + 1}"
        )
        insights.append(
            f"Frequência média: {expense_frequency.mean():.1f} transações/dia"
        )
        insights.append(f"Dia com mais gastos: {expense_frequency.max()} transações")
        insights.append(
            f"Dias sem gastos: {(end_date - start_date).days + 1 - len(expense_frequency)}"
        )

        # Distribuição de valores
        self.add_section("DISTRIBUIÇÃO DE VALORES", insights)
        value_ranges = [
            (0, 10, "Até R$ 10"),
            (10, 25, "R$ 10-25"),
            (25, 50, "R$ 25-50"),
            (50, 100, "R$ 50-100"),
            (100, 200, "R$ 100-200"),
            (200, 500, "R$ 200-500"),
            (500, float("inf"), "Acima de R$ 500"),
        ]

        for min_val, max_val, label in value_ranges:
            if max_val == float("inf"):
                count = len(expenses[expenses["amount"] >= min_val])
                total_range = expenses[expenses["amount"] >= min_val]["amount"].sum()
            else:
                count = len(
                    expenses[
                        (expenses["amount"] >= min_val) & (expenses["amount"] < max_val)
                    ]
                )
                total_range = expenses[
                    (expenses["amount"] >= min_val) & (expenses["amount"] < max_val)
                ]["amount"].sum()

            percentage = (count / len(expenses)) * 100
            value_percentage = (total_range / total_expenses) * 100
            insights.append(
                f"  {label}: {count:.0f} transações ({percentage:.1f}%) - R$ {total_range:,.2f} ({value_percentage:.1f}%)"
            )

        # Estatísticas avançadas
        self.add_section("ESTATÍSTICAS AVANÇADAS", insights)
        insights.append(f"Desvio padrão: R$ {expenses['amount'].std():.2f}")
        insights.append(f"Variância: R$ {expenses['amount'].var():.2f}")
        insights.append(
            f"Coeficiente de variação: {(expenses['amount'].std() / expenses['amount'].mean()) * 100:.1f}%"
        )
        insights.append(f"Quartil 25%: R$ {expenses['amount'].quantile(0.25):.2f}")
        insights.append(f"Quartil 75%: R$ {expenses['amount'].quantile(0.75):.2f}")
        insights.append(
            f"Amplitude interquartil: R$ {expenses['amount'].quantile(0.75) - expenses['amount'].quantile(0.25):.2f}"
        )

        # Métricas estatísticas comportamentais
        self.add_section("MÉTRICAS ESTATÍSTICAS COMPORTAMENTAIS", insights)

        # Distribuição de transações de baixo valor
        small_purchases = len(expenses[expenses["amount"] <= 10])
        small_purchase_pct = (small_purchases / len(expenses)) * 100
        insights.append(
            f"Frequência de microtransações (≤R$ 10): {small_purchases:.0f} ({small_purchase_pct:.1f}%)"
        )

        # Distribuição de frequência diária
        daily_transactions = expenses.groupby(expenses["date"].dt.date).size()
        high_frequency_days = len(daily_transactions[daily_transactions >= 5])
        insights.append(
            f"Dias com alta frequência transacional (≥5): {high_frequency_days:.0f} dias"
        )

        # Índice de diversificação de estabelecimentos
        unique_merchants = expenses["title"].nunique()
        merchants_per_month = unique_merchants / 5  # 5 meses
        insights.append(
            f"Cardinalidade de estabelecimentos únicos: {unique_merchants:.0f} ({merchants_per_month:.1f}/mês)"
        )

        # Coeficiente de concentração (Índice de Herfindahl)
        top5_merchants = expenses.groupby("title")["amount"].sum().nlargest(5).sum()
        concentration_pct = (top5_merchants / total_expenses) * 100
        insights.append(
            f"Coeficiente de concentração (top 5): {concentration_pct:.1f}%"
        )

        # Frequência de recorrência
        recurring_merchants = len(
            expenses["title"].value_counts()[expenses["title"].value_counts() >= 3]
        )
        insights.append(
            f"Estabelecimentos com recorrência ≥3: {recurring_merchants:.0f}"
        )

        # Coeficiente de variação médio por estabelecimento
        merchant_cv = (
            expenses.groupby("title")["amount"]
            .agg(["mean", "std"])
            .apply(
                lambda x: (x["std"] / x["mean"]) * 100 if x["mean"] > 0 else 0, axis=1
            )
        )
        avg_cv = merchant_cv.mean()
        insights.append(
            f"Coeficiente de variação médio por estabelecimento: {avg_cv:.1f}%"
        )

        # Análise de otimização por percentis
        p75_global = expenses["amount"].quantile(0.75)
        high_value_transactions = expenses[expenses["amount"] > p75_global]
        potential_savings = (high_value_transactions["amount"] - p75_global).sum()
        insights.append(
            f"Oportunidade de otimização (P95→P75): R$ {potential_savings:.2f}"
        )

        # Análise de sazonalidade semanal
        expenses["is_weekend"] = expenses["date"].dt.dayofweek >= 5
        weekend_avg = expenses[expenses["is_weekend"]]["amount"].mean()
        weekday_avg = expenses[~expenses["is_weekend"]]["amount"].mean()
        weekend_premium = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        insights.append(
            f"Coeficiente de sazonalidade semanal: {weekend_premium:+.1f}% (μ_weekend={weekend_avg:.2f}, μ_weekday={weekday_avg:.2f})"
        )

        # Coeficiente de variação temporal
        monthly_totals = expenses.groupby(expenses["date"].dt.to_period("M"))[
            "amount"
        ].sum()
        monthly_cv = (monthly_totals.std() / monthly_totals.mean()) * 100
        insights.append(f"Coeficiente de variação temporal mensal: {monthly_cv:.1f}%")

        # Análise de outliers estatísticos
        mean_amount = expenses["amount"].mean()
        std_amount = expenses["amount"].std()
        outliers = expenses[abs(expenses["amount"] - mean_amount) > 2 * std_amount]
        outlier_pct = (len(outliers) / len(expenses)) * 100
        insights.append(
            f"Outliers estatísticos (|x-μ|>2σ): {len(outliers):.0f} transações ({outlier_pct:.1f}%)"
        )

        # Análise detalhada de outliers
        self.add_section("ANÁLISE DETALHADA DE OUTLIERS", insights)

        # Outliers por método IQR
        Q1 = expenses["amount"].quantile(0.25)
        Q3 = expenses["amount"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_iqr = expenses[
            (expenses["amount"] < lower_bound) | (expenses["amount"] > upper_bound)
        ]
        insights.append(
            f"Outliers por IQR (Q1-1.5*IQR, Q3+1.5*IQR): {len(outliers_iqr):.0f} transações"
        )
        insights.append(f"Limites IQR: R$ {lower_bound:.2f} - R$ {upper_bound:.2f}")

        # Outliers por percentis
        p99 = expenses["amount"].quantile(0.99)
        p95 = expenses["amount"].quantile(0.95)
        outliers_p99 = expenses[expenses["amount"] > p99]
        outliers_p95 = expenses[expenses["amount"] > p95]

        insights.append(
            f"Outliers P99+ (>{p99:.2f}): {len(outliers_p99):.0f} transações"
        )
        insights.append(
            f"Outliers P95+ (>{p95:.2f}): {len(outliers_p95):.0f} transações"
        )

        # Top 10 outliers por valor
        top_outliers = expenses.nlargest(10, "amount")
        insights.append("\nTop 10 outliers por valor:")
        for i, (_, row) in enumerate(top_outliers.iterrows(), 1):
            z_score = (row["amount"] - mean_amount) / std_amount
            insights.append(
                f"{i:2d}. {row['title']}: R$ {row['amount']:.2f} (Z-score: {z_score:.2f}) - {row['date'].strftime('%Y-%m-%d')}"
            )

        # Análise de outliers por estabelecimento
        insights.append("\nEstabelecimentos com mais outliers:")
        merchant_outliers = (
            outliers.groupby("title").size().sort_values(ascending=False).head(10)
        )
        for merchant, count in merchant_outliers.items():
            merchant_data = expenses[expenses["title"] == merchant]
            outlier_ratio = (count / len(merchant_data)) * 100
            insights.append(
                f"  {merchant}: {count:.0f} outliers ({outlier_ratio:.1f}% das transações)"
            )

        # Análise temporal de outliers
        insights.append("\nDistribuição temporal de outliers:")
        outliers_copy = outliers.copy()
        outliers_copy["month"] = outliers_copy["date"].dt.to_period("M")
        monthly_outliers = outliers_copy.groupby("month").size()
        for month, count in monthly_outliers.items():
            total_month = len(expenses[expenses["date"].dt.to_period("M") == month])
            outlier_pct_month = (count / total_month) * 100
            insights.append(
                f"  {month}: {count:.0f} outliers ({outlier_pct_month:.1f}% do mês)"
            )

        # Impacto financeiro dos outliers
        total_outlier_value = outliers["amount"].sum()
        outlier_impact = (total_outlier_value / total_expenses) * 100
        insights.append(f"\nImpacto financeiro dos outliers:")
        insights.append(f"  Valor total: R$ {total_outlier_value:.2f}")
        insights.append(f"  % do total de gastos: {outlier_impact:.1f}%")
        insights.append(
            f"  Valor médio por outlier: R$ {outliers['amount'].mean():.2f}"
        )

        # Análise de séries temporais
        self.add_section("ANÁLISE DE SÉRIES TEMPORAIS", insights)

        # Análise de sazonalidade horária
        if "time" in expenses.columns:
            expenses["hour"] = pd.to_datetime(expenses["time"]).dt.hour
            hourly_spending = expenses.groupby("hour")["amount"].sum()
            peak_hour = hourly_spending.idxmax()
            peak_amount = hourly_spending.max()
            insights.append(
                f"Moda temporal (horário de pico): {peak_hour:02d}h (R$ {peak_amount:.2f})"
            )

        # Análise de sazonalidade mensal
        expenses["day_of_month"] = expenses["date"].dt.day
        daily_spending = expenses.groupby("day_of_month")["amount"].sum()
        expensive_days = daily_spending.nlargest(3)
        insights.append("Máximos locais por dia do mês:")
        for day, amount in expensive_days.items():
            insights.append(f"  Dia {day}: R$ {amount:.2f}")

        # Análise de intervalos inter-transacionais
        expenses_sorted = expenses.sort_values("date")
        expenses_sorted["days_since_last"] = expenses_sorted["date"].diff().dt.days
        avg_days_between = expenses_sorted["days_since_last"].mean()
        insights.append(
            f"Intervalo médio inter-transacional: {avg_days_between:.1f} dias"
        )

        # Análise de sequências consecutivas
        consecutive_days = 0
        max_consecutive = 0
        current_consecutive = 0
        for _, row in expenses_sorted.iterrows():
            if row["days_since_last"] <= 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        insights.append(
            f"Comprimento máximo de sequência consecutiva: {max_consecutive}"
        )

        # Análise de distribuição de valores
        self.add_section("ANÁLISE DE DISTRIBUIÇÃO DE VALORES", insights)

        # Coeficiente de assimetria (skewness)
        median_vs_mean = (expenses["amount"].median() / expenses["amount"].mean()) * 100
        insights.append(
            f"Coeficiente de eficiência (mediana/média): {median_vs_mean:.1f}%"
        )

        # Distribuição por intervalos de classe
        ranges = [
            (0, 20, "Classe I"),
            (20, 50, "Classe II"),
            (50, 100, "Classe III"),
            (100, float("inf"), "Classe IV"),
        ]
        insights.append("Distribuição por intervalos de classe:")
        for min_val, max_val, label in ranges:
            if max_val == float("inf"):
                count = len(expenses[expenses["amount"] >= min_val])
                total = expenses[expenses["amount"] >= min_val]["amount"].sum()
            else:
                count = len(
                    expenses[
                        (expenses["amount"] >= min_val) & (expenses["amount"] < max_val)
                    ]
                )
                total = expenses[
                    (expenses["amount"] >= min_val) & (expenses["amount"] < max_val)
                ]["amount"].sum()

            pct_count = (count / len(expenses)) * 100
            pct_value = (total / total_expenses) * 100
            insights.append(
                f"  {label}: {count:.0f} ({pct_count:.1f}%) - R$ {total:.2f} ({pct_value:.1f}%)"
            )

        # Análise de tendências temporais
        self.add_section("ANÁLISE DE TENDÊNCIAS TEMPORAIS", insights)

        # Coeficiente de tendência por estabelecimento
        merchant_trends = []
        for merchant in expenses["title"].value_counts().head(10).index:
            merchant_data = expenses[expenses["title"] == merchant].sort_values("date")
            if len(merchant_data) >= 3:
                first_half = merchant_data.iloc[: len(merchant_data) // 2][
                    "amount"
                ].mean()
                second_half = merchant_data.iloc[len(merchant_data) // 2 :][
                    "amount"
                ].mean()
                trend = ((second_half - first_half) / first_half) * 100
                merchant_trends.append((merchant, trend))

        merchant_trends.sort(key=lambda x: x[1], reverse=True)
        insights.append("Coeficientes de tendência por estabelecimento (top 5):")
        for merchant, trend in merchant_trends[:5]:
            insights.append(f"  {merchant}: {trend:+.1f}%")

        # Análise de sazonalidade mensal
        monthly_analysis = expenses.groupby(expenses["date"].dt.month).agg(
            {"amount": ["sum", "count", "mean"]}
        )
        best_month = monthly_analysis[("amount", "sum")].idxmax()
        worst_month = monthly_analysis[("amount", "sum")].idxmin()
        insights.append(
            f"Máximo mensal: {best_month} (R$ {monthly_analysis.loc[best_month, ('amount', 'sum')]:.2f})"
        )
        insights.append(
            f"Mínimo mensal: {worst_month} (R$ {monthly_analysis.loc[worst_month, ('amount', 'sum')]:.2f})"
        )

        # Métricas de previsibilidade estatística
        self.add_section("MÉTRICAS DE PREVISIBILIDADE ESTATÍSTICA", insights)

        # Coeficiente de previsibilidade por estabelecimento
        predictable_merchants = 0
        for merchant in expenses["title"].value_counts().head(20).index:
            merchant_data = expenses[expenses["title"] == merchant]["amount"]
            if len(merchant_data) >= 5:
                cv = (merchant_data.std() / merchant_data.mean()) * 100
                if cv < 30:  # Baixa variabilidade = previsível
                    predictable_merchants += 1

        insights.append(
            f"Estabelecimentos com baixa variabilidade (CV<30%): {predictable_merchants}/20"
        )

        # Coeficiente de previsibilidade temporal
        daily_totals = expenses.groupby(expenses["date"].dt.date)["amount"].sum()
        daily_cv = (daily_totals.std() / daily_totals.mean()) * 100
        insights.append(
            f"Coeficiente de previsibilidade temporal diária: {daily_cv:.1f}%"
        )

        return "\n".join(insights)


    def generate_complete_analysis(self) -> str:
        """Gera análise completa de todas as transações"""
        insights = []

        start_date = self.df["date"].min()
        end_date = self.df["date"].max()
        total_transactions = len(self.df)
        total_expenses = self.df_expenses["amount"].sum()
        total_profits = self.df_profits["amount"].sum()
        net_balance = total_profits - total_expenses

        self.add_section("ANÁLISE COMPLETA DE TRANSAÇÕES", insights)
        insights.append(
            f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}"
        )
        insights.append(f"Total de dias: {(end_date - start_date).days + 1}")
        insights.append(f"Total de transações: {total_transactions:,}")
        insights.append(f"Gastos: {len(self.df_expenses):,} transações")
        insights.append(f"Recebimentos: {len(self.df_profits):,} transações")
        insights.append("")

        self.add_section("RESUMO FINANCEIRO", insights)
        insights.append(f"Total de gastos: R$ {total_expenses:,.2f}")
        insights.append(f"Total de recebimentos: R$ {total_profits:,.2f}")
        insights.append(f"Saldo líquido: R$ {net_balance:,.2f}")
        insights.append(
            f"Gasto médio diário: R$ {total_expenses / ((end_date - start_date).days + 1):.2f}"
        )
        insights.append(f"Recebimento médio mensal: R$ {total_profits / 5:,.2f}")
        insights.append("")

        # Análise de fluxo de caixa
        self.add_section("ANÁLISE DE FLUXO DE CAIXA", insights)

        # Por mês
        monthly_expenses = self.df_expenses.groupby(
            self.df_expenses["date"].dt.to_period("M")
        )["amount"].sum()
        monthly_profits = self.df_profits.groupby(
            self.df_profits["date"].dt.to_period("M")
        )["amount"].sum()

        insights.append("\nFluxo de caixa mensal:")
        for month in monthly_expenses.index:
            expenses_month = monthly_expenses.get(month, 0)
            profits_month = monthly_profits.get(month, 0)
            balance_month = profits_month - expenses_month

            insights.append(f"  {month}:")
            insights.append(f"    Recebimentos: R$ {profits_month:,.2f}")
            insights.append(f"    Gastos: R$ {expenses_month:,.2f}")
            insights.append(f"    Saldo: R$ {balance_month:,.2f}")

        # Análise de tendências
        self.add_section("ANÁLISE DE TENDÊNCIAS", insights)

        if len(monthly_expenses) > 1:
            expense_trend = monthly_expenses.iloc[-1] - monthly_expenses.iloc[0]
            expense_trend_pct = (expense_trend / monthly_expenses.iloc[0]) * 100
            insights.append(
                f"Tendência de gastos: {'+' if expense_trend > 0 else ''}R$ {expense_trend:.2f} ({expense_trend_pct:+.1f}%)"
            )

        if len(monthly_profits) > 1:
            profit_trend = monthly_profits.iloc[-1] - monthly_profits.iloc[0]
            profit_trend_pct = (profit_trend / monthly_profits.iloc[0]) * 100
            insights.append(
                f"Tendência de recebimentos: {'+' if profit_trend > 0 else ''}R$ {profit_trend:.2f} ({profit_trend_pct:+.1f}%)"
            )

        # Análise de estabelecimentos mais frequentes
        self.add_section("ESTABELECIMENTOS MAIS FREQUENTES", insights)

        # Top 10 estabelecimentos por frequência
        most_frequent = self.df["title"].value_counts().head(10)
        insights.append("\nTop 10 por frequência de transações:")
        for i, (merchant, count) in enumerate(most_frequent.items(), 1):
            insights.append(f"{i:2d}. {merchant}: {count} transações")

        # Top 10 estabelecimentos por valor total
        top_by_value = (
            self.df.groupby("title")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
        insights.append("\nTop 10 por valor total (absoluto):")
        for i, (merchant, value) in enumerate(top_by_value.items(), 1):
            insights.append(f"{i:2d}. {merchant}: R$ {value:,.2f}")

        # Métricas de saúde financeira
        self.add_section("MÉTRICAS DE SAÚDE FINANCEIRA", insights)

        # Taxa de poupança
        if total_profits > 0:
            savings_rate = (net_balance / total_profits) * 100
            insights.append(f"Taxa de poupança: {savings_rate:.1f}%")

        # Razão gastos/recebimentos
        if total_profits > 0:
            expense_ratio = (total_expenses / total_profits) * 100
            insights.append(f"Razão gastos/recebimentos: {expense_ratio:.1f}%")

        # Consistência de recebimentos
        if len(monthly_profits) > 1:
            profit_consistency = (monthly_profits.std() / monthly_profits.mean()) * 100
            insights.append(
                f"Consistência de recebimentos: {profit_consistency:.1f}% (menor = mais consistente)"
            )

        # Consistência de gastos
        if len(monthly_expenses) > 1:
            expense_consistency = (
                monthly_expenses.std() / monthly_expenses.mean()
            ) * 100
            insights.append(
                f"Consistência de gastos: {expense_consistency:.1f}% (menor = mais consistente)"
            )

        # Análise de sazonalidade
        self.add_section("ANÁLISE DE SAZONALIDADE", insights)

        # Por dia da semana
        self.df["day_of_week"] = self.df["date"].dt.day_name()
        daily_analysis = (
            self.df.groupby(["day_of_week", self.df["amount"] > 0])
            .agg({"amount": ["sum", "count", "mean"]})
            .round(2)
        )

        insights.append("\nAtividade por dia da semana:")
        for day in [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]:
            try:
                expenses_day = daily_analysis.loc[(day, True), ("amount", "sum")]
                profits_day = daily_analysis.loc[(day, False), ("amount", "sum")]
                insights.append(
                    f"  {day}: Gastos R$ {expenses_day:,.2f}, Recebimentos R$ {profits_day:,.2f}"
                )
            except KeyError:
                continue

        # Métricas de eficiência estatística
        self.add_section("MÉTRICAS DE EFICIÊNCIA ESTATÍSTICA", insights)

        # Taxa de gastos temporal
        total_days = (end_date - start_date).days + 1
        spending_velocity = total_expenses / total_days
        insights.append(f"Taxa de gastos temporal: R$ {spending_velocity:.2f}/dia")

        # Valor esperado por transação
        avg_transaction_value = total_expenses / len(self.df_expenses)
        insights.append(f"Valor esperado por transação: R$ {avg_transaction_value:.2f}")

        # Moda de estabelecimentos
        most_frequent_merchant = self.df_expenses["title"].value_counts().iloc[0]
        most_frequent_name = self.df_expenses["title"].value_counts().index[0]
        insights.append(
            f"Moda de estabelecimentos: {most_frequent_name} (frequência={most_frequent_merchant})"
        )

        # Índice de diversificação de Shannon
        total_merchants = self.df_expenses["title"].nunique()
        diversification_index = total_merchants / len(self.df_expenses)
        insights.append(
            f"Índice de diversificação de Shannon: {diversification_index:.3f}"
        )

        # Coeficiente de variação diária
        daily_expenses = self.df_expenses.groupby(self.df_expenses["date"].dt.date)[
            "amount"
        ].sum()
        spending_consistency = (daily_expenses.std() / daily_expenses.mean()) * 100
        insights.append(f"Coeficiente de variação diária: {spending_consistency:.1f}%")

        # Análise de cauda da distribuição
        same_day_multiple = len(
            daily_expenses[daily_expenses > daily_expenses.quantile(0.8)]
        )
        impulse_days = (same_day_multiple / len(daily_expenses)) * 100
        insights.append(
            f"Observações na cauda superior (P80+): {same_day_multiple:.0f} dias ({impulse_days:.1f}%)"
        )

        # Coeficiente de sazonalidade temporal
        weekend_expenses = self.df_expenses[self.df_expenses["date"].dt.dayofweek >= 5][
            "amount"
        ].sum()
        weekday_expenses = self.df_expenses[self.df_expenses["date"].dt.dayofweek < 5][
            "amount"
        ].sum()
        weekend_ratio = weekend_expenses / (weekend_expenses + weekday_expenses) * 100
        insights.append(f"Coeficiente de sazonalidade temporal: {weekend_ratio:.1f}%")

        # Métricas de risco estatístico
        self.add_section("MÉTRICAS DE RISCO ESTATÍSTICO", insights)

        # Desvio padrão temporal
        daily_volatility = (
            self.df_expenses.groupby(self.df_expenses["date"].dt.date)["amount"]
            .sum()
            .std()
        )
        insights.append(f"Desvio padrão temporal diário: R$ {daily_volatility:.2f}")

        # Análise de Value at Risk (VaR)
        high_spending_days = len(
            daily_expenses[daily_expenses > daily_expenses.quantile(0.9)]
        )
        risk_days = (high_spending_days / len(daily_expenses)) * 100
        insights.append(
            f"Value at Risk (P90): {high_spending_days:.0f} dias ({risk_days:.1f}%)"
        )

        # Coeficiente de concentração de risco
        top_merchant_risk = self.df_expenses.groupby("title")["amount"].sum().max()
        risk_concentration = (top_merchant_risk / total_expenses) * 100
        insights.append(
            f"Coeficiente de concentração de risco: {risk_concentration:.1f}%"
        )

        # Métricas de otimização matemática
        self.add_section("MÉTRICAS DE OTIMIZAÇÃO MATEMÁTICA", insights)

        # Função de economia por consolidação
        small_transactions = self.df_expenses[self.df_expenses["amount"] <= 20]
        consolidation_savings = (
            len(small_transactions) * 2
        )  # Economia estimada por consolidação
        insights.append(
            f"Função de economia por consolidação (≤R$ 20): R$ {consolidation_savings:.2f}"
        )

        # Coeficiente de redução de frequência
        frequent_merchants = self.df_expenses["title"].value_counts()
        top_frequent = frequent_merchants.head(5)
        avg_frequency_reduction = top_frequent.mean() * 0.1  # 10% de redução
        insights.append(
            f"Coeficiente de redução de frequência (top 5): {avg_frequency_reduction:.1f} transações/mês"
        )

        # Análise de clustering por categorias implícitas
        self.add_section("ANÁLISE DE CLUSTERING POR CATEGORIAS IMPLÍCITAS", insights)

        # Identificar padrões por palavras-chave
        keywords = {
            "Transporte": ["uber", "taxi", "99"],
            "Alimentação": ["ifood", "rappi", "restaurante", "lanchonete"],
            "Supermercado": ["carrefour", "walmart", "extra"],
            "Tecnologia": ["amazon", "apple", "google", "netflix"],
            "Lazer": ["cinema", "teatro", "hotel", "viagem"],
        }

        for category, words in keywords.items():
            category_expenses = self.df_expenses[
                self.df_expenses["title"]
                .str.lower()
                .str.contains("|".join(words), na=False)
            ]
            if len(category_expenses) > 0:
                total_cat = category_expenses["amount"].sum()
                pct_cat = (total_cat / total_expenses) * 100
                insights.append(
                    f"Cluster {category}: R$ {total_cat:.2f} ({pct_cat:.1f}%) - {len(category_expenses)} transações"
                )

        # Métricas de estabilidade financeira
        self.add_section("MÉTRICAS DE ESTABILIDADE FINANCEIRA", insights)

        # Coeficiente de estabilidade temporal
        monthly_totals = self.df_expenses.groupby(
            self.df_expenses["date"].dt.to_period("M")
        )["amount"].sum()
        monthly_consistency = (monthly_totals.std() / monthly_totals.mean()) * 100
        if monthly_consistency < 20:
            stability = "Alta"
        elif monthly_consistency < 40:
            stability = "Média"
        else:
            stability = "Baixa"
        insights.append(
            f"Coeficiente de estabilidade temporal: {stability} ({monthly_consistency:.1f}%)"
        )

        # Coeficiente de previsibilidade
        predictable_expenses = 0
        for merchant in self.df_expenses["title"].value_counts().head(10).index:
            merchant_data = self.df_expenses[self.df_expenses["title"] == merchant][
                "amount"
            ]
            if len(merchant_data) >= 3:
                cv = (merchant_data.std() / merchant_data.mean()) * 100
                if cv < 25:
                    predictable_expenses += 1

        predictability = (predictable_expenses / 10) * 100
        insights.append(f"Coeficiente de previsibilidade: {predictability:.1f}%")

        # Coeficiente de controle estatístico
        impulse_control = (
            len(daily_expenses[daily_expenses <= daily_expenses.median()])
            / len(daily_expenses)
        ) * 100
        insights.append(
            f"Coeficiente de controle estatístico (≤mediana): {impulse_control:.1f}%"
        )

        return "\n".join(insights)


def main():
    """Função principal para executar a análise"""
    try:
        analyzer = NubankAnalyzer()

        # Gerar análises separadas
        print("Gerando análise de gastos...")
        expenses_insights = analyzer.generate_expenses_analysis()
        expenses_insights_no_fixed = analyzer.generate_expenses_analysis(
            no_include=["uber", "spotify", "discord"]
        )

        print("Gerando análise completa...")
        complete_insights = analyzer.generate_complete_analysis()

        # Salvar em arquivos separados
        with open("gastos_analysis.txt", "w", encoding="utf-8") as f:
            f.write(expenses_insights)

        with open("analise_completa.txt", "w", encoding="utf-8") as f:
            f.write(complete_insights)

        with open("gastos_analysis_no_fixed.txt", "w", encoding="utf-8") as f:
            f.write(expenses_insights_no_fixed)

        print("\n" + "=" * 60)
        print("Análises salvas em:")
        print("  - gastos_analysis.txt")
        print("  - analise_completa.txt")
        print("=" * 60)

        # Mostrar resumo
        print("\nRESUMO EXECUTIVO:")
        print(f"Total de gastos: R$ {analyzer.df_expenses['amount'].sum():,.2f}")
        print(f"Total de recebimentos: R$ {analyzer.df_profits['amount'].sum():,.2f}")
        print(
            f"Saldo líquido: R$ {analyzer.df_profits['amount'].sum() - analyzer.df_expenses['amount'].sum():,.2f}"
        )

    except Exception as e:
        print(f"Erro na análise: {e}")


if __name__ == "__main__":
    main()
