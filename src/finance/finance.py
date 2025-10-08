import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from collections import defaultdict, Counter

class NubankAnalyzer:
    def __init__(self, data_folder: str = "secret"):
        self.data_folder = data_folder
        self.df = None
        self.df_profits = None
        self.df_expenses = None
        self.load_data()
        self.separate_data()
    
    def load_data(self):
        """Carrega todos os arquivos CSV do Nubank"""
        csv_files = list(Path(self.data_folder).glob("Nubank_*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {self.data_folder}")
        
        dfs = []
        for file in csv_files:
            df = pd.read_csv(file)
            df['file_source'] = file.name
            dfs.append(df)
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Converter data e amount
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['amount'] = pd.to_numeric(self.df['amount'])
        
        # Ordenar por data
        self.df = self.df.sort_values('date')

        self.df_profits = self.df[self.df['amount'] > 0]
        self.df_expenses = self.df[self.df['amount'] < 0]
        
        print(f"Carregados {len(self.df)} transações de {len(csv_files)} arquivos")
    
    
    def separate_data(self):
        """Separa os dados em gastos e recebimentos"""
        self.df_expenses = self.df[self.df['amount'] > 0].copy()
        self.df_profits = self.df[self.df['amount'] < 0].copy()
        
        # Converter valores negativos para positivos nos recebimentos
        self.df_profits['amount'] = abs(self.df_profits['amount'])
        
        print(f"Gastos: {len(self.df_expenses)} transações")
        print(f"Recebimentos: {len(self.df_profits)} transações")
    
    def generate_insights(self) -> str:
        """Gera insights financeiros completos"""
        insights = []
        
        # Informações básicas
        insights.append("=== ANÁLISE FINANCEIRA NUBANK ===\n")
        
        # Período de análise
        start_date = self.df['date'].min()
        end_date = self.df['date'].max()
        insights.append(f"Período analisado: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        insights.append(f"Total de dias: {(end_date - start_date).days + 1}")
        insights.append(f"Total de transações: {len(self.df):,}")
        insights.append("")
        
        # Separar gastos e recebimentos
        expenses = self.df[self.df['amount'] > 0]
        income = self.df[self.df['amount'] < 0]
        
        total_expenses = expenses['amount'].sum()
        total_income = abs(income['amount'].sum())
        net_balance = total_income - total_expenses
        
        avg_daily_expenses = total_expenses / ((end_date - start_date).days + 1)
        avg_monthly_expenses = total_expenses / 5  # 5 meses de dados
        
        insights.append("=== RESUMO FINANCEIRO ===")
        insights.append(f"Total de gastos: R$ {total_expenses:,.2f}")
        insights.append(f"Total de recebimentos: R$ {total_income:,.2f}")
        insights.append(f"Saldo líquido: R$ {net_balance:,.2f}")
        insights.append(f"Gasto médio diário: R$ {avg_daily_expenses:.2f}")
        insights.append(f"Gasto médio mensal: R$ {avg_monthly_expenses:,.2f}")
        insights.append(f"Valor médio por gasto: R$ {expenses['amount'].mean():.2f}")
        insights.append(f"Valor médio por recebimento: R$ {abs(income['amount'].mean()):.2f}")
        insights.append(f"Maior gasto único: R$ {expenses['amount'].max():.2f}")
        insights.append(f"Maior recebimento único: R$ {abs(income['amount'].min()):.2f}")
        insights.append("")
        
        # Análise por categoria - apenas gastos
        insights.append("=== GASTOS POR CATEGORIA ===")
        expenses_by_category = expenses.groupby('category').agg({
            'amount': ['sum', 'count', 'mean'],
            'date': 'nunique'
        }).round(2)
        
        expenses_by_category.columns = ['Total_Gasto', 'Qtd_Transacoes', 'Valor_Medio', 'Dias_Com_Gasto']
        expenses_by_category = expenses_by_category.sort_values('Total_Gasto', ascending=False)
        
        for category, row in expenses_by_category.iterrows():
            percentage = (row['Total_Gasto'] / total_expenses) * 100
            insights.append(f"{category}:")
            insights.append(f"  - Total: R$ {row['Total_Gasto']:,.2f} ({percentage:.1f}%)")
            insights.append(f"  - Transações: {row['Qtd_Transacoes']:.0f}")
            insights.append(f"  - Valor médio: R$ {row['Valor_Medio']:.2f}")
            insights.append(f"  - Dias com gasto: {row['Dias_Com_Gasto']:.0f}")
            insights.append("")
        
        # Análise de recebimentos
        insights.append("=== RECEBIMENTOS ===")
        if len(income) > 0:
            income_by_category = income.groupby('category').agg({
                'amount': ['sum', 'count', 'mean'],
                'date': 'nunique'
            }).round(2)
            
            income_by_category.columns = ['Total_Recebido', 'Qtd_Transacoes', 'Valor_Medio', 'Dias_Com_Recebimento']
            income_by_category = income_by_category.sort_values('Total_Recebido', ascending=True)  # Mais negativo primeiro
            
            for category, row in income_by_category.iterrows():
                total_received = abs(row['Total_Recebido'])
                percentage = (total_received / total_income) * 100
                insights.append(f"{category}:")
                insights.append(f"  - Total: R$ {total_received:,.2f} ({percentage:.1f}%)")
                insights.append(f"  - Transações: {row['Qtd_Transacoes']:.0f}")
                insights.append(f"  - Valor médio: R$ {abs(row['Valor_Medio']):.2f}")
                insights.append(f"  - Dias com recebimento: {row['Dias_Com_Recebimento']:.0f}")
                insights.append("")
        else:
            insights.append("Nenhum recebimento identificado no período.")
            insights.append("")
        
        # Análise temporal
        insights.append("=== ANÁLISE TEMPORAL ===")
        
        # Por mês - separando gastos e recebimentos
        monthly_expenses = expenses.groupby(expenses['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count']
        }).round(2)
        
        monthly_income = income.groupby(income['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count']
        }).round(2)
        
        insights.append("Gastos por mês:")
        for month, row in monthly_expenses.iterrows():
            insights.append(f"  {month}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações)")
        
        insights.append("")
        insights.append("Recebimentos por mês:")
        if len(monthly_income) > 0:
            for month, row in monthly_income.iterrows():
                insights.append(f"  {month}: R$ {abs(row[('amount', 'sum')]):,.2f} ({row[('amount', 'count')]:.0f} transações)")
        else:
            insights.append("  Nenhum recebimento identificado por mês.")
        
        insights.append("")
        
        # Por dia da semana - apenas gastos
        expenses['day_of_week'] = expenses['date'].dt.day_name()
        daily_avg_expenses = expenses.groupby('day_of_week')['amount'].mean().round(2)
        insights.append("Gasto médio por dia da semana:")
        for day, amount in daily_avg_expenses.items():
            insights.append(f"  {day}: R$ {amount:.2f}")
        
        insights.append("")
        
        # Top estabelecimentos - apenas gastos
        insights.append("=== TOP ESTABELECIMENTOS (GASTOS) ===")
        top_merchants = expenses.groupby('title').agg({
            'amount': ['sum', 'count'],
            'date': 'nunique'
        }).round(2)
        
        top_merchants.columns = ['Total_Gasto', 'Qtd_Transacoes', 'Dias_Com_Gasto']
        top_merchants = top_merchants.sort_values('Total_Gasto', ascending=False).head(10)
        
        for merchant, row in top_merchants.iterrows():
            insights.append(f"{merchant}:")
            insights.append(f"  - Total: R$ {row['Total_Gasto']:,.2f}")
            insights.append(f"  - Visitas: {row['Qtd_Transacoes']:.0f}")
            insights.append(f"  - Dias: {row['Dias_Com_Gasto']:.0f}")
        
        insights.append("")
        
        # Top recebimentos
        if len(income) > 0:
            insights.append("=== TOP RECEBIMENTOS ===")
            top_income = income.groupby('title').agg({
                'amount': ['sum', 'count'],
                'date': 'nunique'
            }).round(2)
            
            top_income.columns = ['Total_Recebido', 'Qtd_Transacoes', 'Dias_Com_Recebimento']
            top_income = top_income.sort_values('Total_Recebido', ascending=True).head(10)  # Mais negativo primeiro
            
            for merchant, row in top_income.iterrows():
                total_received = abs(row['Total_Recebido'])
                insights.append(f"{merchant}:")
                insights.append(f"  - Total: R$ {total_received:,.2f}")
                insights.append(f"  - Transações: {row['Qtd_Transacoes']:.0f}")
                insights.append(f"  - Dias: {row['Dias_Com_Recebimento']:.0f}")
            
            insights.append("")
        
        # Padrões de gasto
        insights.append("=== PADRÕES DE GASTO ===")
        
        # Frequência de transações
        transaction_frequency = self.df.groupby(self.df['date'].dt.date).size()
        insights.append(f"Dias com transações: {len(transaction_frequency)} de {(end_date - start_date).days + 1}")
        insights.append(f"Frequência média: {transaction_frequency.mean():.1f} transações/dia")
        insights.append(f"Dia com mais transações: {transaction_frequency.max()} transações")
        
        # Análise de valores
        insights.append("")
        insights.append("Distribuição de valores:")
        insights.append(f"  - Até R$ 10: {len(self.df[self.df['amount'] <= 10]):.0f} transações")
        insights.append(f"  - R$ 10-50: {len(self.df[(self.df['amount'] > 10) & (self.df['amount'] <= 50)]):.0f} transações")
        insights.append(f"  - R$ 50-100: {len(self.df[(self.df['amount'] > 50) & (self.df['amount'] <= 100)]):.0f} transações")
        insights.append(f"  - Acima de R$ 100: {len(self.df[self.df['amount'] > 100]):.0f} transações")
        
        # Insights comportamentais
        insights.append("")
        insights.append("=== INSIGHTS COMPORTAMENTAIS ===")
        
        # Categoria dominante
        top_category = category_summary.index[0]
        top_category_pct = (category_summary.iloc[0]['Total_Gasto'] / total_spent) * 100
        insights.append(f"• Categoria dominante: {top_category} ({top_category_pct:.1f}% dos gastos)")
        
        # Estabelecimento mais frequente
        most_frequent = self.df['title'].value_counts().index[0]
        most_frequent_count = self.df['title'].value_counts().iloc[0]
        insights.append(f"• Estabelecimento mais visitado: {most_frequent} ({most_frequent_count} vezes)")
        
        # Tendência mensal
        monthly_totals = monthly[('amount', 'sum')]
        if len(monthly_totals) > 1:
            trend = monthly_totals.iloc[-1] - monthly_totals.iloc[0]
            trend_pct = (trend / monthly_totals.iloc[0]) * 100
            insights.append(f"• Tendência de gastos: {'+' if trend > 0 else ''}R$ {trend:.2f} ({trend_pct:+.1f}%)")
        
        # Recomendações
        insights.append("")
        insights.append("=== RECOMENDAÇÕES ===")
        
        if top_category_pct > 40:
            insights.append(f"• Atenção: {top_category} representa mais de 40% dos gastos")
        
        if avg_daily > 50:
            insights.append(f"• Gasto diário alto (R$ {avg_daily:.2f}). Considere revisar hábitos")
        
        if len(self.df[self.df['amount'] <= 5]) > len(self.df) * 0.3:
            insights.append("• Muitas transações pequenas. Considere consolidar compras")
        
        # Identificar gastos recorrentes
        recurring = self.df['title'].value_counts()
        recurring_high = recurring[recurring >= 5]
        if len(recurring_high) > 0:
            insights.append("• Gastos recorrentes identificados:")
            for merchant, count in recurring_high.head(3).items():
                insights.append(f"  - {merchant}: {count} vezes")
        
        return "\n".join(insights)
    
    def generate_expenses_analysis(self) -> str:
        """Gera análise detalhada dos gastos"""
        insights = []
        expenses = self.df_expenses
        
        if len(expenses) == 0:
            return "Nenhum gasto encontrado no período."
        
        start_date = expenses['date'].min()
        end_date = expenses['date'].max()
        total_expenses = expenses['amount'].sum()
        
        insights.append("=== ANÁLISE DETALHADA DE GASTOS ===\n")
        insights.append(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        insights.append(f"Total de gastos: {len(expenses):,} transações")
        insights.append(f"Valor total gasto: R$ {total_expenses:,.2f}")
        insights.append(f"Gasto médio diário: R$ {total_expenses / ((end_date - start_date).days + 1):.2f}")
        insights.append(f"Valor médio por gasto: R$ {expenses['amount'].mean():.2f}")
        insights.append(f"Mediana dos gastos: R$ {expenses['amount'].median():.2f}")
        insights.append(f"Maior gasto: R$ {expenses['amount'].max():.2f}")
        insights.append(f"Menor gasto: R$ {expenses['amount'].min():.2f}")
        insights.append("")
        
        # Análise de percentis
        insights.append("=== ANÁLISE DE PERCENTIS ===")
        percentis = [10, 25, 50, 75, 90, 95, 99]
        for p in percentis:
            valor = expenses['amount'].quantile(p/100)
            insights.append(f"Percentil {p}%: R$ {valor:.2f}")
        insights.append("")
        
        # Top estabelecimentos (análise principal)
        insights.append("=== TOP 30 ESTABELECIMENTOS ===")
        top_merchants = expenses.groupby('title').agg({
            'amount': ['sum', 'count', 'mean', 'median', 'std'],
            'date': 'nunique'
        }).round(2)
        
        top_merchants.columns = ['Total', 'Qtd_Transacoes', 'Media', 'Mediana', 'Desvio_Padrao', 'Dias_Com_Gasto']
        top_merchants = top_merchants.sort_values('Total', ascending=False).head(30)
        
        for i, (merchant, row) in enumerate(top_merchants.iterrows(), 1):
            percentage = (row['Total'] / total_expenses) * 100
            
            # Calcular percentis para este estabelecimento
            merchant_data = expenses[expenses['title'] == merchant]['amount']
            p75 = merchant_data.quantile(0.75)
            p90 = merchant_data.quantile(0.90)
            p95 = merchant_data.quantile(0.95)
            
            insights.append(f"\n{i:2d}. {merchant}:")
            insights.append(f"     Total: R$ {row['Total']:,.2f} ({percentage:.1f}%)")
            insights.append(f"     Transações: {row['Qtd_Transacoes']:.0f}")
            insights.append(f"     Valor médio: R$ {row['Media']:.2f} ({row['Desvio_Padrao']:.0f})")
            insights.append(f"     Mediana: R$ {row['Mediana']:.2f}")
            insights.append(f"     P75: R$ {p75:.2f} | P90: R$ {p90:.2f} | P95: R$ {p95:.2f}")
            insights.append(f"     Dias com gasto: {row['Dias_Com_Gasto']:.0f}")
        
        # Análise temporal
        insights.append("\n=== ANÁLISE TEMPORAL DE GASTOS ===")
        
        # Por mês
        monthly_expenses = expenses.groupby(expenses['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count', 'mean'],
            'date': 'nunique'
        }).round(2)
        
        insights.append("\nGastos por mês:")
        for month, row in monthly_expenses.iterrows():
            insights.append(f"  {month}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})")
        
        # Por dia da semana
        expenses['day_of_week'] = expenses['date'].dt.day_name()
        daily_analysis = expenses.groupby('day_of_week').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        insights.append("\nGastos por dia da semana:")
        for day, row in daily_analysis.iterrows():
            insights.append(f"  {day}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})")
        
        # Por hora (se disponível)
        if 'time' in expenses.columns:
            expenses['hour'] = pd.to_datetime(expenses['time']).dt.hour
            hourly_analysis = expenses.groupby('hour').agg({
                'amount': ['sum', 'count', 'mean']
            }).round(2)
            
            insights.append("\nGastos por hora do dia:")
            for hour, row in hourly_analysis.iterrows():
                insights.append(f"  {hour:02d}h: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações)")
        
        
        # Análise de frequência
        insights.append("\n=== ANÁLISE DE FREQUÊNCIA ===")
        expense_frequency = expenses.groupby(expenses['date'].dt.date).size()
        insights.append(f"Dias com gastos: {len(expense_frequency)} de {(end_date - start_date).days + 1}")
        insights.append(f"Frequência média: {expense_frequency.mean():.1f} transações/dia")
        insights.append(f"Dia com mais gastos: {expense_frequency.max()} transações")
        insights.append(f"Dias sem gastos: {(end_date - start_date).days + 1 - len(expense_frequency)}")
        
        # Distribuição de valores
        insights.append("\n=== DISTRIBUIÇÃO DE VALORES ===")
        value_ranges = [
            (0, 10, "Até R$ 10"),
            (10, 25, "R$ 10-25"),
            (25, 50, "R$ 25-50"),
            (50, 100, "R$ 50-100"),
            (100, 200, "R$ 100-200"),
            (200, 500, "R$ 200-500"),
            (500, float('inf'), "Acima de R$ 500")
        ]
        
        for min_val, max_val, label in value_ranges:
            if max_val == float('inf'):
                count = len(expenses[expenses['amount'] >= min_val])
                total_range = expenses[expenses['amount'] >= min_val]['amount'].sum()
            else:
                count = len(expenses[(expenses['amount'] >= min_val) & (expenses['amount'] < max_val)])
                total_range = expenses[(expenses['amount'] >= min_val) & (expenses['amount'] < max_val)]['amount'].sum()
            
            percentage = (count / len(expenses)) * 100
            value_percentage = (total_range / total_expenses) * 100
            insights.append(f"  {label}: {count:.0f} transações ({percentage:.1f}%) - R$ {total_range:,.2f} ({value_percentage:.1f}%)")
        
        # Estatísticas avançadas
        insights.append("\n=== ESTATÍSTICAS AVANÇADAS ===")
        insights.append(f"Desvio padrão: R$ {expenses['amount'].std():.2f}")
        insights.append(f"Variância: R$ {expenses['amount'].var():.2f}")
        insights.append(f"Coeficiente de variação: {(expenses['amount'].std() / expenses['amount'].mean()) * 100:.1f}%")
        insights.append(f"Quartil 25%: R$ {expenses['amount'].quantile(0.25):.2f}")
        insights.append(f"Quartil 75%: R$ {expenses['amount'].quantile(0.75):.2f}")
        insights.append(f"Amplitude interquartil: R$ {expenses['amount'].quantile(0.75) - expenses['amount'].quantile(0.25):.2f}")
        
        return "\n".join(insights)
    
    def generate_profits_analysis(self) -> str:
        """Gera análise detalhada dos recebimentos"""
        insights = []
        profits = self.df_profits
        
        if len(profits) == 0:
            return "Nenhum recebimento encontrado no período."
        
        start_date = profits['date'].min()
        end_date = profits['date'].max()
        total_profits = profits['amount'].sum()
        
        insights.append("=== ANÁLISE DETALHADA DE RECEBIMENTOS ===\n")
        insights.append(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        insights.append(f"Total de recebimentos: {len(profits):,} transações")
        insights.append(f"Valor total recebido: R$ {total_profits:,.2f}")
        insights.append(f"Recebimento médio mensal: R$ {total_profits / 5:,.2f}")
        insights.append(f"Valor médio por recebimento: R$ {profits['amount'].mean():.2f}")
        insights.append(f"Mediana dos recebimentos: R$ {profits['amount'].median():.2f}")
        insights.append(f"Maior recebimento: R$ {profits['amount'].max():.2f}")
        insights.append(f"Menor recebimento: R$ {profits['amount'].min():.2f}")
        insights.append("")
        
        # Análise de percentis
        insights.append("=== ANÁLISE DE PERCENTIS ===")
        percentis = [10, 25, 50, 75, 90, 95, 99]
        for p in percentis:
            valor = profits['amount'].quantile(p/100)
            insights.append(f"Percentil {p}%: R$ {valor:.2f}")
        insights.append("")
        
        # Top fontes de recebimento (análise principal)
        insights.append("=== TODAS AS FONTES DE RECEBIMENTO ===")
        top_sources = profits.groupby('title').agg({
            'amount': ['sum', 'count', 'mean', 'median', 'std'],
            'date': 'nunique'
        }).round(2)
        
        top_sources.columns = ['Total', 'Qtd_Transacoes', 'Media', 'Mediana', 'Desvio_Padrao', 'Dias_Com_Recebimento']
        top_sources = top_sources.sort_values('Total', ascending=False)
        
        for i, (source, row) in enumerate(top_sources.iterrows(), 1):
            percentage = (row['Total'] / total_profits) * 100
            insights.append(f"\n{i}. {source}:")
            insights.append(f"   Total: R$ {row['Total']:,.2f} ({percentage:.1f}%)")
            insights.append(f"   Transações: {row['Qtd_Transacoes']:.0f}")
            insights.append(f"   Valor médio: R$ {row['Media']:.2f} ({row['Desvio_Padrao']:.0f})")
            insights.append(f"   Mediana: R$ {row['Mediana']:.2f}")
            insights.append(f"   Dias com recebimento: {row['Dias_Com_Recebimento']:.0f}")
        
        # Análise temporal
        insights.append("\n=== ANÁLISE TEMPORAL DE RECEBIMENTOS ===")
        
        # Por mês
        monthly_profits = profits.groupby(profits['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count', 'mean'],
            'date': 'nunique'
        }).round(2)
        
        insights.append("\nRecebimentos por mês:")
        for month, row in monthly_profits.iterrows():
            insights.append(f"  {month}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})")
        
        # Por dia da semana
        profits['day_of_week'] = profits['date'].dt.day_name()
        daily_analysis = profits.groupby('day_of_week').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        insights.append("\nRecebimentos por dia da semana:")
        for day, row in daily_analysis.iterrows():
            insights.append(f"  {day}: R$ {row[('amount', 'sum')]:,.2f} ({row[('amount', 'count')]:.0f} transações, média: R$ {row[('amount', 'mean')]:.2f})")
        
        
        # Análise de frequência
        insights.append("\n=== ANÁLISE DE FREQUÊNCIA ===")
        profit_frequency = profits.groupby(profits['date'].dt.date).size()
        insights.append(f"Dias com recebimentos: {len(profit_frequency)} de {(end_date - start_date).days + 1}")
        insights.append(f"Frequência média: {profit_frequency.mean():.1f} transações/dia")
        insights.append(f"Dia com mais recebimentos: {profit_frequency.max()} transações")
        insights.append(f"Dias sem recebimentos: {(end_date - start_date).days + 1 - len(profit_frequency)}")
        
        # Distribuição de valores
        insights.append("\n=== DISTRIBUIÇÃO DE VALORES ===")
        value_ranges = [
            (0, 100, "Até R$ 100"),
            (100, 500, "R$ 100-500"),
            (500, 1000, "R$ 500-1.000"),
            (1000, 2000, "R$ 1.000-2.000"),
            (2000, 5000, "R$ 2.000-5.000"),
            (5000, float('inf'), "Acima de R$ 5.000")
        ]
        
        for min_val, max_val, label in value_ranges:
            if max_val == float('inf'):
                count = len(profits[profits['amount'] >= min_val])
                total_range = profits[profits['amount'] >= min_val]['amount'].sum()
            else:
                count = len(profits[(profits['amount'] >= min_val) & (profits['amount'] < max_val)])
                total_range = profits[(profits['amount'] >= min_val) & (profits['amount'] < max_val)]['amount'].sum()
            
            percentage = (count / len(profits)) * 100
            value_percentage = (total_range / total_profits) * 100
            insights.append(f"  {label}: {count:.0f} transações ({percentage:.1f}%) - R$ {total_range:,.2f} ({value_percentage:.1f}%)")
        
        # Estatísticas avançadas
        insights.append("\n=== ESTATÍSTICAS AVANÇADAS ===")
        insights.append(f"Desvio padrão: R$ {profits['amount'].std():.2f}")
        insights.append(f"Variância: R$ {profits['amount'].var():.2f}")
        insights.append(f"Coeficiente de variação: {(profits['amount'].std() / profits['amount'].mean()) * 100:.1f}%")
        insights.append(f"Quartil 25%: R$ {profits['amount'].quantile(0.25):.2f}")
        insights.append(f"Quartil 75%: R$ {profits['amount'].quantile(0.75):.2f}")
        insights.append(f"Amplitude interquartil: R$ {profits['amount'].quantile(0.75) - profits['amount'].quantile(0.25):.2f}")
        
        return "\n".join(insights)
    
    def generate_complete_analysis(self) -> str:
        """Gera análise completa de todas as transações"""
        insights = []
        
        start_date = self.df['date'].min()
        end_date = self.df['date'].max()
        total_transactions = len(self.df)
        total_expenses = self.df_expenses['amount'].sum()
        total_profits = self.df_profits['amount'].sum()
        net_balance = total_profits - total_expenses
        
        insights.append("=== ANÁLISE COMPLETA DE TRANSAÇÕES ===\n")
        insights.append(f"Período: {start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
        insights.append(f"Total de dias: {(end_date - start_date).days + 1}")
        insights.append(f"Total de transações: {total_transactions:,}")
        insights.append(f"Gastos: {len(self.df_expenses):,} transações")
        insights.append(f"Recebimentos: {len(self.df_profits):,} transações")
        insights.append("")
        
        insights.append("=== RESUMO FINANCEIRO ===")
        insights.append(f"Total de gastos: R$ {total_expenses:,.2f}")
        insights.append(f"Total de recebimentos: R$ {total_profits:,.2f}")
        insights.append(f"Saldo líquido: R$ {net_balance:,.2f}")
        insights.append(f"Gasto médio diário: R$ {total_expenses / ((end_date - start_date).days + 1):.2f}")
        insights.append(f"Recebimento médio mensal: R$ {total_profits / 5:,.2f}")
        insights.append("")
        
        # Análise de fluxo de caixa
        insights.append("=== ANÁLISE DE FLUXO DE CAIXA ===")
        
        # Por mês
        monthly_expenses = self.df_expenses.groupby(self.df_expenses['date'].dt.to_period('M'))['amount'].sum()
        monthly_profits = self.df_profits.groupby(self.df_profits['date'].dt.to_period('M'))['amount'].sum()
        
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
        insights.append("\n=== ANÁLISE DE TENDÊNCIAS ===")
        
        if len(monthly_expenses) > 1:
            expense_trend = monthly_expenses.iloc[-1] - monthly_expenses.iloc[0]
            expense_trend_pct = (expense_trend / monthly_expenses.iloc[0]) * 100
            insights.append(f"Tendência de gastos: {'+' if expense_trend > 0 else ''}R$ {expense_trend:.2f} ({expense_trend_pct:+.1f}%)")
        
        if len(monthly_profits) > 1:
            profit_trend = monthly_profits.iloc[-1] - monthly_profits.iloc[0]
            profit_trend_pct = (profit_trend / monthly_profits.iloc[0]) * 100
            insights.append(f"Tendência de recebimentos: {'+' if profit_trend > 0 else ''}R$ {profit_trend:.2f} ({profit_trend_pct:+.1f}%)")
        
        # Análise de estabelecimentos mais frequentes
        insights.append("\n=== ESTABELECIMENTOS MAIS FREQUENTES ===")
        
        # Top 10 estabelecimentos por frequência
        most_frequent = self.df['title'].value_counts().head(10)
        insights.append("\nTop 10 por frequência de transações:")
        for i, (merchant, count) in enumerate(most_frequent.items(), 1):
            insights.append(f"{i:2d}. {merchant}: {count} transações")
        
        # Top 10 estabelecimentos por valor total
        top_by_value = self.df.groupby('title')['amount'].sum().abs().sort_values(ascending=False).head(10)
        insights.append("\nTop 10 por valor total (absoluto):")
        for i, (merchant, value) in enumerate(top_by_value.items(), 1):
            insights.append(f"{i:2d}. {merchant}: R$ {value:,.2f}")
        
        # Métricas de saúde financeira
        insights.append("\n=== MÉTRICAS DE SAÚDE FINANCEIRA ===")
        
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
            insights.append(f"Consistência de recebimentos: {profit_consistency:.1f}% (menor = mais consistente)")
        
        # Consistência de gastos
        if len(monthly_expenses) > 1:
            expense_consistency = (monthly_expenses.std() / monthly_expenses.mean()) * 100
            insights.append(f"Consistência de gastos: {expense_consistency:.1f}% (menor = mais consistente)")
        
        # Análise de sazonalidade
        insights.append("\n=== ANÁLISE DE SAZONALIDADE ===")
        
        # Por dia da semana
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        daily_analysis = self.df.groupby(['day_of_week', self.df['amount'] > 0]).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        insights.append("\nAtividade por dia da semana:")
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            try:
                expenses_day = daily_analysis.loc[(day, True), ('amount', 'sum')]
                profits_day = daily_analysis.loc[(day, False), ('amount', 'sum')]
                insights.append(f"  {day}: Gastos R$ {expenses_day:,.2f}, Recebimentos R$ {profits_day:,.2f}")
            except KeyError:
                continue
        
        return "\n".join(insights)

def main():
    """Função principal para executar a análise"""
    try:
        analyzer = NubankAnalyzer()
        
        # Gerar análises separadas
        print("Gerando análise de gastos...")
        expenses_insights = analyzer.generate_expenses_analysis()
        
        print("Gerando análise completa...")
        complete_insights = analyzer.generate_complete_analysis()
        
        # Salvar em arquivos separados
        with open("gastos_analysis.txt", "w", encoding="utf-8") as f:
            f.write(expenses_insights)
        
        with open("analise_completa.txt", "w", encoding="utf-8") as f:
            f.write(complete_insights)
        
        print("\n" + "="*60)
        print("Análises salvas em:")
        print("  - gastos_analysis.txt")
        print("  - analise_completa.txt")
        print("="*60)
        
        # Mostrar resumo
        print("\nRESUMO EXECUTIVO:")
        print(f"Total de gastos: R$ {analyzer.df_expenses['amount'].sum():,.2f}")
        print(f"Total de recebimentos: R$ {analyzer.df_profits['amount'].sum():,.2f}")
        print(f"Saldo líquido: R$ {analyzer.df_profits['amount'].sum() - analyzer.df_expenses['amount'].sum():,.2f}")
        
    except Exception as e:
        print(f"Erro na análise: {e}")

if __name__ == "__main__":
    main()
