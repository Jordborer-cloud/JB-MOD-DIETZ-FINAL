from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Add seaborn import
import yfinance as yf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
try:
    from fredapi import Fred
except Exception:
    Fred = None
import numpy as np

@dataclass
class Period:
    year: int
    start_balance: float
    end_balance: float
    movements: List[Dict[str, Any]]
    return_rate: float = 0.0

class CPIDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        # Initialize FRED client only if fredapi is available
        self.fred = Fred(api_key) if (api_key and Fred is not None) else None
        # Fallback CPI data if API is not available
        self.fallback_cpi = {
            2020: 0.012,  # 1.2%
            2021: 0.070,  # 7.0%
            2022: 0.065,  # 6.5%
            2023: 0.041,  # 4.1%
            2024: 0.032,  # 3.2% (projected)
            2025: 0.025   # 2.5% (projected)
        }

    def get_cpi_returns(self, start_year: int, end_year: int) -> Dict[int, float]:
        """Get annual CPI returns for given year range"""
        if self.fred:
            try:
                # Fetch CPI data from FRED
                cpi = self.fred.get_series('CPIAUCSL')  # Consumer Price Index for All Urban Consumers
                # Calculate annual returns
                annual_cpi = cpi.resample('Y').last()
                cpi_returns = annual_cpi.pct_change()
                # Convert to dictionary
                return {year: rate for year, rate in 
                       cpi_returns[str(start_year):str(end_year)].items()}
            except Exception as e:
                print(f"Warning: Failed to fetch CPI data from FRED: {e}")
                
        # Use fallback data if API fails or is not available
        return {year: self.fallback_cpi.get(year, 0.025) 
                for year in range(start_year, end_year + 1)}

class ModifiedDietzCalculator:
    def __init__(self, fred_api_key: Optional[str] = None):
        self.periods = []
        self.cpi_fetcher = CPIDataFetcher(fred_api_key)

    def add_period(self, year: int, start: float, end: float, movements: List[Dict[str, Any]]) -> None:
        norm_movements = []
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        for m in movements:
            move_date = datetime.strptime(m['date'], '%Y-%m-%d').date()
            if move_date < year_start or move_date > year_end:
                raise ValueError(f"Movement date {m['date']} not in year {year}")
            norm_movements.append({'date': m['date'], 'amount': float(m['amount'])})
        period = Period(
            year=year,
            start_balance=float(start),
            end_balance=float(end),
            movements=norm_movements
        )
        period.return_rate = self._calculate_return(period)
        self.periods.append(period)

    def _calculate_return(self, period: Period) -> float:
        year_start = date(period.year, 1, 1)
        year_end = date(period.year, 12, 31)
        days_in_year = (year_end - year_start).days + 1
        weighted_flows = 0.0
        total_flows = 0.0
        for m in period.movements:
            move_date = datetime.strptime(m['date'], '%Y-%m-%d').date()
            days_weight = (year_end - move_date).days / days_in_year
            weighted_flows += float(m['amount']) * days_weight
            total_flows += float(m['amount'])
        numerator = period.end_balance - period.start_balance - total_flows
        denominator = period.start_balance + weighted_flows
        if abs(denominator) < 1e-10:
            return 0.0
        return numerator / denominator

    def calculate_cumulative(self) -> float:
        cumulative = 1.0
        for period in sorted(self.periods, key=lambda x: x.year):
            cumulative *= (1.0 + period.return_rate)
        return cumulative - 1.0

    def _get_benchmarks(self) -> Dict[str, List[float]]:
        years = sorted(p.year for p in self.periods)
        start_year, end_year = min(years), max(years)
        cpi_data = self.cpi_fetcher.get_cpi_returns(start_year, end_year)
        # Only CPI benchmark is returned. AOR/ETF comparisons removed.
        benchmarks = {
            'CPI': [cpi_data.get(year, 0.025) for year in years]
        }
        return benchmarks
        # Download USD/ZAR rates for each year-end
        try:
            fx_data = yf.download('ZAR=X', start=f"{years[0]}-01-01", end=f"{years[-1]+1}-01-10", interval='1d', progress=False)
            fx_rates = []
            for d in year_end_dates:
                # If market closed on 31 Dec, get last available price before year-end
                date_obj = pd.to_datetime(d)
                if d in fx_data.index:
                    fx_rates.append(float(fx_data.loc[d]['Close']))
                else:
                    # Find last available date before year-end
                    prev_dates = fx_data.loc[fx_data.index <= date_obj]
                    if not prev_dates.empty:
                        fx_rates.append(float(prev_dates.iloc[-1]['Close']))
                    else:
                        fx_rates.append(np.nan)
        except Exception as e:
            print(f"Warning: Could not fetch USD/ZAR rates: {e}")
            fx_rates = [np.nan] * len(years)

        end_balances = [p.end_balance for p in sorted(self.periods, key=lambda x: x.year)]
        zar_balances = [bal * rate if rate == rate else np.nan for bal, rate in zip(end_balances, fx_rates)]

        fig, ax = plt.subplots()
        ax.plot(years, zar_balances, marker='o', color='darkgreen', linewidth=2, label='Portfolio Value (ZAR)')
        for i, val in enumerate(zar_balances):
            if val == val:  # not nan
                ax.annotate(f'R{val:,.0f}', xy=(years[i], val), xytext=(0, 8), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='darkgreen')
        ax.set_title('Portfolio Value in ZAR (Year-End)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Portfolio Value (ZAR)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))

    def __init__(self, fred_api_key: Optional[str] = None):
        self.periods = []
        self.cpi_fetcher = CPIDataFetcher(fred_api_key)

    def add_period(self, year: int, start: float, end: float, movements: List[Dict[str, Any]]) -> None:
        norm_movements = []
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        for m in movements:
            move_date = datetime.strptime(m['date'], '%Y-%m-%d').date()
            if move_date < year_start or move_date > year_end:
                raise ValueError(f"Movement date {m['date']} not in year {year}")
            norm_movements.append({'date': m['date'], 'amount': float(m['amount'])})
        period = Period(
            year=year,
            start_balance=float(start),
            end_balance=float(end),
            movements=norm_movements
        )
        period.return_rate = self._calculate_return(period)
        self.periods.append(period)

    def _calculate_return(self, period: Period) -> float:
        year_start = date(period.year, 1, 1)
        year_end = date(period.year, 12, 31)
        days_in_year = (year_end - year_start).days + 1
        weighted_flows = 0.0
        total_flows = 0.0
        for m in period.movements:
            move_date = datetime.strptime(m['date'], '%Y-%m-%d').date()
            days_weight = (year_end - move_date).days / days_in_year
            weighted_flows += float(m['amount']) * days_weight
            total_flows += float(m['amount'])
        numerator = period.end_balance - period.start_balance - total_flows
        denominator = period.start_balance + weighted_flows
        if abs(denominator) < 1e-10:
            return 0.0
        return numerator / denominator

    def calculate_cumulative(self) -> float:
        cumulative = 1.0
        for period in sorted(self.periods, key=lambda x: x.year):
            cumulative *= (1.0 + period.return_rate)
        return cumulative - 1.0

    def _get_benchmarks(self) -> Dict[str, List[float]]:
        years = sorted(p.year for p in self.periods)
        start_year, end_year = min(years), max(years)
        cpi_data = self.cpi_fetcher.get_cpi_returns(start_year, end_year)
        # Only CPI benchmark is returned.
        benchmarks = {
            'CPI': [cpi_data.get(year, 0.025) for year in years]
        }
        return benchmarks

    def export_pdf(self, filename: str) -> None:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.formatter.use_locale'] = True

        # Title
        elements.append(Paragraph("Portfolio Results", styles['Heading1']))
        elements.append(Spacer(1, 12))

        # Get benchmark data
        benchmarks = self._get_benchmarks()
        years = [p.year for p in sorted(self.periods, key=lambda x: x.year)]
        returns = [p.return_rate for p in sorted(self.periods, key=lambda x: x.year)]

        # Calculate performance metrics
        cum_return = self.calculate_cumulative()
        cum_cpi = (1 + pd.Series(benchmarks['CPI'])).prod() - 1

        # Calculate annual compound return
        n_years = len(years)
        annual_compound = (1 + cum_return) ** (1/n_years) - 1

        # Calculate outperformance vs CPI
        outperf_cpi = cum_return - cum_cpi

        # Figure 1: Annual Returns Comparison (Portfolio vs CPI)
        fig, ax = plt.subplots()
        x = np.arange(len(years))
        width = 0.35

        bars_portfolio = ax.bar(x - width/2, returns, width, label='Portfolio', color='royalblue')
        bars_cpi = ax.bar(x + width/2, benchmarks['CPI'], width, label='CPI', color='lightgreen')

        # Add data labels
        for bars in [bars_portfolio, bars_cpi]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        ax.set_title('Annual Returns Comparison')
        ax.set_xlabel('Year')
        ax.set_ylabel('Return (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
        ax.legend()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))


        # Figure 2: Yearly Money Made/Lost (Nominal) with Cumulative Line, actual values
        total_made_lost = []
        cumulative_nominal = []
        cum_sum = 0.0
        for period in sorted(self.periods, key=lambda x: x.year):
            net_flows = sum(float(m['amount']) for m in period.movements)
            made_lost = period.end_balance - period.start_balance - net_flows
            total_made_lost.append(made_lost)
            cum_sum += made_lost
            cumulative_nominal.append(cum_sum)

        fig, ax = plt.subplots()
        x = np.arange(len(years))
        bar1 = ax.bar(x, total_made_lost, width=0.5, label='Yearly Gain/Loss (Nominal)', color='mediumseagreen')
        line1 = ax.plot(x, cumulative_nominal, color='navy', marker='o', label='Cumulative (Nominal)')

        # Data labels for bars
        for bar, val in zip(bar1, total_made_lost):
            height = bar.get_height()
            ax.annotate(f'${height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

        # Data labels for cumulative line
        for i, val in enumerate(cumulative_nominal):
            ax.annotate(f'${val:,.0f}',
                         xy=(x[i], val),
                         xytext=(0, 8),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, color='navy')

        ax.set_title('Yearly Money Made/Lost (Nominal)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount ($)')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))



        # Create summary table (Portfolio vs CPI)
        summary_data = [
            ['Metric', 'Portfolio', 'CPI'],
            ['Cumulative Return', f"{cum_return:.2%}", f"{cum_cpi:.2%}"],
            ['Annual Compound Return', f"{annual_compound:.2%}", f"{((1 + cum_cpi) ** (1/n_years) - 1):.2%}"],
            ['Outperformance vs CPI', f"{outperf_cpi:+.2%}", '']
        ]

        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # Create detailed annual returns table (Year, Start, End, Return, vs CPI)
        data = [['Year', 'Start Balance', 'End Balance', 'Return', 'vs. CPI']]
        
        for i, period in enumerate(sorted(self.periods, key=lambda x: x.year)):
            vs_cpi = period.return_rate - benchmarks['CPI'][i]
            data.append([
                str(period.year),
                f"${period.start_balance:,.2f}",
                f"${period.end_balance:,.2f}",
                f"{period.return_rate:.2%}",
                f"{vs_cpi:+.2%}"
            ])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 24))  # Add extra space before cash movements
        
        # Create cash movements table
        elements.append(Paragraph("Cash Movements Detail", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        movements_data = [['Date', 'Amount', 'Year', 'Type']]
        
        for period in sorted(self.periods, key=lambda x: x.year):
            for movement in sorted(period.movements, key=lambda x: x['date']):
                amount = float(movement['amount'])
                movements_data.append([
                    movement['date'],
                    f"${amount:,.2f}",
                    str(period.year),
                    'Inflow' if amount > 0 else 'Outflow'
                ])
        
        if len(movements_data) > 1:  # If we have any movements
            movements_table = Table(movements_data)
            movements_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                # Color code inflows and outflows
                ('TEXTCOLOR', (-1, 1), (-1, -1), colors.black),
                # Conditional formatting for amount column
                ('TEXTCOLOR', (1, 1), (1, -1), colors.green),  # Default color for amounts
            ]))
            
            # Add color coding for positive/negative amounts
            for i in range(1, len(movements_data)):
                amount = float(movements_data[i][1].replace('$', '').replace(',', ''))
                if amount < 0:
                    movements_table.setStyle(TableStyle([
                        ('TEXTCOLOR', (1, i), (1, i), colors.red)
                    ]))
            
            elements.append(movements_table)
            
            # Add summary of cash flows
            total_inflows = sum(float(m['amount']) for p in self.periods for m in p.movements if float(m['amount']) > 0)
            total_outflows = sum(float(m['amount']) for p in self.periods for m in p.movements if float(m['amount']) < 0)
            net_flows = total_inflows + total_outflows
            
            elements.append(Spacer(1, 12))
            summary_flow_data = [
                ['Total Inflows', 'Total Outflows', 'Net Flow'],
                [f"${total_inflows:,.2f}", f"${total_outflows:,.2f}", f"${net_flows:,.2f}"]
            ]
            
            flow_summary = Table(summary_flow_data)
            flow_summary.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('TEXTCOLOR', (0, 1), (0, 1), colors.green),  # Inflows in green
                ('TEXTCOLOR', (1, 1), (1, 1), colors.red),    # Outflows in red
            ]))
            
            elements.append(flow_summary)
        else:
            elements.append(Paragraph("No cash movements recorded", styles['Normal']))

        # Figure: Portfolio Value in ZAR (using USD/ZAR year-end rates)
        # Figure: Portfolio Value in ZAR (using USD/ZAR year-end rates)
        year_end_dates = [pd.to_datetime(f"{y}-12-31") for y in years]
        fx_rates = []
        if 'yf' in globals() and yf is not None:
            try:
                # Download daily FX data covering the range and pick last available price on or before each year-end
                fx_data = yf.download('ZAR=X', start=f"{years[0]}-01-01", end=f"{years[-1]+1}-01-10", interval='1d', progress=False, auto_adjust=False)
                if fx_data is None or fx_data.empty:
                    raise Exception('empty fx data')
                fx_data.index = pd.to_datetime(fx_data.index)
                for date_obj in year_end_dates:
                    prev = fx_data.loc[fx_data.index <= date_obj]
                    if not prev.empty:
                        fx_rates.append(float(prev['Close'].iloc[-1]))
                    else:
                        fx_rates.append(np.nan)
            except Exception as e:
                print(f"Warning: Could not fetch USD/ZAR rates from yfinance: {e}")
                fx_rates = [np.nan] * len(years)
        else:
            print("Warning: yfinance not available; using fallback FX rates")
            fx_rates = [np.nan] * len(years)

        end_balances = [p.end_balance for p in sorted(self.periods, key=lambda x: x.year)]
        zar_balances = [bal * rate if rate == rate else np.nan for bal, rate in zip(end_balances, fx_rates)]

        fig, ax = plt.subplots()
        ax.plot(years, [v/1e6 for v in zar_balances], marker='o', color='darkgreen', linewidth=2, label='Portfolio Value (ZAR)')
        for i, val in enumerate(zar_balances):
            if val == val:  # not nan
                ax.annotate(f'R{val/1e6:,.2f}M', xy=(years[i], val/1e6), xytext=(0, 8), textcoords="offset points", ha='center', va='bottom', fontsize=8, color='darkgreen')
        ax.set_title('Portfolio Value in ZAR (Year-End)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Portfolio Value (ZAR millions)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))

        # Add exchange rate table below the ZAR chart, including USD and ZAR values
        end_balances = [p.end_balance for p in sorted(self.periods, key=lambda x: x.year)]
        zar_balances = [bal * rate if rate == rate else None for bal, rate in zip(end_balances, fx_rates)]
        fx_table_data = [['Year', 'USD/ZAR Rate', 'USD Portfolio Value', 'ZAR Portfolio Value']]
        for y, rate, usd, zar in zip(years, fx_rates, end_balances, zar_balances):
            fx_table_data.append([
                str(y),
                f'{rate:.4f}' if rate == rate else 'N/A',
                f'${usd:,.0f}',
                f'R{zar:,.0f}' if zar is not None else 'N/A'
            ])
        fx_table = Table(fx_table_data)
        fx_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(fx_table)
        elements.append(Spacer(1, 12))

        doc.build(elements)