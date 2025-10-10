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
from fredapi import Fred
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
        self.fred = Fred(api_key) if api_key else None
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
        self.periods: List[Period] = []
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
        """Get benchmark returns for comparison"""
        years = sorted(p.year for p in self.periods)
        start_year, end_year = min(years), max(years)
        
        # Get CPI data
        cpi_data = self.cpi_fetcher.get_cpi_returns(start_year, end_year)
        benchmarks = {
            'CPI': [cpi_data.get(year, 0.025) for year in years]
        }
        
        # Get AOR data
        try:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            aor = yf.download('AOR', start=start_date, end=end_date, interval='1y', progress=False)
            if not aor.empty:
                aor_returns = aor['Adj Close'].pct_change().fillna(0)
                benchmarks['AOR'] = aor_returns.values.tolist()
            else:
                # Fallback data if AOR fetch fails
                aor_rates = {2021: 0.102, 2022: -0.154, 2023: 0.142, 2024: 0.065}
                benchmarks['AOR'] = [aor_rates.get(year, 0.07) for year in years]
        except Exception as e:
            print(f"Warning: Using fallback data for AOR benchmark: {e}")
            aor_rates = {2021: 0.102, 2022: -0.154, 2023: 0.142, 2024: 0.065}
            benchmarks['AOR'] = [aor_rates.get(year, 0.07) for year in years]
        
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
        cum_aor = (1 + pd.Series(benchmarks['AOR'])).prod() - 1
        cum_cpi = (1 + pd.Series(benchmarks['CPI'])).prod() - 1
        
        # Calculate annual compound return
        n_years = len(years)
        annual_compound = (1 + cum_return) ** (1/n_years) - 1
        
        # Calculate outperformance vs benchmarks
        outperf_aor = cum_return - cum_aor
        outperf_cpi = cum_return - cum_cpi

        # Figure 1: Annual Returns Comparison
        fig, ax = plt.subplots()
        x = np.arange(len(years))
        width = 0.25
        
        bars_portfolio = ax.bar(x - width, returns, width, label='Portfolio', color='royalblue')
        bars_aor = ax.bar(x, benchmarks['AOR'], width, label='AOR ETF', color='lightcoral')
        bars_cpi = ax.bar(x + width, benchmarks['CPI'], width, label='CPI', color='lightgreen')
        
        ax.set_title('Annual Returns Comparison')
        ax.set_xlabel('Year')
        ax.set_ylabel('Return (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add % labels to bars
        def add_percent_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{value:.1%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        add_percent_labels(bars_portfolio, returns)
        add_percent_labels(bars_aor, benchmarks['AOR'])
        add_percent_labels(bars_cpi, benchmarks['CPI'])

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))

        # Figure 2: Cumulative Growth
        fig, ax = plt.subplots()
        cum_portfolio = (1 + pd.Series(returns)).cumprod()
        cum_aor_series = (1 + pd.Series(benchmarks['AOR'])).cumprod()
        cum_cpi_series = (1 + pd.Series(benchmarks['CPI'])).cumprod()
        
        ax.plot(years, cum_portfolio, marker='o', label='Portfolio', linewidth=2)
        ax.plot(years, cum_aor_series, marker='s', label='AOR ETF', linewidth=2)
        ax.plot(years, cum_cpi_series, marker='^', label='CPI', linewidth=2)
        
        ax.set_title('Cumulative Growth of $1 Investment')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value ($)')
        ax.grid(True)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '${:.2f}'.format(y)))
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            elements.append(Image(tmp.name, width=6*inch, height=4*inch))
            elements.append(Spacer(1, 12))

        # Create summary table
        summary_data = [
            ['Metric', 'Portfolio', 'AOR ETF', 'CPI'],
            ['Cumulative Return', f"{cum_return:.2%}", f"{cum_aor:.2%}", f"{cum_cpi:.2%}"],
            ['Annual Compound Return', f"{annual_compound:.2%}", 
             f"{((1 + cum_aor) ** (1/n_years) - 1):.2%}", 
             f"{((1 + cum_cpi) ** (1/n_years) - 1):.2%}"],
            ['Outperformance', 'N/A', f"{outperf_aor:+.2%}", f"{outperf_cpi:+.2%}"]
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

        # Create detailed annual returns table
        data = [['Year', 'Start Balance', 'End Balance', 'Return', 'vs. AOR', 'vs. CPI']]
        
        for i, period in enumerate(sorted(self.periods, key=lambda x: x.year)):
            vs_aor = period.return_rate - benchmarks['AOR'][i]
            vs_cpi = period.return_rate - benchmarks['CPI'][i]
            data.append([
                str(period.year),
                f"${period.start_balance:,.2f}",
                f"${period.end_balance:,.2f}",
                f"{period.return_rate:.2%}",
                f"{vs_aor:+.2%}",
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
        
        doc.build(elements)