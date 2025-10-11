from modified_dietz import ModifiedDietzCalculator

def main():
    # Create calculator instance
    calc = ModifiedDietzCalculator()
    
    # Add 2021 data
    calc.add_period(
        year=2021,
        start=3667991,
        end=10810823,
        movements=[
            {'date': '2021-01-13', 'amount': 500000},
            {'date': '2021-01-14', 'amount': 4463126},
            {'date': '2021-02-07', 'amount': 1600000}
        ]
    )
    
    # Add 2022 data
    calc.add_period(
        year=2022,
        start=10810823,
        end=9946421,
        movements=[]
    )
    
    # Add 2023 data
    calc.add_period(
        year=2023,
        start=9946421,
        end=12032083,
        movements=[
            {'date': '2023-04-14', 'amount': 901810},
            {'date': '2023-04-17', 'amount': -500000}
        ]
    )
    
    # Add 2024 data
    calc.add_period(
        year=2024,
        start=12032083,
        end=26706398,
        movements=[
            {'date': '2024-01-30', 'amount': 798773},
            {'date': '2024-09-09', 'amount': -200000},
            {'date': '2024-10-21', 'amount': 12467465}
        ]
    )
    

    # Add 2025 data
    calc.add_period(
        year=2025,
        start=26706398.00,
        end=35001066.00,
        movements=[
            {'date': '2025-01-02', 'amount': 5091500.00},
            {'date': '2025-05-15', 'amount': -200000.00},
            {'date': '2025-05-25', 'amount': 1589454.00}
        ]
    )

    # Print results to console
    print("\nModified Dietz Returns:")
    print("-" * 20)
    for period in sorted(calc.periods, key=lambda x: x.year):
        print(f"Year {period.year}: {period.return_rate:.2%}")
    print(f"\nCumulative Return: {calc.calculate_cumulative():.2%}")

    # Export to PDF
    try:
        calc.export_pdf('JB Mod Dietz Calc.pdf')
        print("\nDetailed results exported to JB Mod Dietz Calc.pdf")
    except Exception as e:
        print(f"\nError exporting PDF: {str(e)}")

if __name__ == "__main__":
    main()