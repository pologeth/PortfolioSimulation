# Portfolio Simulation: Monte Carlo Analysis

## **Overview**
This project provides a Monte Carlo simulation to analyze a diversified portfolio's potential outcomes over one year of trading days (252 days). The simulation calculates key portfolio statistics such as average portfolio value, standard deviation, and Value at Risk (VaR), and provides a sensitivity analysis of assets. The results are visualized using Python libraries like **Bokeh** and **Seaborn**.

## **Features**

### **Monte Carlo Simulation**:
- Models 1,000 portfolio trajectories using randomly generated daily returns.
- Incorporates each asset's mean returns, volatility, and portfolio weight.

### **Statistical Analysis**:
- Computes average portfolio value, standard deviation, and VaR (Value at Risk).

### **Interactive and Static Visualizations**:
- Interactive visualizations with **Bokeh**.
- Static plots using **Matplotlib** and **Seaborn**.

### **Sensitivity Analysis**:
- Identifies the assets with the greatest impact on portfolio performance.

## **Portfolio Configuration**
| Asset | Portfolio Weight (%) | Mean Return | Volatility |
|-------|----------------------|-------------|------------|
| QQQ   | 20                   | 8.0%        | 15.0%      |
| SP600 (IJR) | 15              | 7.0%        | 14.0%      |
| XLP   | 10                   | 5.0%        | 10.0%      |
| VTI   | 10                   | 6.0%        | 12.0%      |
| XLI   | 5                    | 6.0%        | 11.0%      |
| XLRE  | 5                    | 5.0%        | 10.0%      |
| MSFT  | 10                   | 12.0%       | 20.0%      |
| NVDA  | 8                    | 15.0%       | 30.0%      |
| AAPL  | 8                    | 10.0%       | 22.0%      |
| TACK ETF | 6                 | 8.0%        | 18.0%      |
| AMZN  | 5                    | 9.0%        | 25.0%      |
| TSLA  | 3                    | 15.0%       | 35.0%      |

## **Results**

### **Key Portfolio Statistics**:
- **Mean Final Portfolio Value**: $116,211.26
- **Standard Deviation**: $6,435.24
- **VaR (5%)**: $106,184.11
- **95th Percentile**: $127,455.75

### **Sensitivity Analysis**:
The graph below highlights the assets most sensitive to changes in their returns.

## **Visualizations**

### **Portfolio Distribution**
Interactive and static visualizations show the final distribution of simulated portfolio values.

**Histogram of Final Portfolio Values**:

### **Sensitivity Analysis**
**Sensitivity of Portfolio to Changes in Asset Returns**:

## **Installation**

### **Clone the Repository**:
```bash
git clone https://github.com/yourusername/PortfolioSimulation.git
cd PortfolioSimulation

