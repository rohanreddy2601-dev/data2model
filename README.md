# ApexPlanet Data Analytics Internship — Task 1

## 📌 Project Overview
This project is part of my **Data Analytics Internship at ApexPlanet Software Pvt. Ltd.**
Task 1 focuses on foundational setup, data cleaning, and exploratory data analysis (EDA) on a global e-commerce sales dataset.

## 🎯 Objective
Set up a proper analytics environment, clean a real-world messy dataset, and extract meaningful business insights through exploratory analysis.

## 🗂️ Dataset
- **Name:** E-commerce Sales Dataset
- **Size:** 51,290 orders × 24 columns
- **Coverage:** Global orders across multiple markets (US, APAC, EU, Africa, and more)
- **Key fields:** Order Date, Ship Date, Category, Sub-Category, Sales, Quantity, Discount, Profit, Shipping Cost, Market, Region

## 🛠️ Tools & Libraries
- Python 3.11
- Jupyter Notebook
- pandas, numpy
- matplotlib, seaborn
- sqlalchemy

## 📁 Project Structure
```
apexplanet-data-analytics/
├── data/               # Raw and cleaned datasets
├── notebooks/          # Jupyter notebooks with EDA
├── scripts/            # Reusable Python scripts
├── reports/            # Summary reports
├── dashboards/         # Visualization dashboards
└── README.md
```

## 🧹 Data Cleaning Steps
1. Removed 232 empty/junk columns introduced during Excel → CSV conversion
2. Checked and confirmed no duplicate records
3. Converted `Order Date` and `Ship Date` from text to proper datetime format
4. Converted categorical columns (Ship Mode, Segment, Market, Region, Category, Sub-Category, Order Priority) to category dtype for efficiency
5. Identified and handled outliers in `Sales` and `Profit` using the IQR method

## 📊 Exploratory Data Analysis
- Distribution analysis of Sales (histogram, boxplot)
- Total Sales by Category (bar chart)
- Monthly Sales trend over time (line chart)
- Correlation heatmap across Sales, Quantity, Discount, Profit, and Shipping Cost

## 💡 Key Insights
1. **Postal Code is missing in ~80% of orders** — not a data quality issue, but a reflection of the dataset's global scope, since postal codes were only reliably captured for US-based orders.
2. **Some orders are highly unprofitable**, with Profit ranging from -6,599 to +8,399 — heavy discounting and high shipping costs can turn otherwise strong sales into losses.
3. **Sales are right-skewed**, with most orders falling in the low-to-mid value range and a small number of large orders pulling the average up — typical of retail/e-commerce data.
4. Outlier removal on Sales and Profit reduced the dataset by ~26.5%, showing a meaningful share of orders sit outside "typical" ranges — worth analyzing separately rather than discarding entirely.
5. Certain product categories consistently outperform others in total sales, highlighting where the business generates the most revenue.

## 🚀 How to Run
1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn plotly sqlalchemy openpyxl
   ```
3. Open `notebooks/01_eda.ipynb` in Jupyter Notebook
4. Run all cells

## 👤 Author
**Yelletiwar Rohan Reddy**
Data Analytics Intern @ ApexPlanet Software Pvt. Ltd.
