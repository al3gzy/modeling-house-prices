# House Price Modeling  

---

## Key Findings:

- **Simple Linear Regression** between house price (SalePrice) and above-ground living area (GrLivArea):
  - Model:  
    \[
    \hat{Y} = 14,379.96 + 110.4182 \times X
    \]
  - Coefficient \( \beta \) statistically significant (p < 0.05), with \( R^2 = 0.5125 \), explaining ~51.25% of price variance.
  - Test set \( R^2 = 0.4437 \) indicates moderate generalization.

- **Multiple Linear Regression** with predictors (GrLivArea, OverallQual, GarageArea, YearBuilt, TotRmsAbvGrd, FullBath, LotArea, X1stFlrSF):
  - Full model statistically significant (F = 595.9, p < 0.05).
  - Adjusted \( R^2 = 0.7667 \), explaining ~76.7% of price variance.
  - Reduced model excluding insignificant predictors shows negligible difference in \( R^2 \).

- **Nonlinear Models:**
  - Parabolic model achieves \( R^2 = 0.5085 \).
  - Parabola with sinusoidal oscillations marginally improves fit \( R^2 = 0.5103 \).
  - Cubic spline model less effective, \( R^2 = 0.3315 \).
  - Decision tree model performs best with \( R^2 = 0.8134 \), indicating high explanatory power.

- **ANOVA Analysis:**
  - One-way ANOVA confirms significant effect of house quality (OverallQual) on price (F = 349.03, p < 0.0001).
  - Two-way ANOVA shows significant main effects of OverallQual and YearBuilt, but no significant interaction effect between them.

---

## Summary:

The project successfully models real estate prices using various regression techniques and data analysis methods. Simple linear regression demonstrates a strong linear relationship between living area and price, while multiple regression incorporating additional features substantially improves explanatory power. Among nonlinear methods, decision trees provide the best fit. ANOVA results highlight the critical impact of house quality and year built on pricing. These insights can inform pricing strategies and predictive modeling in real estate markets.
