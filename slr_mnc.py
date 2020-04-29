# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:03:54 2020

@author: ajayy
"""

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import statsmodels.formula.api as smf

mnc = pd.read_excel("E:/All Projects 2.0/simple Linear Regression/MncFund_Nifty_Sensex/Dataset/MNC_Fund_Data.xlsx")

mnc_data = mnc[['NAV','Nifty50']].copy()

#mnc_data = mnc_data.apply(np.log)

mnc_data.columns
mnc_data.describe()

#plt.hist(mnc_data["NAV"])
#plt.boxplot(mnc_data["NAV"])
#
#plt.hist(mnc_data["Nifty50"])
#plt.boxplot(mnc_data["Nifty50"])
#
#plt.scatter(x = mnc_data["Nifty50"], y = mnc_data["NAV"], color = "blue")


np.corrcoef(mnc_data["Nifty50"], mnc_data["NAV"] )


# Model without  applying transformation both x and y

model1 = smf.ols('NAV ~ Nifty50', data = mnc_data).fit()

model1.params

model1.summary()

pred1 = model1.predict(mnc_data)

#plt.scatter(x = mnc_data["Nifty50"], y = mnc_data["NAV"], color = "red")
#plt.plot(mnc_data["Nifty50"], pred1, color="black")
pred1.corr(mnc_data["NAV"])     # Rsquared = 0.597 & Accuracy = 0.772; low R^2 and accuracy



# Model applying square root for X


model2 = smf.ols('NAV ~ np.sqrt(Nifty50)', data = mnc_data).fit()

model2.params

model2.summary()

pred2 = model2.predict(mnc_data)

#plt.scatter(x = mnc_data["Nifty50"], y = mnc_data["NAV"], color = "red")
#plt.plot(mnc_data["Nifty50"], pred2, color="black")
pred2.corr(mnc_data["NAV"])     # Rsquared = 0.596 & Accuracy = 0.772 ; no improvements in R^2 & accuracy


# Model applying quadratic transformation for X

mnc_data["Nifty50_sq"] = mnc_data["Nifty50"] * mnc_data["Nifty50"]

model3 = smf.ols('NAV ~ Nifty50 + Nifty50_sq', data = mnc_data).fit()

model3.params

model3.summary()

pred3 = model3.predict(mnc_data)

#plt.scatter(x = mnc_data["Nifty50"], y = mnc_data["NAV"], color = "red")
#plt.plot(mnc_data["Nifty50"], pred3, color="black")
pred3.corr(mnc_data["NAV"])     # Rsquared = 0.645 & Accuracy = 0.803 ; good improvements in R^2 & in accuracy


# Model applying cubic transformation for X

mnc_data["Nifty50_cb"] = mnc_data["Nifty50"] * mnc_data["Nifty50"] * mnc_data["Nifty50"]

model4 = smf.ols('NAV ~ Nifty50 + Nifty50_sq + Nifty50_cb', data = mnc_data).fit()

model4.params

model4.summary()

#infl = model4.get_influence()

#type(infl)
pred4 = model4.predict(mnc_data)
pred4.corr(mnc_data["NAV"])  # Rsquared = 0.645 & Accuracy = 0.803 ; good improvements in R^2 & in accuracy

#plt.scatter(x = mnc_data["Nifty50"], y = mnc_data["NAV"], color = "red")
#plt.plot(mnc_data["Nifty50"], pred4, color="black")
pred4.corr(mnc_data["NAV"])     # Rsquared = 0.645 & Accuracy = 0.803 ; good improvements in R^2 & none in accuracy


# since there is no difference of result in squares & cubic transformation. we consider squares which is model3


#predF = np.exp(pred3)
mnc["pred"] = pred3

x = model3.predict(pd.DataFrame([[6001.7,36020402.89]], columns=["Nifty50", "Nifty50_sq"]))
print(float(round(x,2)))

model3.save("slr_mnc.pkl")









