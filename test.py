from Technical_Analysis import TechnicalAnalysis
import pandas as pd

ta_4h = TechnicalAnalysis("BTC-USD", "4h" ,'technical_analysis_4h.csv') 
ta_daily = TechnicalAnalysis("BTC-USD", "1d" ,'technical_analysis_daily.csv') 
data_4h = ta_4h.get_technical_data()
data = ta_daily.get_technical_data()
print(data_4h)

print(data_4h['nearby_priceaction'])