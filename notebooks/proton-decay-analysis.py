import matplotlib
matplotlib.use('Agg')
import pandas as p
df = p.read_csv("proton-decay-inference.csv")
query1 = df.query("label != prediction")
query2 = query1.query('kaon_ke != -1')
import matplotlib.pyplot as plt
query2.plot(x='kaon_ke', y='probability')
plt.show()