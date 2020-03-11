import numpy as np
from mlxtend.evaluate import mcnemar

cc = 688
cw = 127
wc = 85
ww = 204

cont_matrix = np.array([[cc, cw],[wc, ww]])
chi2, p = mcnemar(ary=cont_matrix, corrected=False)
print('chi-squared:', chi2)
print('p-value:', p)