import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import seaborn as sns

sns.set('talk', 'darkgrid', 'dark', font_scale=1.5, font='Ricty',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})

plt.style.use('seaborn-white')
jj = np.loadtxt("roc_values_instance")
fpr, tpr, _ = roc_curve(jj[:,0], jj[:,1])
roc_auc = auc(fpr, tpr)

lw = 2
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC (Receiver Operating Characteristic curve)')
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_auc.png')
plt.close()
