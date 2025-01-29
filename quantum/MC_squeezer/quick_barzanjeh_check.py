import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0,3,100)

G = 100

AN = 0.5*(1 - 1/G)

nbar = 5

SNR_PA = 2*np.sqrt(nbar)*np.abs(np.sin(theta))/np.sqrt(2*AN+1)


#%%

G1 = 2
G2 = 100

AN1 = 0.5*(1 - 1/G1)
AN2 = 0.5*(1 - 1/G2)

SNR_SU11PA = 2*np.sqrt(nbar)*np.abs(np.sin(theta))/np.sqrt( (2*AN1 + 1)*(2*AN2 + 1) - 8*np.cos(theta) * np.sqrt(AN1*AN2) )

plt.plot(theta, SNR_SU11PA/SNR_PA)


#%%

import numpy as np
import matplotlib.pyplot as plt


chikapparatio = np.linspace(0,5,100)

theta = 2*np.arctan(chikapparatio)

G = 100

AN = 0.5*(1 - 1/G)

nbar = 5

SNR_PA = 2*np.sqrt(nbar)*np.abs(np.sin(theta))/np.sqrt(2*AN+1)

#%%

G1 = 2
G2 = 100

AN1 = 0.5*(1 - 1/G1)
AN2 = 0.5*(1 - 1/G2)

SNR_SU11PA = 2*np.sqrt(nbar)*np.abs(np.sin(theta))/np.sqrt( (2*AN1 + 1)*(2*AN2 + 1) - 8*np.cos(theta) * np.sqrt(AN1*AN2) )

plt.plot(chikapparatio, SNR_SU11PA)
plt.plot(chikapparatio, SNR_PA)
plt.grid()