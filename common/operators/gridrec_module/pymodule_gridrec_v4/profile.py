import numpy as np
import gridrec

sino = np.zeros((901,2016),dtype=np.float32)

reco = gridrec.backproj(sino,np.linspace(0,180,901).astype(np.float32),np.array([1008,1],dtype=np.float32),np.zeros(1,dtype=np.float32),"profile.wis")
