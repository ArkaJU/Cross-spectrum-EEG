import numpy as np
import pycwt as wavelet
from utils import get_sums, correntropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, skew, moment, trim_mean
from constants import DJ
'''
pycwt.xwt()
Parameters:	
y2 (y1,) – Input signal array to calculate cross wavelet transform.
dt (float) – Sample spacing.
dj (float, optional) – Spacing between discrete scales. Default value is 1/12. Smaller values will result in better scale resolution, but slower calculation and plot.
s0 (float, optional) – SmaAllest scale of the wavelet. Default value is 2*dt.
J (float, optional) – Number of scales less one. Scales range from s0 up to s0 * 2**(J * dj), which gives a total of (J + 1) scales. Default is J = (log2(N*dt/so))/dj.
wavelet (instance of a wavelet class, optional) – Mother wavelet class. Default is Morlet wavelet.
significance_level (float, optional) – Significance level to use. Default is 0.95.
normalize (bool, optional) – If set to true, normalizes CWT by the standard deviation of the signals.
Returns:	
xwt (array like) – Cross wavelet transform according to the selected mother wavelet.
x (array like) – Intersected independent variable.
coi (array like) – Cone of influence, which is a vector of N points containing the maximum Fourier period of useful information at that particular time. Periods greater than those are subject to edge effects.
freqs (array like) – Vector of Fourier equivalent frequencies (in 1 / time units) that correspond to the wavelet scales.
signif (array like) – Significance levels as a function of scale.
'''

def backward_difference_operator(s,k):
  if k==0:
    return s
  return backward_difference_operator(np.diff(s),k-1)


#@profile
def feature_gen(s1, s2):    

  #time domain features
  f_t1 = np.max(s1)
  f_t2 = np.min(s1)
  f_t3 = np.mean(s1)
  f_t4 = np.std(s1)
  f_t5 = f_t1-f_t2
  f_t6 = np.percentile(s1, 25)
  f_t7 = np.percentile(s1, 50)
  f_t8 = np.percentile(s1, 75)
  f_t9 = skew(s1)
  f_t10 = kurtosis(s1)

  f_t11 = np.mean(np.absolute(s1))
  f_t12 = np.where(np.diff(np.sign( [i for i in s1 if i] )))[0].shape[0] #zero crossings
  f_t13 = np.where(np.diff(np.sign( [i for i in np.gradient(s1) if i] )))[0].shape[0] #slope sign change
  
  f_t14 = np.sum([n**2 for n in s1])/len(s1)    #power
  f_t15 = np.sum(np.abs(s1[i+1]-s1[i])for i in range(len(s1)-1))/(len(s1)-1) #First difference
  f_t16 = f_t15/f_t4 # Normalized first difference
  f_t17 = np.sum(np.abs(s1[i+2]-s1[i])for i in range(len(s1)-2))/(len(s1)-2) #Second difference
  f_t18 = f_t17/f_t4  # Normalized second difference
  f_t19 = np.sum((s1[i]-f_t3)**2 for i in range(len(s1)))/(len(s1)) # Activity
  f_t20 = np.std(np.diff(s1))/f_t4  # Mobility
  f_t21 = (np.std(np.diff(np.diff(s1)))/np.std(np.diff(s1)))/f_t20 # Complexity
  bd = backward_difference_operator(s1,10) # Higher order crossings
  f_t22 = np.where(np.diff(np.sign( [i for i in bd if i] )))[0].shape[0]


  #frequency domain features
  s1 = np.array(s1)
  s2 = np.array(s2)

  dt = 1
  W_complex, _, _, _ = wavelet.xwt(s1, s2, dt, dj=1/DJ, normalize=True)                  
  W = np.abs(W_complex)   #row->scale, col->time
  

  total_scales = W.shape[0]
  total_time = W.shape[1]
  
  accum, accum_sq, accum_2, accum_sq_2 = get_sums(W, inv=False)                           
  W_sum = np.sum(W)

  #xwt matrix
  f1 = accum/W_sum
  f1_1 = accum_2/W_sum
  f2 = np.sqrt(accum_sq/W_sum)
  f2_1 = np.sqrt(accum_sq_2/W_sum)

  accum, accum_sq, accum_2, accum_sq_2 = get_sums(W, inv=True)  
  f1_inv = accum/W_sum
  f1_inv1 = accum_2/W_sum
  f2_inv = np.sqrt(accum_sq/W_sum)
  f2_inv1 = np.sqrt(accum_sq_2/W_sum)

  f3 = W_sum/np.max(W)

  x = total_scales*total_time

  f4 = W_sum/x              #adding small eps to avoid divide by zero error
  f5 = np.sqrt( (np.sum((np.square(f4 - W))) )/x)

  s_max, t_max = np.unravel_index(W.argmax(), W.shape)
  s_max += 1      #to take care of 0 indexing
  t_max += 1

  f6 = s_max     
  f7 = t_max     
  f8 = np.median(W)

  W_flat = W.flatten()
  f9 = skew(W_flat)
  f10 = 3*(np.mean(W_flat)-np.median(W_flat))/np.std(W_flat)

  #phi matrix
  phi = np.abs(np.angle(W_complex))                   
  assert(phi.shape == W_complex.shape)  

  accum_phase, accum_sq_phase, accum_2_phase, accum_sq_2_phase = get_sums(phi, inv=False)             
  phi_sum = np.sum(phi)

  f11 = accum_phase/phi_sum
  f11_1 = accum_2_phase/phi_sum
  f12 = np.sqrt(accum_sq_phase/phi_sum)
  f12_1 = np.sqrt(accum_sq_2_phase/phi_sum)

  accum_phase, accum_sq_phase, accum_2_phase, accum_sq_2_phase = get_sums(phi, inv=True) 
  
  f11_inv = accum_phase/phi_sum
  f11_inv1 = accum_2_phase/phi_sum
  f12_inv = np.sqrt(accum_sq_phase/phi_sum)
  f12_inv1 = np.sqrt(accum_sq_2_phase/phi_sum)

  f13 = phi_sum/np.max(phi)
  f14 = phi_sum/x
  f15 = np.sqrt((np.sum( np.square((f14-phi)) )) /x)
  f16 = np.median(phi)

  phi_flat = phi.flatten()
  f17 = skew(phi_flat)                                             #moment measure
  f18 = 3*(np.mean(phi_flat)-np.median(phi_flat))/np.std(phi_flat) #2nd moment measure




  #wct matrix
  # R, _, _, _, _ = wavelet.wct(s1, s2, dt, dj=1/DJ, normalize=True, sig=False)

  # assert(R.shape == W_complex.shape)  

  # accum, accum_sq, accum_2, accum_sq_2 = get_sums(R, inv=False)  
  # R_sum = np.sum(R)

  # f19 = accum/R_sum
  # f19_1 = accum_2/R_sum
  # f20 = np.sqrt(accum_sq/R_sum)
  # f20_1 = accum_sq_2/R_sum

  # accum, accum_sq, accum_2, accum_sq_2 = get_sums(R, inv=True)  
  # f19_inv = accum/R_sum
  # f19_inv1 = accum_2/R_sum
  # f20_inv = np.sqrt(accum_sq/R_sum)
  # f20_inv1 = accum_sq_2/R_sum

  # f21 = R_sum/np.max(R) 
  # f22 = R_sum/x
  # f23 = np.sqrt((np.sum( (np.square(f22 - R)) ))/x) 

  # s_max, t_max = np.unravel_index(R.argmax(), R.shape)
  # s_max += 1    #to take care of 0 indexing
  # t_max += 1
  # x = total_scales*total_time  

  # f24 = s_max    
  # f25 = t_max
  # f26 = np.median(R)

  # R_flat = R.flatten()
  # f27 = skew(R_flat)
  # f28 = 3*(np.mean(R_flat)-np.median(R_flat))/np.std(R_flat)

  f29 = np.sqrt(np.sum(np.square(s1-s2)))     #euclidean distance between s1 and mcv ?
  f30 = cosine_similarity(s1.reshape(1,-1), s2.reshape(1,-1), dense_output=True).reshape(1,1)[0][0]

  corr = correntropy(s1,s2)
  f31 = np.mean(corr)
  f32 = kurtosis(corr)
  f33 = skew(corr)
  f34 = moment(corr, moment=2)
  f35 = moment(corr, moment=3)
  f36 = trim_mean(corr, 0.1)

  #print(f_t22)
  f = [f_t1,f_t2,f_t3,f_t4,f_t5,f_t6,f_t7,f_t8,f_t9,f_t10,f_t11,f_t12,f_t13,f_t14,f_t15,f_t16,f_t17,f_t18,f_t19,f_t20,f_t21,f_t22,   #time
       f1, f1_1, f2, f2_1,  f1_inv, f1_inv1, f2_inv, f2_inv1,   f3, f4, f5, f6, f7, f8, f9, f10,                                      #W
       f11,f11_1,f12,f12_1, f11_inv,f11_inv1,f12_inv,f12_inv1, f13,f14,f15,f16,f17,f18,                                              #phi
       #f19,f19_1,f20,f20_1, f19_inv,f19_inv1,f20_inv,f20_inv1, f21,f22,f23,f24,f25,f26,f27,f28,                                       #wct
       f29,f30,f31,f32,f33,f34,f35,f36]                                                                                              #misc.

    
  return np.array(f)