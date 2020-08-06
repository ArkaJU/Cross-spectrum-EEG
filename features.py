import numpy as np
import pycwt as wavelet
from utils import get_sums, correntropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kurtosis, skew, moment, trim_mean

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

#@profile
#def feature_gen(s1, s2, Mean, Std, Max, encoder):   
def feature_gen(s1, s2, Mean, Std):   

  # x_test=s1/Max
  # x_test = np.reshape(x_test, (len(x_test), -1))
  # encoded_segs = encoder.predict(x_test.T)

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

  s1 = (s1-Mean)/Std  #pre-processing now
  dt = 1
  W_complex, _, _, _ = wavelet.xwt(s1, s2, dt, dj=1/12)                  
  W = np.abs(W_complex)   #row->scale, col->time
  phi = np.abs(np.angle(W_complex))                   

  assert(phi.shape == W_complex.shape)  

  total_scales = W.shape[0]
  total_time = W.shape[1]
  
  accum, accum_sq = get_sums(W)                           
  accum_phase, accum_sq_phase = get_sums(phi)             

  phi_sum = np.sum(phi)
  W_sum = np.sum(W)

  #frequency domain features
  f1 = accum/W_sum
  f2 = np.sqrt(accum_sq/W_sum)
  f3 = W_sum/np.max(W)

  s_min, t_min = np.unravel_index(W.argmin(), W.shape)
  s_max, t_max = np.unravel_index(W.argmax(), W.shape)
  x = total_scales*total_time


  eps = 1e-5
  f4 = W_sum/(x+eps)               #adding small eps to avoid divide by zero error
  f5 = np.sqrt((np.sum((np.square(f4 - W))))/(x+eps))

  f6 = s_max     
  f7 = t_max     
  f8 = s_min      
  f81 = t_min

  f9 = np.sum(np.multiply(W, np.arange(1,total_scales+1).reshape(-1,1) * np.ones_like(W)))/(x+eps)
  f10 = np.sum(np.square(W))
  f11 = f10/(x+eps)
  f12 = np.sqrt(np.sum(np.square(W))/(x+eps))

  w = np.array(W)
  s = 0
  for i in range(1, total_scales):
    s += np.sum(np.square(w[i,:] - w[i-1,:]))

  consec_scale_diff = np.sqrt(s) 
  
  f14 = consec_scale_diff
  f15 = f14/(x+eps)
  f16 = np.sqrt((f14**2)/(x+eps))
  f17 = np.log10(f14)
  f18 = W_sum
  f19 = accum_phase/phi_sum
  f20 = np.sqrt(accum_sq_phase/phi_sum)
  f21 = phi_sum
  f22 = phi_sum/(np.max(phi))
  f23 = phi_sum/(x+eps)
  f24 = np.sqrt((np.sum(np.square((f22-phi))))/(x+eps))
  f25 = np.sum(np.multiply(phi, np.arange(1,total_scales+1).reshape(-1,1) * np.ones_like(phi)))

  f26 = np.sqrt(np.sum(np.square(s1-s2)))     #euclidean distance between s1 and mcv ?
  f27 = cosine_similarity(s1.reshape(1,-1), s2.reshape(1,-1), dense_output=True).reshape(1,1)[0][0]

  corr = correntropy(s1,s2)
  f28 = np.mean(corr)
  f29 = kurtosis(corr)
  f30 = skew(corr)
  f31 = moment(corr, moment=2)
  f32 = moment(corr, moment=3)
  f33 = trim_mean(corr, 0.1)

  f = [f_t1,f_t2,f_t3,f_t4,f_t5,f_t6,f_t7,f_t8,f_t9,f_t10,f_t11,f_t12,f_t13,
       f1,f2,f3,f4,f5, f6,f7,f8,f81,f9,f10,f11, f12,f14,f15,f16,f17,f18, f19,f20,f21,f22,f23,f24,f25, f26,f27, f28,f29,f30,f31,f32, f33]

  # autoencoder_features = encoded_segs.tolist()
  # #print(autoencoder_features)
  # for enc_feature in autoencoder_features[0]:
  #   f.append(enc_feature)
    
  return np.array(f)