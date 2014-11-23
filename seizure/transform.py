'''
Created on Sep 4, 2014

@author: newuser
'''
from collections import namedtuple

import numpy
import scipy
from scipy.signal import resample, welch
from scipy.stats import moment
from sklearn.decomposition import FastICA, PCA
try:
    #Optional import for wavelet transform
    import pywt
except:
    pass
try:
    from pykalman import KalmanFilter
except:
    pass

DataSet = namedtuple("DataSet", ["data", "data_length_sec", "sampling_rate", "channels", "sequence", "name"])

class Transform(object):
    """
    Converts input data to output data as a function of prior transforms, which
    are defined using the requires() function.
    """
    def get_name(self):
        pass
    
    def requires(self):
        return []
    
    def apply(self, data):
        raise NotImplementedError("You forgot to override this method.")
    
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__
    
class _IdentityTransform(Transform):
    """
    Transform whose input is identical to output. Used internally.
    """
    def get_name(self):
        return "identity"
    
    def apply(self, data):
        return data

class ResampledDataSet(Transform):
    """
    Resamples time series dataset to target sampling rate.
    """
    def __init__(self, sampling_rate = 200):
        self.f = sampling_rate
        
    def requires(self):
        return []
        
    def get_name(self):
        return 'resample-%0.1f' % self.f
    
    def apply(self, dataset):
        data = dataset.data
        axis = data.ndim - 1
        if dataset.sampling_rate > self.f:
            data = resample(data, self.f*dataset.data_length_sec, axis=axis)
        return DataSet(data, dataset.data_length_sec, self.f, dataset.channels, dataset.sequence, dataset.name)

class DatasetData(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, dataset):
        return dataset.data

class Shift(Transform):
    """
    Shifts all rows of the dataset by the given amount, which may be useful for bootstrapping.
    """
    def __init__(self, shift):
        self.s = shift
        
    def requires(self):
        return []
    
    def get_name(self):
        return 'shift-%d' % self.r
    
    def apply(self, dataset):
        n = len(dataset.data)
        data = dataset.data[[(i+self.s) % n for i in range(0,n)]]
        return DataSet(data, dataset.data_length_sec, dataset.sampling_rate, dataset.channels, dataset.sequence, dataset.name)

class TimeOutliers(Transform):
    def requires(self):
        return [ResampledDataSet()]
        
    def get_name(self):
        return "TimeOutliers"
        
    def apply(self, dataset):
        data = dataset.data
        output = []
        for channel in data:
            abschannel = numpy.absolute(channel)
            m = numpy.mean(abschannel)
            output.append(abschannel > (m*2))
        return numpy.array(output)

class FFT(Transform):
    """
    Apply a 1D Fast Fourier Transform to each channel then limit max frequency.
    """
    def __init__(self, max_frequency_limit = 48):
        self.max_frequency_limit = max_frequency_limit
    
    def get_name(self):
        return "fft"
    
    def requires(self):
        return [ResampledDataSet()]

    def apply(self, dataset):
        output = []
        # Last FFT bin we sample from. The formula is max frequency/frequency resolution
        # with frequency resolution equal to 1/data time length.
        #max_index = int(math.floor(self.max_frequency_limit*dataset.data_length_sec))
        # Apply FFT to each channel of ECG recording
        for channel_row in dataset.data:
            axis = channel_row.ndim - 1
            fft = numpy.fft.rfft(channel_row, axis=axis)
            #fft = numpy.fabs(fft[1:max_index])
            fft = numpy.abs(fft)
            output.append(fft)
        return numpy.array(output)
    
class DWT(Transform):
    """
    Apply Discrete Wavelet Transform to the last axis.
    """
    def __init__(self, wavelet_name="db4"):
        self.wavelet = pywt.Wavelet(wavelet_name)
    
    def get_name(self):
        return "dwt"
    
    def requires(self):
        return [ResampledDataSet()]

    def apply(self, dataset):
        output = []
        for channel in dataset.data:
            output.append(pywt.dwt(channel, wavelet=self.wavelet))
        return numpy.array(output)
    
class Moments(Transform):
    """
    Calculate the moments of the time series data
    """
    def __init__(self, num_moments = 4):
        self.num_moments = num_moments
    
    def get_name(self):
        return "moments-%d" % self.num_moments
    
    def requires(self):
        return [ResampledDataSet()]

    def apply(self, dataset):
        axis = dataset.data.ndim - 1
        output = []
        for i in range(0, self.num_moments+1):
            output.append(moment(dataset.data, i, axis=axis))
        return numpy.array(output)

class FrequencyCorrelation(Transform):
    """
    Calculate frequency correlation coefficients matrix across all EEG channels.
    """
    def __init__(self, max_frequency_limit = 48):
        self.max_frequency_limit = max_frequency_limit
    
    def get_name(self):
        return 'freq-corr-%s' % self.max_frequency_limit
    
    def requires(self):
        return [FFT(max_frequency_limit=self.max_frequency_limit)]

    def apply(self, data):
        return numpy.corrcoef(data)

class TimeCorrelation(Transform):
    """
    Calculate time correlation coefficients matrix across all EEG channels.
    """
    def get_name(self):
        return 'time-corr'
    
    def requires(self):
        return [ResampledDataSet()]
    
    def apply(self, dataset):
        return numpy.corrcoef(dataset.data)
    
class PowerBand(Transform):
    """
    Splits the FFT into n bands, each representing a range of frequencies,
    then computes power using the Welch algorithm.
    """
    def __init__(self, dependency, n = 5):
        self.numbands = n
        self.dependency = dependency
    
    def get_name(self):
        return 'freq-band-power'
    
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        bands = numpy.array_split(data, self.numbands, axis=1)
        output = numpy.zeros([len(bands), len(bands[0])])
        for i, band in enumerate(bands):
            for j, channel in enumerate(band):
                f, Pwelch_spec = welch(channel, nperseg=min(len(channel),256))
                diff = f[-1] - f[0]
                power = numpy.trapz(Pwelch_spec, f)/diff
                output[i][j] = power
        return output
    
class FrequencyBandPower(Transform):
    """
    Splits the FFT into n bands, each representing a range of frequencies,
    then computes power using the Welch algorithm.
    """
    def __init__(self, n = 5):
        self.numbands = n
    
    def get_name(self):
        return 'freq-band-power'
    
    def requires(self):
        return [PowerBand(FFT(),self.numbands)]
    
    def apply(self, data):
        return data
    
class TimeBandPower(Transform):
    """
    Splits the time series into n bands, each representing a range of frequencies,
    then computes power using the Welch algorithm.
    """
    def __init__(self, n = 5):
        self.numbands = n
    
    def get_name(self):
        return 'timebandpower'
    
    def requires(self):
        return [ResampledDataSet(), PowerBand(DatasetData(ResampledDataSet()),self.numbands)]
    
    def apply(self, inputs):
        dataset, power = inputs
        return power/(dataset.data_length_sec*dataset.sampling_rate)
    
class TimeBandPowerDiff(Transform):
    """
    Calculates the squared difference between power in adjacent time bands, multiplies
    by the sign of the difference, then calculates sum to determine if power is increasing or decreasing.
    This indicates if power increases with time and whether the changes are positive or negative.
    """
    def get_name(self):
        return 'timebandpowerdiff'
    
    def requires(self):
        return [TimeBandPower()]
    
    def apply(self, data):
        data = data/numpy.min(data)
        output = numpy.zeros([len(data), 1])
        for j, channel in enumerate(data):
            diff_sum = 0
            for i in range(0,len(channel)-1):
                diff = channel[i+1] - channel[i]
                diff_sum = diff_sum + numpy.sign(diff)*numpy.square(diff)
            output[j] = diff_sum
        return output

class PowerSpectrumCorrelation(Transform):
    def get_name(self):
        return 'power-spectrum-correlation'
    
    def requires(self):
        return [ResampledDataSet()]
    
    def apply(self, dataset):
        output = []
        for channel in dataset.data:
            output.append(welch(channel))
        return numpy.corrcoef(numpy.array(output))
    
class Autocorrelation(Transform):
    def get_name(self):
        return 'autocorrelation-time'
    
    def requires(self):
        return [ResampledDataSet()]
    
    def apply(self, dataset):
        output = []
        for channel in dataset.data:
            result = numpy.correlate(channel, channel, mode='full')
            result = result[result.size/2:]
            output.append(result)
        return numpy.array(output)

class DecorrelationTime(Transform):
    def get_name(self):
        return 'decorrelation-time'
    
    def requires(self):
        return [ResampledDataSet(), Autocorrelation()]
    
    def apply(self, data):
        dataset, autocorrelation = data
        output = []
        for channel in autocorrelation:
            sign = numpy.sign(channel[0])
            for i, sample in enumerate(channel):
                if numpy.sign(sample) != sign:
                    result = i/dataset.data_length_sec/dataset.sampling_rate
                    output.append(result)
        return numpy.array(output)

class SpectralPower(Transform):
    def get_name(self):
        return "spectralpower"
    
    def requires(self):
        return []
    
    def apply(self, data):
        pass
    
class AverageEnergy(Transform):
    """
    Represents the average energy of the signal
    """
    def get_name(self):
        return 'average-energy'
    
    def requires(self):
        return [ResampledDataSet()]
    
    def apply(self, dataset):
        return numpy.sum(dataset.data, axis=0)/dataset.data_length_sec/dataset.sampling_rate
    
class LowPassFilter(Transform):
    """
    Applies a low pass filter to the input data.
    """
    pass

class UpperTriangle(Transform):
    """
    Obtains the flattened upper right triangle of an array. Typically used with
    a correlation coefficient matrix.
    """
    def __init__(self, dependency):
        self.dependency = dependency
        
    def get_name(self):
        return 'uppertriangle-%s' % self.dependency.get_name()
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        output = []
        for i in range(0, len(data)-1):
            for j in range(i+1,len(data)):
                output.append(data[i][j])
        return numpy.array(output)
    
class LowerTriangle(Transform):
    """
    Obtains the flattened lower left triangle of an array.
    """
    def __init__(self, dependency):
        self.dependency = dependency
        
    def get_name(self):
        return 'lowertriangle-%s' % self.dependency.get_name()
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        output = []
        for i in range(0, len(data)):
            for j in range(0,i+1):
                output.append(data[i][j])
        return numpy.array(output)
    
class WaveletCoefficients(Transform):
    pass

class Log10(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
    
    def get_name(self):
        return "log10-%s" % self.dependency.get_name()
    
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        return numpy.log10(data)
    
class Eigenvalues(Transform):
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """
    def __init__(self, dependency):
        self.dependency = dependency
    
    def requires(self):
        return [self.dependency]
    
    def get_name(self):
        return 'eigenvalues-%s' % self.dependency.get_name()

    def apply(self, data):
        w, __ = numpy.linalg.eig(data)
        w = numpy.absolute(w)
        w.sort()
        return w
    
class STFT(Transform):
    def __init__(self, framesz=100.0, hop=50.0, resample=0):
        self.framesz = framesz
        self.hop = hop
        self.resample = resample
    
    def get_name(self):
        return 'stft-%s-%s-%s' % (self.framesz, self.hop, self.resample)
    
    def requires(self):
        if self.resample:
            return [ResampledDataSet(sampling_rate=self.resample)]
        return []
     
    def apply(self, dataset):
        data = dataset.data
        output = []
        for channel in data:
            output.append(STFT.stft(channel, dataset.sampling_rate, self.framesz, self.hop))
        return numpy.array(output)
    
    @staticmethod
    def stft(x, fs, framesz, hop):
        """
         x - signal
         fs - sample rate
         framesz - frame size
         hop - hop size (frame size = overlap + hop size)
        """
        framesamp = int(framesz*fs)
        hopsamp = int(hop*fs)
        w = scipy.hamming(framesamp)
        X = numpy.array([scipy.fft(w*x[i:i+framesamp]) 
          for i in range(0, len(x)-framesamp, hopsamp)])
        return numpy.median(X, axis=0)

class STFTBands(Transform):
    def __init__(self, nbands = 5, framesz=100.0, hop=50.0, resample=0):
        self.framesz = framesz
        self.hop = hop
        self.resample = resample
        self.numbands = nbands
    
    def get_name(self):
        return 'stftbands-%s-%s-%s-%s' % (self.framesz, self.hop, self.resample, self.numbands)
    
    def requires(self):
        return [PowerBand(STFT(self.framesz, self.hop, self.resample))]
    
    def apply(self, output):
        medians = numpy.median(output, axis=1)
        maxes = numpy.percentile(output, 90, axis=1)
        mins = numpy.percentile(output, 10, axis=1)
        midmaxes = numpy.percentile(output, 70, axis=1)
        midmins = numpy.percentile(output, 30, axis=1)
        return numpy.concatenate((medians, maxes, mins, midmaxes, midmins),axis=1)
    
class Real(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        return data.real
    
class Imag(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        return data.imag
    
class Magnitude(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        return numpy.abs(data)
    
class Correlation(Transform):
    def __init__(self, dependency):
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
    
    def apply(self, data):
        return numpy.corrcoef(data)
    
class FFTHistogram(Transform):
    pass

class ICA(Transform):
    def __init__(self, dependency, n_components=6):
        self.ica = FastICA(n_components)
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
        
    def apply(self, data):
        return self.ica.fit_transform(data.T).T

class ICASTFTBands(Transform):
    def __init__(self, nbands = 5, framesz=100.0, hop=50.0, resample=0):
        self.framesz = framesz
        self.hop = hop
        self.resample = resample
        self.numbands = nbands
    
    def get_name(self):
        return 'stftbands-ica-%s-%s-%s-%s' % (self.framesz, self.hop, self.resample, self.numbands)
    
    def requires(self):
        return [PowerBand(ICA(Magnitude(STFT(self.framesz, self.hop, self.resample))))]
    
    def apply(self, data):
        return data

class PCA(Transform):
    def __init__(self, dependency, n_components=3):
        self.pca = PCA(n_components)
        self.dependency = dependency
        
    def requires(self):
        return [self.dependency]
        
    def apply(self, data):
        return self.pca.fit_transform(data.T).T
    
class PCASTFTBands(Transform):
    def __init__(self, nbands = 5, framesz=100.0, hop=50.0, resample=0):
        self.framesz = framesz
        self.hop = hop
        self.resample = resample
        self.numbands = nbands
    
    def get_name(self):
        return 'stftbands-pca-%s-%s-%s-%s' % (self.framesz, self.hop, self.resample, self.numbands)
    
    def requires(self):
        return [PowerBand(PCA(Magnitude(STFT(self.framesz, self.hop, self.resample))))]
    
    def apply(self, data):
        return data
    
class Kalmam(Transform):
    def requires(self):
        return []
    
    def apply(self, dataset):
        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=len(dataset.data))
        predicted = kf.em(dataset.data).smooth(dataset.data)
        return data