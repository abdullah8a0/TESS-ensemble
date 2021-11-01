# py_code
ML Pipeline designed to solve the problem of detecting transients in light curve data from TESS (Transiting Exoplanet Survey Satellite)




Instructions:
Set base in run_classif to where your sector files are.
Install HDBSCAN package (“pip install hdbscan”)
Run run_classif.

Notes:
The pipeline uses all cores available, so while the program is running you won't be able to do much.
Time to run sector 36. 
A light curve is initiated as an LC(sector, cam, ccd, col, row) object in lcobj.py. 




Pipeline (Behind the scenes documentation. Do not follow, for reference only):
Change the variable base in lcobj to the folder where all the sec_transient_lc are. Change working_folder to where hdbscan_classif is.
Set the sector variable in feature_extract_scaler and run the file.
Set the sectors list in cluster_anomaly (only keep one sector in the list for now) and the plot_flag boolean. Run the python file.
Run cleanup_anomaly and input the sector number. This removes obvious lcs missed by the clustering algorithm.
The final lcs are in the Results folder.

LC:

Takes flux, time and background clips it according to background and saves them as attributes.

Self.smooth_flux is the flux passed through a Savitzky–Golay filter with a cubic polynomial used for smoothing.

Self.normed_flux is the flux normalized to [-1,1]

Useful methods:

self.plot()	
 Plots the flux on matplotlib

self.make_percentiles()	 
Creates self.percentiles, a dictionary mapping percentages to values.
Flags instance as self.is_percentile = True

self.pad_flux() 
Pads normed_flux with zeros as per Rahul’s code.
Creates self.padded_flux
Flags instance as self.is_padded = True

self.make_FFT()
Depends on self.pad_flux() being run before this !!!
Creates self.fft_freq which is a tuple (frequencies list, power list).
Creates self.significant_frequencies as top 5 most powerful frequencies. In the same format as self.fft_freq.
Creates self.phase_space as folded flux on the most significant frequency.
Flags instance as self.is_FFT = True

self.plot_FFT()
Plots FFT power plot.

self.plot_phase()
Plots the self.phase_space flux (flux folded on predicted frequency)

self.smooth_plot()
Plots the smoothed out flux.


Features:-	(Total = 30, after dim reduction = 15)

Amp: 
The peak to peak amplitude of flux divided by 2.
Better_amp:
The amplitude of the median of top 5th percentile to the median of bottom 5th percentile.
Med:
Median
Mean:
Mean of original flux
Std:
Standard deviation
Slope:
Slope of a linear regression fit
R:
Correlation Coefficient of a linear fit.
Skew:
Standard Skew
Max_slope:
	Biggest absolute slope in the data.
Beyond1std:
Fraction of data points beyond 1 standard deviation from the mean (weighted by error).
Delta_quartiles:
Inter quartile range
flux_mid_(x):
The ratio of the middle x percentile range to 5th - 95th percentile. I take x = {20,35,50,65,80}. Formula;

a = percentile[50+(x/2)] - percentile[50-(x/2)]	# middle x percentile range
b = percentile[95] - percentile[5]

return a/b

Cons:
Ratio of the number of Kim’s consecutives to the number of datapoints. 

A Kim’s consecutive is a consecutive triplet of data points (flux[i], flux[i+1], flux[i+2]) such that all 3 are more than 2 standard deviations from the mean.
Slope_trend:
	Fraction of positive slopes minus the fraction of negative slope
Var_ind:
Von Neumann’s Variability index for unevenly spaced data (~2 for normal distribution)
Med_abs_dev:
The median of differences from the median. Formula:

= median( |flux - median(flux)| )
H1:
	The power of the most significant frequency in FFT
R21:
Ratio of the power of 2nd most significant frequency to the most significant frequency.
R31:
Ratio of the power of 3rd most significant frequency to the most significant frequency.

Rcs:
	Range of Cumulative sum. (~0 for symmetric distributions)
L:
Auto-Correlation Length. The autocorrelation or serial correlation, is the linear dependence of a signal with itself at two points in time. It represents how similar the observations are as a function of the time lag between them. It is used for detecting non-randomness in data or to find repeating patterns.

Med_buffer_ran:
Fraction of datapoints ‘close’ to the median. (within 1/10 of amplitude)
Perr:
Correlation coefficient of fitting a step function to smooth flux.
StetK:
StetsonK, a robust way to calculate kurtosis for uneven data. Kurtosis is the measure of how well the outliers of the data fit the normal distribution.
P_ander:
Anderson-Darling test statistic. Anderson-Darling is a very sensitive way to see departure from normal distribution. (~0.25 for normal distribution)
Days_of_i:
Fractions of days that are flagged as ‘days of interest’.

A day of interest is when the day’s average flux is more than 1 standard deviation from the global mean. (could possibly change this definition)



Processed:

Tag, cluster, isAnom, isRTS, isHTP
