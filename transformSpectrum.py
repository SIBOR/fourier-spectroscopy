# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:54:54 2016

@author: jaymz
"""
import matplotlib.pyplot as plt
import numpy as np
import numexpr
import scipy.signal
from scipy import stats
import sys

#------------------------------------------------------------------------------

#---Input file and user supplied paramters---
folder = '../../FT_HeNeRef3/'
filenames = ['tek0000ALL.csv','tek0003ALL.csv','tek0004ALL.csv','tek0005ALL.csv','tek0006ALL.csv','tek0007ALL.csv']  #input file path
tsVelocityMMsec = [-7.5]*len(filenames)            #Tranlational stage velocity

#filenames = ['../../FT_HeNeRef4/tek0009ALL.csv','../../FT_HeNeRef4/tek0012ALL.csv','../../FT_HeNeRef4/tek0013ALL.csv']  #input file path
#tsVelocityMMsec = [-4.0]*len(filenames)            #Tranlational stage velocity

#----------Plotting Options----------------------------------------------------
plotLabels = []                         #Labels to use in plot legend
plotTitle = '7.5 mm/sec Translational stage velocity'

#----------Reference Settings------------
refWavelengthMM = .6328/1000
refChannel = 'ch2'
interpolationCutoff = 30
#----------------------------------------

#---Plot parameters. Plot range, number of points to plot.---
wF_uM = 2.0
wI_uM = 4.1
#wF_uM = 2.0
#wI_uM = 4.0
#wF_uM = 2.73
#wI_uM = 2.97
#wI_uM = 0.5
#wF_uM = 0.8

numPoints = 10000

#---Program options.---
subtractBaseline = True                 #Removes DC ofset by subtracting the 
                                        #average of the first and last quarters
                                        #of the data. This is critical if this 
                                        #was not done during acquisition.

plotInterferogram = False               #Turns on and off the plotting of the 
                                        #time domain interferogram. Turn this
                                        #off for superimposing multiple plots
                                        #across executions
displayProgress=True
filterSignal = False
fftMode = True

#--------------Command Line Parameter Control----------------------------------

for i in range(1,len(sys.argv)):
    if(sys.argv[i] == '-i'):
        fftMode = False
    elif(sys.argv[i] == '--no-baseline'):
        subtractBaseline = False
    elif(sys.argv[i] == '--filter'):
        filterSignal = True
    elif(sys.argv[i-1] == '-t' or sys.argv[i-1] == '--title'):
        plotTitle = sys.argv[i]
    elif(sys.argv[i] not in ['-b','--no-baseline','--filter','-t','--title']):
        filenames.append(sys.argv[i])

#------------------------------------------------------------------------------

if(wF_uM < wI_uM):
    temp = wF_uM
    wF_uM = wI_uM
    wI_uM = temp

for i in range(0,len(filenames)):
    filenames[i]=folder + filenames[i]
progressString = ''

def getFileData(fName):
    fHandle = open(fName,'r')
    dataOut = {'time':[] , 'ch1':[]}
    foundHeader = False
    for line in fHandle:
        if(foundHeader and len(line) > 2):
            vals = line.split(',')
            dataOut['time'].append(float(vals[0]))
            for j in range(1,len(vals)):
                if(dataOut.has_key('ch'+str(j))):
                    dataOut['ch'+str(j)].append(float(vals[j]))
                else:
                    dataOut['ch'+str(j)]= [float(vals[j])]
        if('TIME,CH' in line):
            foundHeader = True
    fHandle.close()
    for key in dataOut.keys():
        dataOut[key]=np.array(dataOut[key])
    return dataOut

def replaceByRegression(xInput):        
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(0,len(xInput)),xInput)
        return [intercept + slope*i for i in range(0,len(xInput))]

def highPassFilter(y):
    transferNumerator, transferDenominator = scipy.signal.bessel(3,.0005,'highpass')
    y = scipy.signal.lfilter(transferNumerator,transferDenominator,y)
    return y
    
def fourierTrans(datX,datY,wi,wf,n,window = 'none'):
    freqs =np.linspace(wi,wf,n)
    N = len(datX)
    windowFunction = range(0,N)
    if(window == 'none'):
        windowFunction = np.array([1.0]*N)
    elif(window == 'Hann'):
        windowFunction = 0.5*(1.0-np.cos(2*np.pi*np.array(windowFunction)/(N-1)))
    elif(window == 'Hamming'):
        windowFunction = 0.54-0.46*np.cos(2*np.pi*np.array(windowFunction)/(N-1))
    elif(window == 'Welch'):
        windowFunction = 1.0-((np.array(windowFunction)-(N-1)/2)/((N-1)/2))**2
    elif(window == 'Blackman'):
        windowFunction = 0.42  - 0.5*np.cos(2*np.pi*np.array(windowFunction)/(N-1)) + 0.08*np.cos(4*np.pi*np.array(windowFunction)/(N-1))
    elif(window == 'Nuttall'):
        a0, a1, a2, a3 = 0.355768,0.487396,0.144232,0.012604        
        windowFunction = np.array(windowFunction)
        windowFunction = a0 - a1*np.cos(2*np.pi*windowFunction/(N-1))+a2*np.cos(4*np.pi*windowFunction/(N-1))-a3*np.cos(6*np.pi*windowFunction/(N-1))
    wD = [0.0]*n
    pi = np.pi
    progress =0.0
    for i in range(0,n):
        w=freqs[i]
        #This line uses the numexpr library for fast evaluation of the fourier
        #integral. Some cleverness may be needed to use this with the
        #trapezoidal integration rule or anything more advanced.
        sum=np.sum(numexpr.evaluate('datY*windowFunction*exp(1j*2000*pi*datX/w)'))/len(datX)
        wD[i]= sum
        if(displayProgress):
            progress=float(i)/n
            print progressString+'\t'+str(progress*100) +"%"
    return {'f':freqs , 'y':np.array(np.array(wD))}
    
def resample(x,y,scan):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    sR = abs(x[-1]-x[0])/len(x)
    n = int(scan/sR)
    xNew = np.linspace(-scan/2,scan/2,n)
    yNew = np.interp(xNew,x,y)
    zerosF = np.greater(xNew,x[0])
    zerosL = np.less(xNew,x[-1])
    yNew = numexpr.evaluate('zerosF*yNew*zerosL')
    
    return {'x' : xNew , 'y' : yNew}

def resampleUniform(x,y):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    sR = abs(x[-1]-x[0])/len(x)
    xNew = np.linspace(x[0],x[-1],len(x))
    yNew = np.interp(xNew,x,y)
    
    return {'x' : xNew , 'y' : yNew} 
    
def fft(datX,datY):
    rangeX=datX[-1]-datX[0]
    ft = np.fft.fft(datY)
    ft = ft[1::]
    return {'f' : [1000*rangeX/i for i in range(1,len(datX))[::-1]], 'y' : ft[::-1]}

def trimSpectrum(f,y):
    xOut, yOut = [], []
    for i in range(0,len(f)):
        if(f[i] > wI_uM and f[i] < wF_uM):
            xOut.append(f[i])
            yOut.append(y[i])
    return {'f' : xOut, 'y' : yOut}
            
    
def writeSpectrum(spectrum, filename):
    file = open(filename,'w')
    file.write('#Wavelength(uM)\tAmplitude\n')
    for i in range(0,len(spectrum['f'])):
        file.write(str(spectrum['f'][i]) + "\t" + str(np.abs(spectrum['y'][i])) + '\n')
    file.close()

def writeDataset():
    for i in range(0,len(filenames)):
        writeSpectrum(dataset[i],filenames[i].replace('.csv','_spectrum.dat'))
    
def plotDataset():
    for i in range(0,len(dataset)):
        plt.semilogy(dataset[i]['f'],np.abs(dataset[i]['y']),label=dataset[i]['label'])
    plt.xlabel('Wavelength (uM)')
    plt.ylabel('Spectral density')
    plt.title(plotTitle)
    plt.legend()
    
def subtractBaseline(y):
    baseline = np.mean([np.mean(y[1:len(x)/4]),np.mean(y[len(x)*3/4:-1])])
    return y - baseline
    
def getZeroCrossings(x,y):
    inversionPoints = []
    for i in range(1,len(x)):
        if(y[i-1]*y[i]<0):
            inversionPoints.append((x[i]*y[i-1]-x[i-1]*y[i])/(y[i-1]-y[i]))
    return inversionPoints

def constructPhase(zeroCrossings,x):
    phase = [np.pi*i for i in range(0,len(zeroCrossings))]
    interpFunc =  scipy.interpolate.splrep(zeroCrossings,phase,s=0)
    return scipy.interpolate.splev(x,interpFunc,der=0)

def reconstructDistance(x,y,refWavelength,cutoffInterval = 30):
    if(x[0]>x[-1]):
        x = x[::-1]
        y = y[::-1]
    yDat = y - np.mean(y)
    yDat = highPassFilter(yDat)
    zc = getZeroCrossings(x,yDat)
    phi = constructPhase(zc,x[cutoffInterval:-cutoffInterval])
    distOut = phi/(2*np.pi)*refWavelength
    distOut = distOut - distOut[len(distOut)/2]
    return {'distance' : distOut, 'x' : x[cutoffInterval:-cutoffInterval], 'y' : y[cutoffInterval:-cutoffInterval]}

#def stitchDR(x,y1,y2):

dataset = []

if(isinstance(filenames,str)):
    filenames=[filenames]
elif(isinstance(filenames,list)):
    plotInterferogram=False
    
for i in range(0,len(filenames)):
    progressString = 'file: '+filenames[i]+'\t'+str(i+1)+'/'+str(len(filenames))
    filename=filenames[i]
    print '------------------------------------------'
    print 'Retrieving file data for file ' + str(i+1)+'/'+str(len(filenames)) + ' : '+filename
    dat = getFileData(filename)
    dat['time'] = np.array(replaceByRegression(dat['time']))
    
    x, y = None, dat['ch1']
    if(refChannel == None):
        print 'No reference specified. Using translational stage velocity'
        x = dat['time']*2*tsVelocityMMsec[i]
    else:
        print 'Calibrating reference Interferogram'
        dm = reconstructDistance(dat['time'],dat[refChannel],refWavelengthMM,interpolationCutoff)
        x, y = dm['distance'], y[interpolationCutoff:-interpolationCutoff]

    
    if(subtractBaseline):
        y = subtractBaseline(y)
    
    dm = None
    
    if(plotInterferogram):
        plt.plot(x,y)
        plt.xlabel('Path Difference (mm)')
        plt.ylabel('Intensity')
        plt.show()
    if(filterSignal):
        transferNumerator, transferDenominator = scipy.signal.bessel(3,.0005,'highpass')
        y = scipy.signal.lfilter(transferNumerator,transferDenominator,y)
    
    if(fftMode):
        print 'Resampling data'
        resamp = resampleUniform(x,y)
        x, y = resamp['x'], resamp['y']
        print 'Transforming File ' + str(i+1) + '/'+str(len(filenames))
        ft = fft(x,y)
    else:
        ft = fourierTrans(x,y,wI_uM,wF_uM,numPoints,window='Hann')
    #resampled = resample(replaceByRegression(x),y,1200.0)
    #ft = fft(resampled['x'],resampled['y'])
    #ft = ftC(x,y,wI_uM,wF_uM,numPoints)
    if(plotInterferogram):
        plt.figure(2)
    lbl = filenames[i]
    if(len(plotLabels)==len(filenames)):
        lbl = plotLabels[i]
    ft['label'] = lbl
    dataset.append(trimSpectrum(ft['f'],ft['y']))
    dataset[-1]['label']=lbl
    plt.semilogy(ft['f'],np.abs(ft['y'])**2,label=lbl)
    plt.xlabel('Wavelength (uM)')
    plt.ylabel('Spectral Density')
    if(fftMode):
        plt.xlim(wI_uM,wF_uM)

if(len(plotTitle) > 0):
    plt.title(plotTitle)
plt.legend()
print '\nGenerating Plot...'
plt.show()
