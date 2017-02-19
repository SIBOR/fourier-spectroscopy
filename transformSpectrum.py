# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:54:54 2016

@author: jaymz
"""
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne

filenames = ['../1-17/tek0013CH3.csv','../1-17/tek0014CH3.csv','../1-17/tek0015CH3.csv','../1-17/tek0016CH3.csv','../1-17/tek0017CH3.csv']
plotLabels = ['No Scanning','No Scanning','Scanning','Scanning','Scanning']
tsVelocityMMsec = .05

wF_uM = 2.6
wI_uM = 4.0
numPoints = 10
subtractBaseline = True
plotInterferogram = False

c=299792458.0

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
    for key in dataOut.keys():
        dataOut[key]=np.array(dataOut[key])
    return dataOut
    
def fourierTrans(datX,datY,wi,wf,n):
    freqs =np.linspace(wi,wf,n)
    wD = []
    pi = np.pi
    for w in freqs:
        sum=np.sum(ne.evaluate('datY*exp(1j*2000*pi*datX/w)'))/len(datX)
        wD.append(sum)
    return {'f':freqs , 'y':np.array(np.array(wD))}
    
def fft(datX,datY):
    rangeX=datX[-1]-datX[0]
    ft = np.fft.fft(datY)
    ft = ft[1::]
    return {'f' : [rangeX/i for i in range(1,len(datX))[::-1]], 'y' : ft[::-1]}    

if(isinstance(filenames,str)):
    filenames=[filenames]
elif(isinstance(filenames,list)):
    plotInterferogram=False
    
for i in range(0,len(filenames)):
    filename=filenames[i]
    dat = getFileData(filename)
    x,y = dat['time']*2*tsVelocityMMsec,dat['ch1']
    
    baseline = np.mean([np.mean(y[1:len(x)/4]),np.mean(y[len(x)*3/4:-1])])
    if(subtractBaseline):
        y = y - baseline
    
    if(plotInterferogram):
        plt.plot(x,y)
        plt.xlabel('Path Difference (mm)')
        plt.ylabel('Intensity')
        plt.show()
    
    ft = fourierTrans(x,y,wI_uM,wF_uM,numPoints)
    #ft = fft(x,y)
    #ft = ftC(x,y,wI_uM,wF_uM,numPoints)
    if(plotInterferogram):
        plt.figure(2)
    if(len(plotLabels)==len(filenames)):
        plt.plot(ft['f'],np.abs(ft['y'])**2,label=plotLabels[i])
    else:    
        plt.plot(ft['f'],np.abs(ft['y'])**2)
    plt.xlabel('Wavelength (uM)')
    plt.ylabel('Spectral Density')

plt.legend()
plt.show()
