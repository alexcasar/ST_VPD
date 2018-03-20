"""
Simple voice detection and pitch estimation feature extractor
"""

from __future__ import print_function, division
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, hamming

__author__ = "Jose A. R. Fonollosa & Alejandro Casar"

def zero_crossing(frame, sfreq): #rate, lagg
    zrate = 1;
    zlagg = 0;
    prev = 0;
    for i in range(1,len(frame)):
        if(frame[i-1]>0 and frame[i]<0):
            zlagg = zlagg + i-prev
            prev = i
            zrate = zrate+1
    zlagg = zlagg/zrate; 

    return zrate, zlagg

def cepstrum(frame, sfreq): #cepts, peak, freq
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if amax > 0:
        frame /= amax
    else:
        return 0
        
    logspect = np.log(np.abs(np.fft.rfft(frame*hamming(len(frame)))))
    cepstrum = np.abs(np.fft.irfft(logspect))
    cepts = np.max(cepstrum[50:400])
    peak = np.argmax(cepstrum[50:400])
    f0 = sfreq / (peak + 50)

    return cepts, peak, f0

def amdf(frame, sfreq): #amdf, pitch
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if amax > 0:
        frame /= amax
    else:
        return 0

    amdfmin = 100000;
    pitchmin = 0;
    passed = 0;

    for i in range(50,400):
        passed = 0;
        frame1 = frame[i:];
        frame2 = frame[:-i];
        amdfval = sum(abs(frame2-frame1))
        if amdfval < amdfmin:
            amdfmin = amdfval;
            pitchmin = i;

    return amdfmin, pitchmin

def energy(frame, sfreq): #energy
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if amax > 0:
        frame /= amax
    else:
        return 0
    
    energy = 0.0

    for i in range (0,len(frame)):
        energy += frame[i]*frame[i];

    return energy

def autocorr(frame, sfreq): #cor, peak, rmax, freq
    frame = frame.astype(np.float)
    frame -= frame.mean()
    amax = np.abs(frame).max()
    if amax > 0:
        frame /= amax
    else:
        return 0

    corr = correlate(frame, frame)
    corr = corr[len(corr)//2:]

    dcorr = np.diff(corr)
    rmin = np.where(dcorr > 0)[0]
    if len(rmin) > 0:
        rmin1 = rmin[0]
    else:
        return 0

    cor = np.max(corr[rmin1:]) + rmin1
    peak = np.argmax(corr[rmin1:]) + rmin1
    rmax = corr[peak]/corr[0]
    f0 = sfreq / peak
    return cor, peak, rmax, f0

def getFeatures(frame, sfreq):
    zrate, zlag = zero_crossing(frame, sfreq)
    cepts, cpeak, cfreq = cepstrum(frame, sfreq)
    amdfval, apitch = amdf(frame, sfreq)
    egy = energy(frame, sfreq)
    acorr, apeak, rmax, afreq = autocorr(frame, sfreq)
    return zrate, zlag, cepts, cpeak, cfreq, amdfval, apitch, egy, acorr, apeak, rmax, afreq

def wav2f0(options, gui):
    if options.action == 0:
        classbase = os.path.join(options.datadir, "classbase.csv")
        pitchbase = os.path.join(options.datadir, "pitchbase.csv")
        classfile = open(classbase,'wt')
        pitchfile = open(pitchbase,'wt')
        print('zrate, zlag, cepts, cpeak, cfreq, amdfval, apitch, egy, acorr, apeak, rmax, afreq, label',file=classfile)
        print('zrate, zlag, cepts, cpeak, cfreq, amdfval, apitch, egy, acorr, apeak, rmax, afreq, pitch',file=pitchfile)
        with open(gui) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                filename = os.path.join(options.datadir, line + ".wav")
                f0_filename = os.path.join(options.datadir, line + ".f0")
                f0ref_filename = os.path.join(options.datadir, line + ".f0ref")
                print("Processing:", filename, '->', f0_filename)
                sfreq, data = wavfile.read(filename) #sfreq = sample rate = 20000 en fda
                ref = open(f0ref_filename, 'r')
                labels = ref.read().splitlines()
                iLabel = 0
                with open(f0_filename, 'wt') as f0file:
                    nsamples = len(data)
                    ns_windowlength = int(round((options.windowlength * sfreq) / 1000)) #windowlength default = 32
                    ns_frameshift = int(round((options.frameshift * sfreq) / 1000)) #frameshift default = 15
                    ns_padding = int(round((options.padding * sfreq) / 1000)) #padding default = 16
                    for ini in range(-ns_padding, nsamples - ns_windowlength + ns_padding + 1, ns_frameshift):
                        if iLabel < len(labels):
                            first_sample = max(0, ini)
                            last_sample = min(nsamples, ini + ns_windowlength)
                            frame = data[first_sample:last_sample]
                            feat = getFeatures(frame, sfreq)
                            row = ""
                            for it in range(0,len(feat)):
                                row = row + str(feat[it]) + ","
                            if float(labels[iLabel]) == 0:
                                classrow = row + 'unvoiced'
                            else:
                                classrow = row + "voiced"
                            pitchhrow = row + labels[iLabel]
                            print(pitchhrow,file=pitchfile) 
                            print(classrow,file=classfile)
                            iLabel = iLabel+1
        classfile.close()
        pitchfile.close()
    else:
        full = open("data/full.f0", 'r')
        f0 = np.asarray(full.read().splitlines())
        fit = 0
        with open(gui) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                f0_filename = os.path.join(options.datadir, line + ".f0")
                f0ref_filename = os.path.join(options.datadir, line + ".f0ref")
                print("Processing:", "data/full.f0", '->', f0_filename)
                ref = open(f0ref_filename, 'r')
                freqs = ref.read().splitlines()
                with open(f0_filename, 'wt') as f0file:
                    for ini in range(0,len(freqs)):
                        print(float(f0[fit]+'\r\n'),file=f0file)
                        #print(f0[fit])
                        fit = fit+1
                    #print((str(0)+'\r\n'),file=f0file);
        full.close()


def main(options, args):
    wav2f0(options, args[0])

if __name__ == "__main__":
    import optparse
    optparser = optparse.OptionParser(
        usage='python3 %prog [OPTION]... FILELIST\n' + __doc__)
    optparser.add_option(
        '-w', '--windowlength', type='float', default=32,
        help='windows length (ms)')
    optparser.add_option(
        '-a', '--action', type='float', default=0,
        help='create features (0) or create f0 files (1)')
    optparser.add_option(
        '-f', '--frameshift', type='float', default=15,
        help='frame shift (ms)')
    optparser.add_option(
        '-p', '--padding', type='float', default=16,
        help='zero padding (ms)')
    optparser.add_option(
        '-d', '--datadir', type='string', default='data',
        help='data folder')

    options, args = optparser.parse_args()

    if len(args) == 0:
        print("No FILELIST provided")
        optparser.print_help()
        exit(-1)

    main(options, args)
