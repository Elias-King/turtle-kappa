#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:36:09 2017

@author: Alexis Klimasewski

inputs: reads in the text files of NE average spectra from record_spectra

method: reads in record spectra and corrects for distances using dread function, 
uses the Andrews 86 method and svd to compute station and event spectra from record spectra, 
plots the event and station spectra, then if a constraint station or event file is specified, 
the output files are rewritten for the constraint and plots are generated again, 
if no constraint, comment out that part of the code

outputs: writes spectra text files for each event and station in event_site_spectra directory
"""
import glob
import os.path as path
import numpy as np
import obspy
from obspy import read    
#import dread
import time
import random
import pandas as pd
from scipy import linalg

#read in the cut and corrected spectra = records
#records are in m/s
#should be binned frequencies and amplitudes
#make list of records and the corresponding events and stations
#working_dir = '/home/eking/Documents/internship/data/events/SNR_5'
working_dir = '/home/eking/Documents/internship/data/events/SNR_10'
outfile_path = working_dir + '/Andrews_inversion'

#list of record files
#ev = glob.glob(working_dir + '/cut_record_spectra/Event*/*')
ev = glob.glob(working_dir + '/cut_record_spectra/Event*/*')
#flatfile =  pd.read_csv(working_dir + '/SNR_5_file.csv')
flatfile =  pd.read_csv(working_dir + '/SNR_10_file.csv')
flatfile = flatfile.drop_duplicates(subset = ['OrgT', 'Name'])
print(len(flatfile))
def secondo(record_path, out_file_path):
    print('Number of records: ', len(record_path))
    #read in all files to find networks and stations
    stationlist = []
    stn_lat = []
    stn_lon = []
    eventidlist = []
    event_lat = []
    event_lon = []
    event_depth = []
    record_freq = []
    record_spec = []
    record_std = []
    numrec = 0
    ##############################################################################
    ## read in the uncut sac files and get the distances between source and station
    
    t1 = time.time()

    
    for i in range(len(record_path)):
        record = (record_path[i].split('/')[-1])
        base = path.basename(record)
        print(base)
        network, station, channel= base.split('_')[0:3]
        yyyy, month, day, hh, mm, ss = base.split('_')[3:]
        ss = ss.split('.')[0]
        eventid = yyyy + '_' + month + '_' + day + '_' + hh + '_' + mm + '_' + ss
        orgt = yyyy + '-' + month + '-' + day + ' ' + hh + ':' + mm + ':' + ss
        
        #read in uncorrected data for header info
        #tr = flatfile.loc[np.where(orgt == flatfile['OrgT']) & np.where(station == flatfile['Name'])]
        tr = flatfile.loc[((orgt == flatfile['OrgT']) & (station == flatfile['Name']))]
        
        
        print(tr)
        evlon =  tr['Qlon'] #deg
        evlat =  tr['Qlat'] #deg
        evdepth = tr['Qdep'] #km
        stlon = tr['Slon'] #deg
        stlat = tr['Slat'] #deg
        stdepth = tr['Selv'] #km
        evdepth = evdepth/1000
        #find distance between event and station
        #dist =  dread.compute_rrup(evlon, evlat, evdepth, stlon, stlat, stdepth) #in km
        #km to m
        dist = tr['rhyp']
        dist = dist * 1000.
        try:
            dist = float(dist)
        except:
            dist = np.mean(flatfile['rhyp'])
        
        #read in file
        data = np.genfromtxt(record_path[i], dtype = float, comments = '#', delimiter = None, usecols = (0,1,2))#only read in first two cols
        if len(data) == 0:
            continue
        record_freq.append(data[:,0])##
        #data is NE spectra
        #square for power spectra
        print(data)
        print(dist)
        ###########################################################################
        #propagation here for errors going lin to log power
        record_spec.append((data[:,1]*dist)**2.)
        std_prop = np.abs(2*(data[:,2])/(data[:,1]*np.log(10)))
        record_std.append(std_prop)#in log power here
        #if network and station not part of the list yet add
        if station not in stationlist:
            stationlist.append(station)
            stn_lat.append(stlat)
            stn_lon.append(stlon)
        if eventid not in eventidlist:
            eventidlist.append(eventid)
            event_lat.append(evlat)
            event_lon.append(evlon)
            event_depth.append(evdepth)
            
    t2 = time.time()
    
    print ('time to read and distance correct all records: ', (t2-t1)/60.)
    print (len(record_freq))
    print (len(record_spec))

    freq_list = record_freq[0]
    print(freq_list)
    F_bins = len(freq_list)
    print(F_bins)
    
    rows = len(record_freq) #testing first 10
    print(rows)
    
    index_matrix = [[0 for j in range(3)] for i in range(rows)]
    
    #for i in range(len(records)):
    for i in range(len(record_spec)):
        record = record_path[i].split('/')[-1]
        base = path.basename(record)
        network, station, channel = base.split('_')[0:3]
        yyyy, month, day, hh, mm, ss = base.split('_')[3:]
        ss = ss.split('.')[0]
        eventid = yyyy + '_' + month + '_' + day + '_' + hh + '_' + mm + '_' + ss
        #make a tuple of record, event, station so indices can be assigned
        index_matrix[i] = [base, eventidlist.index(eventid), stationlist.index(station)]
        
        
    numrec += 1
    print(eventidlist[0])
    print(stationlist)
    
    I = len(eventidlist)#events
    J = len(stationlist)#stations
    #K = len(record_path)#records
    K = len(record_spec)
    
    print ('Number of events: ', I, ' Number of stations: ', J)
    print ('Number of rows (records): ', K, ' Number of cols (events+stations): ', I+J)
    
    #make the G matrix of 1s and 0s and R matrix of records
    G1 = np.zeros((K,I))
    G2 = np.zeros((K,J))

    
    for k in range(K):#for all records
        G1[k][index_matrix[k][1]] = 1 #record row, eventid col
        G2[k][index_matrix[k][2]] = 1 #record row, station col
    
    
    G = np.concatenate((G1,G2), axis = 1)
    
    print(G)
    
    R = np.zeros((K,F_bins))
    cov = np.zeros((K,F_bins))
    
    #populate R matrix with the log of the record spectra (power record spectra)
    for k in range(K):#for all records
        #each row is a record, and col is a frequency band
        #set row equal to the that spectral array
        #here we take the log
        R[k:,] = np.log10(record_spec[k])
        cov[k:,] = record_std[k]
        
    m1 = np.zeros((I+J, F_bins))
    m_cov = np.zeros((I+J, F_bins))
    
    #do the inversion for each freq
    for f in range(F_bins):
        t1 = time.time()
        d = R[:,f]#record for given frequency col
        dT = d.T
        print ('inverting for frequency: ', f, freq_list[f])
        
        ## Inverting - previous version used numpy linalg pinv, which uses 
        ##   the SVD to get the moore-penrose pseudoinverse. This seems unstable
        ##   often, not sure why... SVD won't converge.
        #G_inv = np.linalg.pinv(G, rcond=1e-13)
        ## VJS added this - scipy linalg pinv which just does a least squares solver.
        G_inv = linalg.pinv(G, rcond=1e-13)
        covd = np.diag(cov[:,f])

        covm = np.dot((np.dot(G_inv, covd)), G_inv.T)
        m1[:,f] = np.dot(G_inv,dT)
        m_cov[:,f]= covm.diagonal()
        t2 = time.time()
        print ('time for inversion: (min) ', round((t2-t1)/60., 4))
    
    
    print(m1.shape)
    #now split m into an event matrix and a station matrix
    event = m1[0:I,:] #take first I rows
    station = m1[I:I+J,:]
    event_cov = m_cov[0:I,:]
    station_cov = m_cov[I:I+J,:]
    print(event.shape, station.shape)
    print(event_cov.shape, station_cov.shape)

    for i in range(I):#for each event
        #go from the log of the power spectra to the regular spectra in m
        amp = np.sqrt(np.power(10.0, event[i,:]))

        std = (np.sqrt(np.abs(event_cov[i,:])/2.)*((amp)*(np.log(10))))
        
        outfile = open(outfile_path + '/' + eventidlist[i] + '.out', 'w')
        out = (np.array([freq_list, amp, std])).T
        outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
        np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
        outfile.close()


    print (outfile_path)
    for i in range(J):#for each station
        amp = np.sqrt(np.power(10.0, station[i,:]))

        std1 = np.sqrt((station_cov[i,:]))
        std = np.abs((std1/2.)*(amp)*(np.log(10)))
        outfile = open(outfile_path + '/' + stationlist[i] + '.out', 'w')
        out = (np.array([freq_list, amp, std])).T
        outfile.write('#freq_bins \t vel_spec_NE_m \t stdev_m \n')
        np.savetxt(outfile, out, fmt=['%E', '%E', '%E'], delimiter='\t')
        outfile.close()



secondo(record_path = ev, out_file_path = outfile_path)
#%%