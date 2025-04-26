'''Mars timing information based on MARS24: http://www.giss.nasa.gov/tools/mars24/help/algorithm.html

Contains several functions for calculating Mars time parameters from Earth time and vice versa.
Probably the most useful functions are:
getMarsSolarGeometry: gets mars time data given a 6 element time list
getSZAfromTime: gets the SZA from a 6 element time list and coordinates
getLTfromTime: gets the LTST from a 6 element time list and longitude
getUTCfromLS: Estimates the Earth time from LS and a Mars year
'''

import datetime
import numpy as np
from collections import namedtuple
import taiutc
from astroquery.jplhorizons import Horizons
from astropy.utils import iers
import astropy.time as aptime
import spiceypy as spice
import os
from matplotlib import pyplot as pp 

# Path to the manually downloaded IERS-A file needed for correct leap seconds
module_dir = os.path.dirname(__file__)
iers_path = os.path.join(module_dir, 'iers_a.txt')

# Load and use the IERS-A file
iers_table = iers.IERS_A.open(iers_path)
iers.IERS_Auto.iers_table = iers_table
iers.conf.auto_download = False
iers.conf.use_iers_auto = True

d2R = np.pi/180.

def getMarsParams(j2000):
	'''Mars time parameters'''


	Coefs = np.array(
	[[0.0071,2.2353,49.409],
	[0.0057,2.7543,168.173],
	[0.0039,1.1177,191.837],
	[0.0037,15.7866,21.736],
	[0.0021,2.1354,15.704],
	[0.0020,2.4694,95.528],
	[0.0018,32.8493,49.095]])

	dims = np.shape(Coefs)
	#Mars mean anomaly:
	M = 19.3871 + 0.52402073 * j2000

	#angle of Fiction Mean Sun
	alpha = 270.3871 + 0.524038496*j2000

	#Perturbers
	# PBS = 0
	# for i in range(dims[0]):
	# 	PBS += Coefs[i,0]*cos(((0.985626* j2000 / Coefs[i,1]) + Coefs[i,2])*d2R)
	angles = (0.985626 * j2000 / Coefs[:,1] + Coefs[:,2]) * d2R
	PBS = np.sum(Coefs[:,0] * np.cos(angles))

	#Equation of Center
	vMinusM = ((10.691 + 3.0e-7 *j2000)*np.sin(M*d2R) + 0.623*np.sin(2*M*d2R) +
	0.050*np.sin(3*M*d2R) + 0.005*np.sin(4*M*d2R) + 0.0005*np.sin(5*M*d2R) + PBS)

	return M, alpha, PBS, vMinusM

def getMarsSolarGeometry(iTime):
	'''Get Mars time information.

	:param iTime: 6 element time list [y,m,d,h,m,s] or a datetime
	:returns: a named tuple containing the LS value as well as
	     several parameters necessary for other calculations

	'''
	
	if isinstance(iTime, datetime.datetime):
		iTime = [iTime.year, iTime.month, iTime.day, iTime.hour, iTime.minute, iTime.second]
	
	DPY = 686.9713

	# Establish a reference
	refTime = [1955,4,11,10,56,0] #Mars year 1
	y, m, d, H, M, S = refTime
	time = f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}"
	rDate = aptime.Time(time,scale='tt').jd

	# Julian Date
	y, m, d, H, M, S = iTime
	time = f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}"
	t =	aptime.Time(time, scale='utc')
	tt = t.tt


	#fixed offset since JD2000 = 2451545.0
	# Subtract an addition 0.5 because JD is defined from 12:00 UT, 
	# Using astropy gets us almost exactly 1 day off?
	j2000 = tt.jd - 2451545.0

	year = int((t.jd - rDate) / DPY) + 1 #MY starts at 1.

	M,alpha,PBS,vMinusM = getMarsParams(j2000)

	LS = (alpha + vMinusM)
	LS = LS % 360

	EOT = 2.861*np.sin(2*LS*d2R)-0.071*np.sin(4*LS*d2R)+0.002*np.sin(6*LS*d2R)-vMinusM #degrees

	MTC = (24*(((tt.jd-2451549.5)/1.027491252)+44796.0 - 0.0009626 )) % 24

	subSolarLon = ((MTC+(EOT*24/360.))*(360/24.)+180) % 360 #convert EOT to hours first
	subSolarLon = 360-subSolarLon #for some reason, this is calculated in deg W, but 
	   #we always use deg E
	   
	solarDec = (np.arcsin(0.42565*np.sin(LS*d2R))/d2R+0.25*np.sin(LS*d2R))
	sol = ((tt.jd - 2451549.5) / 1.027491252) % 1
	data = namedtuple('data','datetime ls year sol M alpha PBS vMinusM MTC EOT subSolarLon solarDec')
	d1 = data(datetime=time, ls = LS,year=year,sol=sol,M=M,alpha=alpha,PBS=PBS,vMinusM=vMinusM,MTC=MTC,EOT=EOT,
		subSolarLon=subSolarLon,solarDec=solarDec)


	return d1

def getSZAfromTime(timedata, lon, lat):
	'''Get SZA from Mars coordinates and precomputed Mars time data.
	:param timedata: output from getMarsSolarGeometry
	:param lon: the longitude in degrees (scalar or array)
	:param lat: the latitude in degrees (scalar or array)
	:returns: the solar zenith angle (same shape as input)'''

	lon = np.asarray(lon)
	lat = np.asarray(lat)
	
	delta_lon = (lon - timedata.subSolarLon + 180) % 360 - 180

	arg = (
        np.sin(timedata.solarDec*d2R)*np.sin(lat*d2R) +
        np.cos(timedata.solarDec*d2R)*np.cos(lat*d2R) *
		np.cos(delta_lon*d2R) 
	)
	
	arg = np.clip(arg, -1.0, 1.0)
	SZA = np.arccos(arg) / d2R
	
	return SZA


def getUTCfromLS(marsyear,LS):
	'''Get a UTC starting with an estimate of LS using an orbit angle approximation
	then iteratively closing in on the correct LS by incrementing the a day first and then hour.

	:param marsyear: an int mars year
	:param ls: ls- mars solar longitude
	:returns: UTC1 (python datetime)'''

	#Get LS to within this value:
	error = 0.001
	DPY = 686.9713

	###Start with estimate

	refTime = [1955,4,11,10,56,0] #Mars year 1
	y, m, d, H, M, S = refTime
	time = f"{y:04d}-{m:02d}-{d:02d}T{H:02d}:{M:02d}:{S:02d}"
	rDate = aptime.Time(time,scale='tt').jd

	#LS 0 of given mars year
	iTime = aptime.Time((rDate+(marsyear-1)*DPY),format='jd',scale='utc').to_datetime

	#Now we have a guess, iterate over the day to get closer and closer.

	thisTime = [iTime.year,iTime.month,iTime.day,iTime.hour,iTime.minute,iTime.second]
	thisLS = 0

	factor = 1 #do we increment up or down?
	iTry = 0
	dt = 60 #hours.  This will get smaller as we get closer
	counter = 0
	olddiff = 1000.
	diff = 100
	while diff > error:

		iTime = iTime+factor*datetime.timedelta(hours=dt)
		thisTime = [iTime.year,iTime.month,iTime.day,iTime.hour,iTime.minute,iTime.second]
		timedata = getMarsSolarGeometry(thisTime)
		thisLS,myear = timedata.ls, timedata.year
		if myear < marsyear and thisLS > 350:
			thisLS = thisLS - 360
			myear = marsyear 
			breakpoint()
		diff = np.abs(thisLS - LS)
		
		#Based on how far we are off our initial guess, move forward in time
		#some fraction of Mars year (360 degrees in ~600 mars days*~24 hours per day- a rough underestimate)
		dt = diff/360*(600*24)
		if diff > olddiff:
			factor = -1*factor
			counter += 1
			if counter > 1:
				dt = dt/60.

		if thisLS < LS: 
			#If we've overshot the original guess, then turn around. 
			#This can happen beacuse getMarsSolarGeometry can return a very small LS 
			#but also the previous mars year for some reason.
			factor = np.abs(factor)
			myear = marsyear
		olddiff = diff
		iTry += 1

		if iTry > 1000:
			print( 'Problem getting UTC from Ls in 2nd diff loop')
			print( 'Quitting if function getUTCfromLS...')
			breakpoint()
			exit(1)

	return iTime


def SZAGetTime(sza,date, lon, lat):
	'''Find the time on a given date and location when the SZA is a given value.

	:param sza: Solar zenith angle in degrees
	:param date: [y,m,d]<
	:param lon: the longitude in degrees
	:param lat: the latitude in degrees
	:returns: A python datetime object
	'''
	thisDate = datetime.datetime(date[0],date[1],date[2])

	count = 0
	counter = 0
	error = 1
	factor = 1
	dt = 15 #minutes
	timedata = getMarsSolarGeometry(thisDate)
	thisSza = getSZAfromTime(timedata,lon,lat)
	diff = np.abs(thisSza - sza)
	while diff > error:
		thisDate += factor*datetime.timedelta(minutes=dt)
		thisSza = getSZAfromTime(thisDate,lon,lat)
		newdiff = np.abs(thisSza - sza)
		if newdiff > diff:
			factor = -1*factor
			counter += 1
			if counter > 1:  #Wait until counter is > 1 in case we start off going the wrong way!
				dt = dt/2.

		count += 1
		if np.abs(diff - newdiff)/2. < error and counter > 5:
			print( 'this location doesnt reach the given SZA.  Returning closest value... {:f}'.format(thisSza))
			return thisDate, thisSza

		diff = newdiff

	return thisDate, thisSza


def getLTfromTime(iTime,lon):
	'''The mars local solar time from an earth time and mars longitude.

	:param iTime: 6 element list: [y,m,d,h,m,s]
	:param lon: the longitude in degrees
	:returns: The local time (float)'''

	timedata = getMarsSolarGeometry(iTime)
	LMST = timedata.MTC-lon*(24/360.)
	LTST = LMST + timedata.EOT*(24/360.)

	return LTST


def mapSZA(iTime,nlons=360,nlats=180,savefile="sza_map.png"):
	'''Create an SZA map given an Earth time

	:param iTime: 6 element list: [y,m,d,h,m,s] or datetime object
	:param nlons: number of longitude points
    :param nlats: number of latitude points
    :param savefile: output filename for the plot
	:returns: null
	'''

	# Define grid
	lats = np.linspace(-90 + 90/nlats, 90 - 90/nlats, nlats)    # centers from -89.5 to 89.5 (if nlats=180)
	lons = np.linspace(-180 + 180/nlons, 180 - 180/nlons, nlons) # centers from -179.5 to 179.5 (if nlons=360)

	timedata = getMarsSolarGeometry(iTime)

	SZA = np.zeros((nlats,nlons))

	for ilat, lat in enumerate(lats):
		for ilon, lon in enumerate(lons):
			SZA[ilat,ilon] = getSZAfromTime(timedata,lon,lat)

	min_idx = np.unravel_index(np.abs(SZA).argmin(), SZA.shape)
	subsolar_lat = lats[min_idx[0]]
	subsolar_lon = lons[min_idx[1]]
	
	pp.figure(figsize=(10, 6))
	contour = pp.contourf(lons, lats, SZA, levels=30, cmap='gist_rainbow')
	contour2 = pp.contour(lons, lats, SZA, levels=[0, 90, 180], colors='black', linewidths=1.5, linestyles='--')

	pp.xlabel('Longitude ($^o$E)')
	pp.ylabel('Latitude ($^o$N)')

	pp.clabel(contour2, fmt='%2.0f', colors='black', fontsize=11)
	cb = pp.colorbar(contour)
	cb.set_label('Solar Zenith Angle (degrees)')

	pp.savefig(savefile,dpi=150)

	print(f"Subsolar point: {subsolar_lon:.1f}°E, {subsolar_lat:.1f}°N")
	print(f"SZA map saved to {savefile}")
	print(f"Grid resolution: {nlons} x {nlats}")


if __name__ == "__main__":
	# itime = [2000,1,6,0,0,0]
	itime = [2004,1,4,13,46,31] #Mars24 examples for testing
	itime = [2000,1,23,13,3,9]
	# testSZA()
	# print(getSZAfromTime(itime,360-184.7,-14.64))
	mapSZA(itime)
	a = getMarsSolarGeometry(itime)
	print( a)
	# print( getLTfromTime(itime,38))
