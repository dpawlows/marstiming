'''Mars timing information based on MARS24: http://www.giss.nasa.gov/tools/mars24/help/algorithm.html

Contains several functions for calculating Mars time parameters from Earth time and vice versa.
Probably the most useful functions are:
getMTfromTime: gets mars time data given a 6 element time list
getSZAfromTime: gets the SZA from a 6 element time list and coordinates
getLTfromTime: gets the LTST from a 6 element time list and longitude
getUTCfromLS: Estimates the Earth time from LS and a Mars year
'''

import datetime
from numpy import pi, floor,array,shape, cos, sin,ceil,arcsin,arccos,arange
from collections import namedtuple
from matplotlib import pyplot


d2R = pi/180.

def getJD(iTime):
	'''getJD(iTime): Get the Julian date in seconds'''

	offset = 2440587.5 #JD on 1/1/1970 00:00:00 

	year = iTime[0]
	month = iTime[1]
	day = iTime[2]
	hour = iTime[3]
	minute = iTime[4]
	sec = iTime[5]
	date = datetime.datetime(year,month,day,hour,minute,sec)

	iTime = [1970,1,1,0,0,0]
	year = iTime[0]
	month = iTime[1]
	day = iTime[2]
	hour = iTime[3]
	minute = iTime[4]
	sec = iTime[5]
	ref = datetime.datetime(year,month,day,hour,minute,sec)
	deltaTime = (date-ref) 
	return deltaTime.total_seconds()/86400. + offset



def getUTC(jd):
	'''getUTC(jd): Get UTC given jd'''

	offset = 2440587.5 #JD on 1/1/1970 00:00:00 

	iTime = [1970,1,1,0,0,0] 
	year = iTime[0]
	month = iTime[1]
	day = iTime[2]
	hour = iTime[3]
	minute = iTime[4]
	sec = iTime[5]

	d1970 = datetime.datetime(year,month,day,hour,minute,sec)
	return d1970 + datetime.timedelta(seconds=((jd-offset)*86400.))


def getJ2000(iTime):
	'''getJ2000(iTime): get offset from J2000 epoch.'''
	jd = getJD(iTime)
	T = (jd - 2451545.0)/36525 if iTime[0] < 1972 else 0

	conversion = 64.184 + 59* T - 51.2* T**2 - 67.1* T**3 - 16.4* T**4

	#convert to Terrestrial Time
	jdTT = jd+(conversion/86400) 

	return jdTT - 2451545.0


def testJ2000():
	iTime = [2001,11,13,2,45,2]
	testJD = getJ2000(iTime)
	callibration = 58891502.000000 #test should be this value.
	diff = testJD - callibration
	print testJD
	print 'difference = {0}s'.format(diff)

def testUTC():
	jd = 2452226.614606
	date = getUTC(jd)

	callibration = datetime.datetime(2001,11,13,2,45,2)
	diff = callibration - date
	print date
	print 'difference = {}'.format(diff)


def testLS():
	
	iTime = [2000,1,6,0,0,0]
	lsdata = getMTfromTime(iTime)
	ls = lsdata.ls
	year = lsdata.year
	callibration = 277.18677
	diff = ls - callibration
	print ls
	print 'Difference = {:f} degrees'.format(diff)

def getMarsParams(j2000):
	'''getMarsParams(j2000): Mars time parameters'''
	
	
	Coefs = array(
	[[0.0071,2.2353,49.409],
	[0.0057,2.7543,168.173],
	[0.0039,1.1177,191.837],
	[0.0037,15.7866,21.736],
	[0.0021,2.1354,15.704],
	[0.0020,2.4694,95.528],
	[0.0018,32.8493,49.095]])

	dims = shape(Coefs)
	#Mars mean anomaly:
	M = 19.3870 + 0.52402075 * j2000

	#angle of Fiction Mean Sun
	alpha = 270.3863 + 0.52403840*j2000

	#Perturbers
	PBS = 0
	for i in range(dims[0]):
		PBS += Coefs[i,0]*cos(((0.985626* j2000 / Coefs[i,1]) + Coefs[i,2])*d2R)

	#Equation of Center
	vMinusM = ((10.691 + 3.0e-7 *j2000)*sin(M*d2R) + 0.623*sin(2*M*d2R) + 
	0.050*sin(3*M*d2R) + 0.005*sin(4*M*d2R) + 0.0005*sin(5*M*d2R) + PBS)

	return M, alpha, PBS, vMinusM

def getMTfromTime(iTime):
	'''getMTfromTime(iTime): Get Mars time information given an iTime: [y,m,d,h,m,s].
	
	Returns a named tuple containing the LS value as well as 
	several parameters necessary for other calculations'''
	
	DPY = 686.9713
	refTime = [1955,4,11,10,56,0] #Mars year 1
	rDate = getJD(refTime)
	thisTime = getJD(iTime)
	year = floor((thisTime - rDate)/DPY)+1

	j2000 = getJ2000(iTime)
	M,alpha,PBS,vMinusM = getMarsParams(j2000)

	LS = (alpha + vMinusM)

	while LS > 360:
		LS -= 360

	if LS < 0:
		LS = 360. + 360.*(LS/360. - ceil(LS/360.0))

	EOT = 2.861*sin(2*LS*d2R)-0.071*sin(4*LS*d2R)+0.002*sin(6*LS*d2R)-vMinusM

	MTC = (24*(((j2000 - 4.5)/1.027491252)+44796.0 - 0.00096 )) % 24
	subSolarLon = ((MTC+EOT*24/360.)*(360/24.)+180) % 360
	solarDec = (arcsin(0.42565*sin(LS*d2R))/d2R+0.25*sin(LS*d2R))

	data = namedtuple('data','ls year M alpha PBS vMinusM MTC EOT subSolarLon solarDec')
	d1 = data(ls = LS,year=year,M=M,alpha=alpha,PBS=PBS,vMinusM=vMinusM,MTC=MTC,EOT=EOT,
		subSolarLon=subSolarLon,solarDec=solarDec)

	return d1

def getUTCfromLS(marsyear,LS):
	'''getUTCfromLS(marsyear,LS): Get a UTC from a given mars year and LS.
	
	This starts with an estimate of LS using an orbit angle approximation
	then iteratively closes in on the correct LS by incrementing the 
	day and then hour.'''

	#Get LS to within this value:
	error = 0.001
	DPY = 686.9713

	###Start with estimate

	refTime = [1955,4,11,10,56,0] #Mars year 1
	rDate = getJD(refTime)
	iTime = getUTC(rDate+(marsyear-1)*DPY) #LS 0 of given mars year

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
		timedata = getMTfromTime(thisTime)
		thisLS,myear = timedata.ls, timedata.year
		diff = abs(thisLS - LS)


		if diff > olddiff:
			factor = -1*factor 
			counter += 1
			if counter > 1:
				dt = dt/60.

		olddiff = diff
		iTry += 1


		if iTry > 1000:
			print 'Problem getting UTC from Ls in 2nd diff loop'
			print 'Quitting if function getUTCfromLS...'
			exit(1)

	return iTime

def getSZAfromTime(iTime,lon,lat):
	'''getSZAfromTime(iTime,lon,lat): Get SZA from Earth time and Mars coordinates.
	inputs: iTime: 6 element list: [y,m,d,h,m,s]
	        lon: the longitude in degrees
	        lat: the latitude in degrees'''
	timedata = getMTfromTime(iTime)
	SZA = arccos(sin(timedata.solarDec*d2R)*sin(lat*d2R)+
		cos(timedata.solarDec*d2R)*cos(lat*d2R)*cos((lon-timedata.subSolarLon)*d2R))/d2R

	return SZA

def testSZA():
	'''test getSZAfromTime'''
	itime = [2000,1,6,0,0,0]
	lon = 0.0
	lat = 0.0
	expected = 154.26182
	sza = getSZAfromTime(itime,lon,lat)
	print sza
	print 'Difference = {:f} degrees'.format(sza-expected)


def getLTfromTime(iTime,lon):
	'''getLTfromTime(iTime,lon): The mars local solar time from an earth time and mars longitude.
	
	inputs: iTime: 6 element list: [y,m,d,h,m,s]
	lon: the longitude in degrees'''

	timedata = getMTfromTime(iTime)
	LMST = timedata.MTC-lon*(24/360.)
	LTST = LMST + timedata.EOT*(24/360.)

	return LTST

def testLTfromTime():
	'''test getLTfromTime function'''
	iTime = [2000,1,6,0,0,0]
	lon = 0.0
	LTST = getLTfromTime(iTime,lon)
	expected = 23.64847
	print LTST
	print 'Difference = {:f} degrees'.format(LTST-expected)

def mapSZA(iTime):
	'''mapSZA(iTime): Create an SZA map given an Earth time
	inputs: iTime: 6 element list: [y,m,d,h,m,s]'''
	import numpy as np
	
	nlons = 72
	nlats = 72
	latitude = arange(nlats-1)*2.5-87.5
	longitude = arange(nlons-1)*5-175.

	SZA = np.zeros((nlats-1,nlons-1))
	for ilat in arange(nlats-1):
		for ilon in arange(nlons-1):
			SZA[ilat,ilon] = getSZAfromTime(iTime,longitude[ilon],latitude[ilat])

	
	pyplot.figure()
	pyplot.xlabel('Longitude')
	pyplot.ylabel('Latitude')
	cont = pyplot.contourf(longitude,latitude,SZA,30,cmap='gist_rainbow')
	
	cb = pyplot.colorbar(cont)
	cb.set_label('Solar Zenith Angle')

	pyplot.savefig('plot.ps')





