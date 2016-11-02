import math,numpy, itertools
import sys, random, copy

Positions=[[0,0,0],[10000,0,0],[20000,0,0],[0,10000,0],[0,20000,0],[10000,10000,0],[20000,10000,0],[10000,20000,0],[20000,20000,0]]
maxAreaX=20000
maxAreaY=20000

def getIDJ(beaconList):
    beaconVectorList=[]
    for b in beaconList:
        beaconVectorList.append(numpy.array(b))
    ex = (beaconVectorList[1] - beaconVectorList[0])
    ex = ex / numpy.linalg.norm(ex)
    i = numpy.dot(ex,beaconVectorList[2] - beaconVectorList[0])
    ey = beaconVectorList[2] - beaconVectorList[0] - i*ex
    #print numpy.linalg.norm(ey)
    ey = ey / numpy.linalg.norm(ey)
    ez = numpy.cross(ex,ey)
    d = numpy.linalg.norm(beaconVectorList[1] - beaconVectorList[0])
    j = numpy.dot(ey,beaconVectorList[2] - beaconVectorList[0])
    return i, d, j, ex, ey, ez

def getXYZ(tofList,i,d,j):
    x = (tofList[0]**2-tofList[1]**2+d**2)/(2*d)
    y = (tofList[0]**2-tofList[2]**2+i**2+j**2)/(2*j)-(i/j)*x
    try:
        z = abs(math.sqrt(tofList[0]**2-x**2-y**2))
        #print "z0",z
        #print "z1",math.sqrt((tofList[1])**2 - (x-d)**2-y**2)
        #print "z2",math.sqrt((tofList[2])**2 - (x-d)**2-(y-i)**2)
    except:
        z = 0
        #print z
    return x,y,z

def transformXYZ(x,y,z,ex,ey,ez,p1):
    transVect=p1+x*ex+y*ey+z*ez
    transVect[2]= abs(transVect[2])
    return transVect
def calcVectorAvg(vectorList):
    sum=numpy.array([0.0,0.0,0.0])
    x=[]
    y=[]
    z=[]
    for v in vectorList:
        sum+=v
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
    xMed=numpy.median(numpy.array(x))
    yMed=numpy.median(numpy.array(y))
    zMed=numpy.median(numpy.array(z))
    return sum / float(len(vectorList)), numpy.array([xMed,yMed,zMed])
def calcZ(tofList,beaconList,xyz):
    z=0
    divisor=0
    #print "-----"
    #print xyz
    for i in range(len(tofList)):
        try:
            #print (math.sqrt(tofList[i]**2-(xyz[0]-beaconList[i][0])**2-(xyz[1]-beaconList[i][1])**2)-beaconList[i][2])
            z+=(math.sqrt(tofList[i]**2-(xyz[0]-beaconList[i][0])**2-(xyz[1]-beaconList[i][1])**2)-beaconList[i][2])/tofList[i]
            divisor+=1/tofList[i]
        except:
            pass
    return z/divisor



def generateTOFdata(beacons):
    xmin=100
    xmax=19900
    ymin=100
    ymax=19900
    zmin=500
    zmax=3000
    i=0
    returnArray=[]
    returnXYZ=[]
    for x in range(xmin,xmax,1000):
        for y in range(ymin,ymax,700):
            for z in range(zmin,zmax,35):
                actrow=[]
                for b in beacons:
                    r=math.sqrt((x-b[0])**2+(y-b[1])**2+(z-b[2])**2)
                    r=r+(random.random()*10-5)
                    actrow.append(r)
                returnArray.append(actrow)
                returnXYZ.append(numpy.array([x,y,z]))
    return returnArray,returnXYZ

def rejectZoutliers(xyzList):
    zArray=[]
    for xyz in xyzList:
        zArray.append(xyz[2])
    zArray=numpy.array(zArray)
    mean=numpy.median(zArray)
    std=numpy.std(zArray)
    returnList=[]
    for i in range(len(xyzList)):
        if  (mean-2*std < zArray[i] < mean+2*std):
            returnList.append(xyzList[i])
    return returnList,mean,std

def dPhi(posI,posPhi):
    return math.sqrt((posPhi[0]-posI[0])**2 + (posPhi[1]-posI[1])**2 + (posPhi[2]-posI[2])**2)
def fPhi(posI,posPhi,rI):
    return dPhi(posI,posPhi)-rI
def jTfItem(posIarray,xyzEst,index,tofArray):
    sum=0
    for i in range(len(posIarray)):
        posI=posIarray[i]
        rI=tofArray[i]
        sum+=(xyzEst[index]-posI[index])*fPhi(posI,xyzEst,rI)/dPhi(posI,xyzEst)
    return sum
def jTf(positions, xyzEst, tof):
    returnArray=[jTfItem(positions, xyzEst, 0, tof), jTfItem(positions, xyzEst, 1, tof), jTfItem(positions, xyzEst, 2, tof)]
    return numpy.transpose(numpy.matrix(returnArray))
def JtJItem(posIarray,posPhi, index1, index2):
    sum=0
    for i in range(len(posIarray)):
        posI=posIarray[i]
        sum+=(posPhi[index1]-posI[index1])*(posPhi[index2]-posI[index2])/(dPhi(posI,posPhi)**2)
    return sum
def JtJ(positions,xyzEst):
    returnArray=[[0,0,0], [0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            returnArray[i][j] = JtJItem(positions,xyzEst,i,j)
    return numpy.matrix(returnArray)
def newPhi(oldPhi, positions, tofArray):
    step=numpy.transpose(numpy.linalg.inv(JtJ(positions,oldPhi))*jTf(positions, oldPhi, tofArray))
    return oldPhi - numpy.squeeze(numpy.asarray(step)),numpy.linalg.norm(step)
def NLSQ_Optimiser(xyzGuess,positions,tofArray):
    phi, difference = newPhi(xyzGuess, positions, tofArray)
    while difference >1:
        phi, difference = newPhi(phi, positions, tofArray)
        #print phi, difference
    return phi

def getNearestBeaconsIds(n,tofArray):
    copyArray= copy.deepcopy(tofArray)
    copyArray.sort()
    returnArray=[]
    for i in range(n):
        returnArray.append(tofArray.index(copyArray[i]))
    return returnArray

infile=open("tofdata.csv",'r')
outfile=open("xyz_NLSQ.csv",'w')
i=0
tofRAW=[]
for l in infile:
    i+=1
    if i==1: continue
    actline=(l.rstrip().split(','))
    newline=[]
    for a in actline:
        newline.append(float(a))
    tofRAW.append(newline)



#tofRAW=[[19710.163986498614, 22056.303642595725, 28004.953484823945, 9711.384754467748, 594.0313452372609, 13872.105550458695, 22145.71853995592, 9918.21330216865, 19907.060891599733]]
#xyzRaW=[numpy.array([100, 19700, 605])]

tofRAW, xyzRaW= generateTOFdata(Positions)
#sys.exit()


for tofIndex in range(len(tofRAW)):
    #XYZlist=[]
    indexList=getNearestBeaconsIds(3,tofRAW[tofIndex])
    bList=[]
    rList=[]
    for i in indexList:
        bList.append(Positions[i])
        rList.append(tofRAW[tofIndex][i])
    i, d, j, ex, ey, ez = getIDJ(bList)
    x, y, z = getXYZ(rList,i,d,j)
    # kann ich ev hier schon z mitteln?
    xyz = transformXYZ(x,y,z,ex,ey,ez,bList[0])
    if min(xyz)<0: continue
    if numpy.isnan(xyz[0]): continue
    if xyz[0]> maxAreaX: continue
    if xyz[1]> maxAreaY: continue
    #print xyz, xyzRaW[tofIndex]
    try:
        xyz[2] = calcZ(rList,bList,xyz)
        #XYZlist.append(xyz)
    except:
        pass
        #print "could not add", xyz
    #XYZlist, mean, std = rejectZoutliers(XYZlist)
    #if len(XYZlist)>0:
    #averag, median=calcVectorAvg(XYZlist)
    averag=xyz
    bList=Positions[:4]
    rList=tofRAW[tofIndex][:4]
    indexList=getNearestBeaconsIds(8,tofRAW[tofIndex])
    bList=[]
    rList=[]
    for i in indexList:
        bList.append(Positions[i])
        rList.append(tofRAW[tofIndex][i])
    median= NLSQ_Optimiser(averag,bList,rList)
    if abs(averag[2]-xyzRaW[tofIndex][2])>50    :
        print "==============================="
        print averag[2],xyzRaW[tofIndex][2]
        print averag

        print "================================"
        print xyzRaW[tofIndex]
        print tofRAW[tofIndex]
    outfile.write( ",".join(map(str,averag.tolist()))+","+  ",".join(map(str,median.tolist()))+","+",".join(map(str,xyzRaW[tofIndex].tolist()))+"\n")
    #print calcVectorAvg(XYZlist).tolist(),xyzRaW[tofIndex].tolist()
infile.close()
outfile.close()
