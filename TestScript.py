# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:11:42 2019

@author: guser
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


from rdflib import  Graph, URIRef, Literal, Namespace, RDF


import time
import requests
import subprocess
import shutil
import os
import signal
import logging as lg

from io import StringIO

import cProfile
import pstats

from sys import platform
#from datetime import datetime

#%%Define parameters:

#Database parameters
user='postgres'
passw='docker'
host='localhost'
port='5432'
db='Sensors'

startTime = '2019-08-01 00:00'
resolution = 15 #min
#nbSensors = [10,1000]
#datapointPerSensor = [96,672,2880]#, 17568]#day,week,month,1/2year
nbSensors = [10]
datapointPerSensor = [100000]

#nbSensors = [10,20]
#datapointPerSensor = [100,200]

execOverall=5 # repeat everything that times
exePerQuery=4 #iteration per query

#%%Configure Logger
logger=lg.getLogger('myLogger')
logger.propagate = False
logger.setLevel(lg.INFO)

if not logger.handlers:
    # Here I created handler, formatter, loglevel
    ch = lg.StreamHandler()
    ch.setLevel(lg.INFO)
    formatter = lg.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fc = lg.FileHandler("logfile.log")
    fc.setLevel(lg.DEBUG)
    fc.setFormatter(formatter)
    logger.addHandler(fc)

#%% Define Functions

def cleanColumns(columns):
    cols = []
    for col in columns:
        col = col.replace(' ', '_')
        cols.append(col)
    return cols

def to_pg(df, table_name, con):
    data = StringIO()
    df.columns = cleanColumns(df.columns)
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    try:
        curs.execute("DROP TABLE " + table_name)
    except:
        logger.warning("Cannot drop table in database")
    empty_table = pd.io.sql.get_schema(df, table_name, con = con)
    empty_table = empty_table.replace('"', '')
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep = ',')
    curs.connection.commit()

#Create Table with data
def createRandomData(sensorCount,datapointCount):
    logger.info("Create random data...")
    #Timestamp column
    timestamps=pd.date_range(pd.to_datetime(startTime), periods=datapointCount, freq=str(resolution)+'min')
    timestamps=np.repeat(timestamps.values,sensorCount,axis=0)
    
    #Sensor ID column
    sensorIds=["S"+str(x) for x in range(sensorCount)]
    sensorIdColumns=sensorIds*datapointCount
    
    #sensor data column - float values between 0 and 100
    np.random.seed(0)
    randomData=np.random.rand(datapointCount*sensorCount,1)*100
    
    index=np.arange(sensorCount*datapointCount)
        
    data={'index':index,'sensorID':sensorIdColumns,'time':timestamps, 'value':randomData.flatten()}
    dataTable=pd.DataFrame(data)

    logger.info("...done!")
    
    return sensorIds, dataTable

#write to psotgresdatabase 
def writeDataToDB(dataTable):
    logger.info("Write data to DB...")
    engine=create_engine('postgresql://'+user+':'+passw+'@'+host+':'+port+'/'+db)
    
    if not database_exists(engine.url):
        create_database(engine.url)
#    
#
#    
#    dataTable.to_sql('SensorData',engine,if_exists='replace')
#    
#    #index columns
#    sqlStatement="""CREATE INDEX "ix_time_index"
#    ON public."SensorData" ("time");"""    
#    engine.execute(sqlStatement)
#    
#    sqlStatement="""CREATE INDEX "ix_sensorID_index"
#    ON public."SensorData" ("sensorID");"""    
#    engine.execute(sqlStatement)
    con=engine.connect()
    to_pg(dataTable, 'SensorData', engine)
 
    
    con.close()
    logger.info("...done!")


#create a RDF File with all observations (timestamp and data)
def createSensorObservationOntology(sensorIds,dataTable,filename):
    logger.info("Create ontology with observations...")
    
    folder='createdOntologies/'
    outputFileName=folder+filename
      
    if not os.path.isfile(outputFileName):          
        sosaNsString='http://www.w3.org/ns/sosa/'
        sosa = Namespace(sosaNsString)
        
        #g=Graph(store='Sleepycat')
        #g.open('GeneratedTripes',create=True)
        g=Graph()
        
        #add every sensor
        for s in sensorIds:    
            g.add((URIRef(sosaNsString + s), RDF.type, sosa.Sensor)) 
        
        #add observation for every row in table
        #for index, row in dataTable.iterrows():
        for row in dataTable.itertuples(index=True, ):

            #Add observation
            g.add((URIRef(sosaNsString + "Observation_" + str(row.index)), RDF.type, sosa.Observation))
            
            # add time and result to observation
            g.add((URIRef(sosaNsString + "Observation_" + str(row.index)), sosa.resultTime, Literal(row.time)))
            g.add((URIRef(sosaNsString + "Observation_" + str(row.index)), sosa.hasSimpleResult, Literal(row.value)))
            
            # observation madeBySensor sensor
            g.add((URIRef(sosaNsString + "Observation_" + str(row.index)), sosa.madeBySensor, URIRef(sosaNsString + row.sensorID)))
            
            # sensor made Observation observation
            g.add((URIRef(sosaNsString + row.sensorID), sosa.madeObservation ,sosaNsString + URIRef("Observation_" + str(row.index))))
            
        g.serialize(destination=outputFileName, format='turtle')
        g.close()

        logger.info("..done!")
    else:
        logger.info("ontology already exists!")
        
    return outputFileName#

#create a RDF File with all observations (timestamp and data)
def createSensorOntology(sensorIds,filename):
    logger.info("Creating ontology file...")
    folder='createdOntologies/'
    outputFileName=folder + filename
       
    if not os.path.isfile(outputFileName):
        sosaNsString='http://www.w3.org/ns/sosa/'
        sosa = Namespace(sosaNsString)
        
        #g=Graph(store='Sleepycat')    #g.open('GeneratedTripes',create=True)
        g=Graph()
        
        #add every sensor
        for s in sensorIds:    
            g.add((URIRef(sosaNsString + s), RDF.type, sosa.Sensor)) 
                
        g.serialize(destination=outputFileName, format='turtle')
        g.close()
        logger.info("...done!")
    else:
        logger.info("Ontology already exists!")

    return outputFileName

## creates a mapping file wich is used for OBDA in the custom property Function
def createMappingFileForDataAccess(sensorIds, filename):
    logger.info("Creating Mapping File...")
    
    folder='createdMappings/'
    outputFileName=folder+filename
    
    if not os.path.isfile(outputFileName): 
        sosaNsString='http://www.w3.org/ns/sosa/'
        #sosa = Namespace(sosaNsString)
        mappingString ='http://sic.auto.tuwien.ac.at/mappings#'
        mapping =Namespace(mappingString)
        
        #g=Graph(store='Sleepycat')
        #g.open('GeneratedTripes',create=True)
        g=Graph()
        
        g.add((URIRef(mappingString+"Connection1"), mapping.url, Literal("localhost:5432")))
        g.add((URIRef(mappingString+"Connection1"), mapping.db, Literal("Sensors")))
        g.add((URIRef(mappingString+"Connection1"), mapping.user, Literal("postgres")))     
        g.add((URIRef(mappingString+"Connection1"), mapping.passw, Literal("docker")))
              
        for s in sensorIds:    
            sqlString="Select \"time\", \"value\" FROM SensorData WHERE (sensorID = '"+ s +"') ORDER BY \"time\" DESC "
            g.add((URIRef(sosaNsString + s), mapping.hasMapping, URIRef(mappingString + "Mapping"+s)))
            g.add((URIRef(mappingString + "Mapping"+s), mapping.hasDBConnection, URIRef(mappingString+"Connection1")))
            g.add((URIRef(mappingString + "Mapping"+s), mapping.hasSQLString, Literal(sqlString)))
        
        g.serialize(destination=outputFileName, format='turtle')
        g.close()
        logger.info("...done!")
    else:
        logger.info("Mapping file already exists!")

    return outputFileName

##test the queries and write results to file
def execTestQueries(endpoint, sparqlQueryList,dsSize,simInfo):
    import random
    import pandas as pd
    
    filePath="./SPARQL Queries/"
    
    execTimes=simInfo["exePerQuery"]
    testCaseString=simInfo["testCaseString"]
    sensorCount=simInfo["sensorCount"]
    datapointCount=simInfo["datapointCount"]
  
    
    
    result=pd.DataFrame(index=np.arange(0, len(sparqlQueryList)*execTimes),columns=['timestamp','testCase','nbOfsensors','datapointPerSensor','datasetSize','query','resultCount','resultByte', 'execOrder', 'execIteration','execTime'])
    
    queries=sparqlQueryList
    random.shuffle(queries)
    
    #logTime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    overallExecOrder=0
    execOrder=0
    for q in queries:
        execOrder=execOrder+1
        logTime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        currentFilePath=filePath+q+".txt"
        f= open(currentFilePath,"r")
        testQuery= f.read()
        f.close()
        
        for i in range(execTimes):
            start=time.time()
            response = requests.post(endpoint, data={'query': testQuery})
            end=time.time()
            
            dtime=end-start
            #print(q +': '+ str(dtime))
            data = response.json()
            dataCount= len(data['results']['bindings'])
                   
            result.loc[overallExecOrder]=[logTime,testCaseString,sensorCount,datapointCount,dsSize,q,str(dataCount),str(len(response.content)),execOrder,i+1,dtime]
            overallExecOrder=overallExecOrder+1
        
    return result

##execTestQueries on Ontop enpoint
def execTestQueriesOntop(endpoint, sparqlQueryList, dsSize, simInfo):
    import random
    import pandas as pd
    
    filePath="./SPARQL Queries/"
    
    execTimes=simInfo["exePerQuery"]
    testCaseString=simInfo["testCaseString"]
    sensorCount=simInfo["sensorCount"]
    datapointCount=simInfo["datapointCount"]
  
    
    
    result=pd.DataFrame(index=np.arange(0, len(sparqlQueryList)*execTimes),columns=['timestamp','testCase','nbOfsensors','datapointPerSensor','datasetSize','query','resultCount','resultByte', 'execOrder','execIteration','execTime'])
    
    queries=sparqlQueryList
    random.shuffle(queries)
    
    #logTime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    overallExecOrder=0
    execOrder=0
    for q in queries:
        execOrder=execOrder+1
        logTime=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        currentFilePath=filePath+q+".txt"
        f= open(currentFilePath,"r")
        testQuery= f.read()
        f.close()
        
        params={'action':'exec','queryLn':'SPARQL','query': testQuery}
        headers = { 'content-type': 'application/x-www-form-urlencoded', 'accept': 'application/sparql-results+json'}
        for i in range(execTimes):
            start=time.time()

            response = requests.get(endpoint, headers=headers, params=params)
            end=time.time()
            
            dtime=end-start
            #print(q +': '+ str(dtime))
            data = response.json()
            dataCount= len(data['results']['bindings'])
                   
            result.loc[overallExecOrder]=[logTime,testCaseString,sensorCount,datapointCount,dsSize,q,str(dataCount),str(len(response.content)),execOrder,i+1,dtime]
            
            overallExecOrder=overallExecOrder+1
        
    return result


#load data file into fuseki server
def deleteDataset(endpoint,datasetName):
    r = requests.delete(endpoint+'/$/datasets/'+datasetName)
    print("Response form Server deleting dataset: ")
    print(r) 
    
def loadintoDataset(endpoint, datasetName, dataFileName):
    headers = {'Content-Type': 'text/turtle;charset=utf-8'}
    data = open(dataFileName).read()
    r = requests.post(endpoint+'/'+datasetName+'/data', data=data, headers=headers,)
    print("Response form Server adding data: ")
    print(r)
    
def loadNewDataset(endpoint,datasetName,dataFileName):
    
    try:
        headers = {'Content-Type': 'text/turtle;charset=utf-8'}
        payload = {'dbType': 'tdb', 'dbName': datasetName}
        r = requests.post(endpoint+'/$/datasets', headers=headers, params=payload)
        print("Response form Server creating dataset: ")
        print(r)
    
        loadintoDataset(endpoint,datasetName,dataFileName)
    except:
        print("Cannot load data to server")
        
    

    
def getDatasetSize(start_path):
     #filename= "./apache-jena-fuseki-3.12.0/run/databases/ds"
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def runTest( ontologyFile, queries,simInfo):#testCaseString, sensorCount, datapointCount,
    #) start fuseki
    if platform =="linux":
        cmd="/home/guser/customPropertyODBARepo/apache-jena-fuseki-3.12.0/fuseki-server"
        wd="/home/guser/customPropertyODBARepo/apache-jena-fuseki-3.12.0/"
        p=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, cwd=wd, preexec_fn=os.setsid)
    else: 
        cmd="start C:/Users/guser/Documents/CustomPropertyOBDA/apache-jena-fuseki-3.12.0/fuseki-server-startWithOBDA.bat"
        wd="C:/Users/guser/Documents/CustomPropertyOBDA/apache-jena-fuseki-3.12.0"
        p=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE, cwd=wd)
    #wait till startup
    time.sleep(7)
    
    #load File into Fuseki server
    loadNewDataset('http://localhost:3030','ds','sosa.ttl')
    
    loadintoDataset('http://localhost:3030','ds',ontologyFile)
    time.sleep(1)
    #get dataset size
    dsSize=getDatasetSize('./apache-jena-fuseki-3.12.0/run/databases/ds')
    
    #run sparql queries and meassure time and size & store to csv
    print("dataset size: "+str(dsSize))

    resultFrame=execTestQueries('http://127.0.0.1:3030/ds', queries, dsSize,simInfo)
    
    
    # if file does not exist write header 
    if not os.path.isfile('result.csv'):
       resultFrame.to_csv('result.csv', header='column_names')
    else: # else it exists so append without writing the header
       resultFrame.to_csv('result.csv', mode='a', header=False)
    
    #delete dataset
    #time.sleep(2)
    deleteDataset('http://localhost:3030','ds')
    
    #close fuseki
    if platform == "linux":
        os.kill(os.getpgid(p.pid),signal.SIGTERM)
    else:
        subprocess.call(['taskkill', '/F', '/T', '/IM', 'cmd.exe'])
    
    
    #delete dataset from disk
    time.sleep(1)
    if platform == "linux":
        #shutil.rmtree('/home/guser/customPropertyODBARepo/apache-jena-fuseki-3.12.0/run/databases/ds')
        pass
    else:
        try:
            shutil.rmtree('C:/Users/guser/Documents/CustomPropertyOBDA/apache-jena-fuseki-3.12.0/run/databases/ds')   
        except:
            logger.warning("Could not delete dataset 'ds'!")


# In[6]:

#1)start postgres database (docker)
    #docker run --rm --name pg-docker -e POSTGRES_PASSWORD=docker -d -p 5432:5432 -v C:\Users\guser\Documents\dockerVolumes\postgres\:\var\lib\postgresql\data postgres
#2)start ontop service with mapping file 
    #cd C:\Users\guser\Documents\OntopSPARQL_Server\jetty-distribution-9.4.6.v20170531\ontop-base
    #java -jar ../start.jar 


logger.info("Start new test run ... ")

maxRuns=len(nbSensors)*len(datapointPerSensor)
z=1; 
for i in nbSensors:
    sensorCount=i
    for j in datapointPerSensor:
        datapointCount=j

        logger.info(f'--Test # {z} of {maxRuns}--------------------------------')
        #print('Test number '+ str(z)+' of ' + str(len(nbSensors)*len(datapointPerSensor)))
        z=z+1 
        logger.info("--Preperation--")
        #create test data and write to database       
        sensorIds, dataTable=createRandomData(sensorCount,datapointCount)         
        writeDataToDB(dataTable)         
        
        for k in range(execOverall):
            logger.info(f'-- Run {k+1} of {execOverall}-----------------------')
            #%% test ontology storage - base case
            logger.info("--Base Case--")
            simInfo={"testCaseString": "base", "sensorCount":sensorCount, "datapointCount":datapointCount,"exePerQuery":exePerQuery}
            
            #create ontology file with all observations       
            ontologyFile=createSensorObservationOntology(sensorIds, dataTable,'ontology_Res'+str(resolution)+'_Sens'+ str(sensorCount)+'_DP'+str(datapointCount)+'_withObservations'+'.ttl')
            
           # queries=["Q1","Q2","Q3","Q4","Q5","Q6","Q7"]
            queries=["Q7a"]

            runTest(ontologyFile,queries,simInfo)
                    
            logger.info("...finished Base Case--")
           
            
            #%% Test for OBDA
            logger.info("--OBDA--")
            simInfo={"testCaseString": "obda", "sensorCount":sensorCount, "datapointCount":datapointCount,"exePerQuery":exePerQuery}
     
            #create mapping file            
            mappingFile=createMappingFileForDataAccess(sensorIds,'Mapping_Res'+str(resolution)+'_Sens'+ str(sensorCount)+'_DP'+str(datapointCount)+'.ttl')
     
            #copy mapping file to fuseki
            shutil.copyfile(mappingFile, './apache-jena-fuseki-3.12.0/Mapping.ttl')
            
            #create ontology file           
            ontologyFile=createSensorOntology(sensorIds, 'ontology_Res'+str(resolution)+'_Sens'+ str(sensorCount)+'_DP'+str(datapointCount)+'.ttl')
    
            
            #queries=["Q1_obda","Q2_obda","Q3_obda","Q4_obda","Q5_obda","Q6_obda","Q7_obda"]
            queries=["Q7a_obda"]
            runTest( ontologyFile,queries,simInfo)
            
            logger.info("...finished OBDA--")
    
    
            #%% test ontop
            logger.info("--ONTOP--")
            simInfo={"testCaseString": "ontop", "sensorCount":sensorCount, "datapointCount":datapointCount,"exePerQuery":exePerQuery}
            #start Ontop with accurate mapping file - keep it open!
            
            #run sparql queries and meassure time size & #store results to csv file
            #queries=["Q1","Q2","Q3","Q4","Q5","Q6","Q7"]
            queries=["Q7a"]
            resultFrame=execTestQueriesOntop('http://127.0.0.1:8080/rdf4j-workbench/repositories/05082019/query', queries, 0, simInfo)
            # if file does not exist write header 
            if not os.path.isfile('result.csv'):
               resultFrame.to_csv('result.csv', header='column_names')
            else: # else it exists so append without writing the header
               resultFrame.to_csv('result.csv', mode='a', header=False)
            logger.info("...finished ONTOP--")


            logger.info("------------------------------------------------------")

        logger.info("------------------------------------------------------")
    logger.info("------------------------------------------------------")
#%% 
results = pd.read_csv('result.csv')
dateObj=datetime.now()
shutil.copyfile('result.csv', 'resultFiles/result_'+str(dateObj.year)+'-'+str(dateObj.month)+'-'+str(dateObj.day)+'__'+str(dateObj.hour)+'_'+str(dateObj.minute)+'.csv')
#results = pd.read_csv('result_1020_50_100__100_200_500_1000_10000.csv')
#%% Test result sets
baseResults=results[results.testCase=='base']
obdaResults=results[results.testCase=='obda']
ontopResults=results[results.testCase=='ontop']

#Check if one is empty
def findEmptyResultSets(df):
    grp=df.groupby(['query','nbOfsensors','datapointPerSensor']).groups
    grpList=list(grp)
    dfGrp=df.groupby(['query','nbOfsensors','datapointPerSensor']).sum()['resultCount']
    
    errorFlag=0
    for i in range(dfGrp.size):
        if (dfGrp.values[i] == 0):
            print("Empty result  in " + str(grpList[i]))
            errorFlag=1
    
    if(errorFlag ==0):
        print("No empty result sets found!")


print()
print("---------------chek result sets--------------------") 
findEmptyResultSets(baseResults)
findEmptyResultSets(obdaResults)
findEmptyResultSets(ontopResults)

#%%
#check if base and obda is euqal
b=baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).sum()['resultCount']
o=obdaResults.groupby(['query','nbOfsensors','datapointPerSensor']).sum()['resultCount']
on=ontopResults.groupby(['query','nbOfsensors','datapointPerSensor']).sum()['resultCount']
grp=baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).groups
grpList=list(grp)

errorFlag=0
for i in range(b.size):
    if (b.values[i] != o.values[i] or b.values[i] != on.values[i] or o.values[i] != on.values[i]):
        print("Difference in " + str(grpList[i]) + " - base: "+str(b.values[i])+"\tobda: "+ str(o.values[i]) + "\tontop: "+ str(on.values[i]))
        errorFlag=1

if(errorFlag == 0):
    print("No differences in result sets!")


#%%

#b=baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).mean()['execTime']
#o=obdaResults.groupby(['query','nbOfsensors','datapointPerSensor']).mean()['execTime']
#grpNameList=list(baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).groups)
#
#d={'base':b.values,'obda':o.values}
#df=pd.DataFrame(d, index=grpNameList)
#df.plot(kind='bar',title="MEAN exec time")

#
#b=baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).max()['execTime']
#o=obdaResults.groupby(['query','nbOfsensors','datapointPerSensor']).max()['execTime']
#on=ontopResults.groupby(['query','nbOfsensors','datapointPerSensor']).max()['execTime']
#grpNameList=list(baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).groups)
#
#d={'base':b.values,'obda':o.values, 'ontop':on.values}
#df=pd.DataFrame(d, index=grpNameList)
#
#df.plot(kind='bar',title="MAX exec time")
#rcParams.update({'figure.autolayout':True})

##plot data size
#b=baseResults.groupby(['query','nbOfsensors','datapointPerSensor']).mean()['datasetSize']
#o=obdaResults.groupby(['query','nbOfsensors','datapointPerSensor']).mean()['datasetSize']
#
#plt.figure()
#b.plot(kind='bar')
#plt.title('Base - mean datasize')
#
#plt.figure()
#o.plot(kind='bar')
#plt.title('obda - mean datasize')
#rcParams.update({'figure.autolayout':True})

#plot result size


##%% Auswertung Profiling
#p = pstats.Stats('cprofile_results')
##sort by standard name
#p.strip_dirs().sort_stats(-1).print_stats(10)
##sort by function name
#p.sort_stats('name').print_stats(10)
##sort by cumulative time in a function
#p.sort_stats('cumulative').print_stats(10)
##sort by time spent in a function
#p.sort_stats('time').print_stats(10)
