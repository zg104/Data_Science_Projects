# first, import pyspark
from pyspark import SparkConf, SparkContext

# create a conf
conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc = SparkContext(conf = conf)

# parse the RDD into key value pairs in tuple which contains integers
def parseLine(line):
    fields = line.split(',')
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)

# read the file and create a new RDD after mapping the parseline function
lines = sc.textFile("file:///SparkCourse/fakefriends.csv")
rdd = lines.map(parseLine)
# mapValues does not modify the key, but the values and we use reduceByKey to sum up the values(tuples)
totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
averagesByAge = totalsByAge.mapValues(lambda x: round((x[0] / x[1])))
results = averagesByAge.sortByKey().collect() # in order to print out
for result in results:
    print(result)
