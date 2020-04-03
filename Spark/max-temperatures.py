from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("MaxTemperatures")
sc = SparkContext(conf = conf)

def sparesfunc(line):
    row = line.split(',')
    station = row[0]
    entry = row[2]
    temp = float(row[3]) * 0.1 * (9.0 / 5.0) + 32.0

    return(station,entry,temp)

lines = sc.textFile("file:///SparkCourse/1800.csv")
lines_sparse = lines.map(sparesfunc)

lines_filter = lines_sparse.filter(lambda x: 'TMAX' in x[1])
lines_drop_TMAX = lines_filter.map(lambda x: (x[0],x[2]))

lines_max = lines_drop_TMAX.reduceByKey(lambda x,y: max(x,y))

results = lines_max.collect();

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))




