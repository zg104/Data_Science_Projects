from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("customer")
sc = SparkContext(conf = conf)

lines = sc.textFile("file:///SparkCourse/customer-orders.csv")

def sparsefunc(line):
    fields = line.split(',')
    customerID = int(fields[0])
    spent = float(fields[2])
    return(customerID,spent)

customer_sparse = lines.map(sparsefunc)
customer_sum = customer_sparse.reduceByKey(lambda x,y: round((x + y),2)).map(lambda x: (x[1],x[0]))
customer_sort = customer_sum.sortByKey()


# customer_filter = customer_sort.filter(lambda x : x[0]>5000)

results = customer_sort.collect();

for result in results:
    print('ID', result[1], '-->', 'Money Spent in total', result[0])

