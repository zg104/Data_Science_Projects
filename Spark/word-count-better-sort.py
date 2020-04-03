import re
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

input = sc.textFile("file:///sparkcourse/book.txt")
words = input.flatMap(normalizeWords)

# sort by counts
wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
# here, we create a key value pair in order to use reduceByKey so that
# we can aggregate the values for the same key (count)


# Simplest way to sort the output by values.
# After the reduceByKey you can swap the output like key as value and value as key and then you can appply sortByKey method where false sorts in the Descending order.
# By default it will sort in the ascending order.

wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
# wordCountsSorted = wordCounts.map(lambda (x,y): (y,x)).sortByKey()
results = wordCountsSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count)