import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]) {

    val inputFile = "/media/danielreis/EXTERNAL_HD/dataset/pagecounts-20100806-030000"
    val outputFile = "/media/danielreis/EXTERNAL_HD/dataset/Exercise1"

    val conf = new SparkConf()
      .setAppName("ArrivalDelayOfComercialFlight").setMaster("local[*]")

    // Create a Scala Spark Context.

    val sc = new SparkContext(conf)

    // Load our input data.
    val pagecounts =  sc.textFile(inputFile)

    pagecounts.take(10).foreach(println(_))

  }
}