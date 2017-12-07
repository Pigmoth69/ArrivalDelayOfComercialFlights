import org.apache.spark.sql.SparkSession

object teste3 {
  import org.apache.spark.mllib.linalg.Vectors
  import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local")
      .appName("teste3")
      .getOrCreate;

    val observations = spark.sparkContext.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )

    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)  // a dense vector containing the mean value for each column
    //println(summary.variance)  // column-wise variance
    //println(summary.numNonzeros)  // number of nonzeros in each column
  }
}
