import explorer.FileExplorer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object test2 {
  Logger.getLogger("org").setLevel(Level.WARN)
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local")
      .appName("ArrivalDelayOfComercialFlights")
      .getOrCreate;

    val fp = new FlightProcessor(spark,"ArrDelay")

    fp.load(FileExplorer.flights_2006,0.7,0.3,102059L)

    fp.transformDataframe()
    fp.RandomForest2()
  }

}
