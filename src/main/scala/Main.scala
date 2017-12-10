import explorer.FileExplorer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object Main {
  Logger.getLogger("org").setLevel(Level.WARN)
  def main(args: Array[String]): Unit = {

    println("Creating Spark Application and Context 1-7")
    val spark = SparkSession.builder
      .master("local")
      .appName("ArrivalDelayOfComercialFlights")
      .getOrCreate;

    println("Creating main class of application 2-7")
    val fp = new FlightProcessor(spark,"ArrDelay")

    println("Loading Model 3-7")
    fp.load(FileExplorer.flights_2008,0.7,0.3,5149)

    println("Transforming Data 4-7")
    fp.transformDataframe()

    println("Applying RandomForest 5-7")
    fp.RandomForest()

    println("Evaluating Model 6-7")
    fp.computeTestError()

    println("DONE 7-7")
    //Just doing this to make more plots of data possible
    var not_done = true
    while(not_done){
      println("Do you wish to plot the predictions with the real values?")
      println("If YES, input the total number record you want to plot like p.e: 50")
      println("If NO, just enter -1")
      val r = scala.io.StdIn.readLine()
      if(r.equals("-1"))
        not_done=false
      else{
        fp.plotPredictions(r)
      }
    }
  }

}
