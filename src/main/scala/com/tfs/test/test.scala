package com.tfs.test

import java.io.FileReader

import com.tfs.test.util.Json
import javax.script.{Invocable, ScriptEngineManager}

import scala.collection.mutable
import org.python.util.PythonInterpreter
import org.python.core._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source


object test{

  def main(args:Array[String]):Unit = {

    val filename = "/home/udhay/searspdqa.txt"
    var count = 0
    var vidarr = new ArrayBuffer[String]()
    var vidCor = new ArrayBuffer[String]()
    for (line <- Source.fromFile(filename).getLines) {
    //var line = "{\"vid\":\"3b487102-0e9b-4d82-ae66-43642fbb78ff\",\"active_session\":6,\"dt\":\"2019-12-19\",\"page_count\":1,\"purchase_flag\":0,\"session_time\":2115,\"session_time_last_visit\":0,\"no_of_hrs_since_last_visit\":0.0,\"no_of_hrs_since_last_proactive_accept\":0.0,\"no_of_hrs_since_last_button_accept\":13.97,\"no_of_hrs_since_last_connect\":0.0,\"no_of_hrs_since_last_interactive\":22.83,\"no_of_hrs_since_last_purchase\":0.0,\"nop_till_purchase\":0,\"nop_till_button_offer\":0,\"nop_till_proactive_offer\":0,\"nop_till_button_accept\":0,\"nop_till_proactive_accept\":0,\"browser_cat\":\"Others\",\"os_cat\":\"iOS\",\"region\":\"Moscow\",\"hour_of_day\":18,\"day_of_week\":4,\"dd\":\"M\",\"referrer\":\"https://secure3.hilton.com/ru_RU/hp/reservation/book.htm?execution=e1s3\",\"pagename\":\"null\",\"current_navigation_path\":\"/en_US/hp/search/findhotels/index.htm\",\"previous_navigation_path\":\"null\",\"exit_page_navpath_last_visit\":\"8\",\"positive_probability\":1.4943232089931858E-5}"
      var json = Json.parse(line).asMap
      //var temp1 = new mutable.HashMap[String,Any]()
      //temp1 += ("x" -> 1)
      val engine = new ScriptEngineManager().getEngineByName("nashorn")
      //val invocable: Invocable = engine.asInstanceOf[Invocable]
      json.foreach(x => engine.put(x._1, x._2))
      //val res:String = engine.eval("function main() {\n       if(x == 0)\n        { return 0;} \n       else \n        { return x+1;}\n   }\n   main();").toString
      val res = engine.eval(new FileReader("/home/udhay/searspd.js")).asInstanceOf[Double]

      //val res = invocable.invokeFunction("transformfun", "temp1")
      if(!json.get("positive_probability").get.toString.toDouble.equals(res) && !res.toString().contains("E-")) {
        println(line)
        println(json.getOrElse("positive_probability",0.111))
        println(res)
        vidarr += json.getOrElse("vid","").toString
        count += 1
      }
      else {
        vidCor += json.getOrElse("vid","").toString
      }
    }
    vidarr.distinct.foreach(println)
    println("Passed rows:")
    vidCor.distinct.take(15).foreach(println)
    println("Total mismatch:" + vidarr.distinct.length)
    println("Mismatch : "+ count)
    /*var temp1 = new mutable.HashMap[String,Any]()
    temp1 += ("x" -> true)
      var interpreter = new PythonInterpreter()
    temp1.foreach(x => interpreter.set(x._1,x._2))
    interpreter.exec("def printinfo( arg1, *vartuple ):\n   \"This prints a variable passed arguments\"\n   print \"Output is: \"\n   print arg1\n   for var in vartuple:\n      print var\n   return arg1;\n\n# Now you can call printinfo function\nresult = printinfo( x )")
    var result = interpreter.get("result")
      println(result)*/
  }
}
