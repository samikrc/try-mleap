# try-mleap
Project to test out Spark code with ScalaTest package, and saving and deploying Spark ML pipelines with [mleap](https://github.com/combust/mleap).

## Building Instructions

### Prerequisites
1. JDK, version 1.8.0_144 or later
2. maven, version 3.3.9 or later
3. scala, version 2.11.8

### Building
1. `git clone git@github.com:samikrc/try-mleap.git`
2. `mvn clean scala:compile package -DskipTests`

## Running Tests through ScalaTest cli
1. `cd docker` (The compiled jars are copied to this folder, where we already have the scalatest jars)

2. Running individual tests: 
    `scala -J-Xmx2g -cp "scalatest_2.11-3.0.5.jar:scalactic_2.11-3.0.5.jar:try-mleap-1.0-SNAPSHOT.jar" org.scalatest.tools.Runner -o -R try-mleap-1.0-SNAPSHOT-tests.jar -s com.tfs.test.CommonModelsSpec` 
        
3. Running all tests together:
    `scala -J-Xmx2g -cp "scalatest_2.11-3.0.5.jar:scalactic_2.11-3.0.5.jar:try-scalatest-1.0-SNAPSHOT.jar" org.scalatest.tools.Runner -o -R try-mleap-1.0-SNAPSHOT-tests.jar`
    
