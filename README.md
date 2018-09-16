# try-scalatest
Project to test out saving and deploying ML pipelines with mleap.

## Building Instructions

### Prerequisites
1. JDK, version 1.8.0_144 or later
2. maven, version 3.3.9 or later
3. scala, version 2.11.8

### Building
1. `git clone https://github.com/samikrc/try-scalatest.git`
2. `mvn clean scala:compile package`

## Running Tests through ScalaTest cli
1. `cd docker` (The compiled jars are copied to this folder, where we already have the scalatest jars)
2. Running individual tests: 
    `scala -J-Xmx2g -cp "scalatest_2.11-3.0.5.jar:scalactic_2.11-3.0.5.jar:try-scalatest-1.0-SNAPSHOT.jar" org.scalatest.tools.Runner -o -R try-scalatest-1.0-SNAPSHOT-tests.jar -s com.tfs.test.MyTest1` 
    
    `scala -J-Xmx2g -cp "scalatest_2.11-3.0.5.jar:scalactic_2.11-3.0.5.jar:try-scalatest-1.0-SNAPSHOT.jar" org.scalatest.tools.Runner -o -R try-scalatest-1.0-SNAPSHOT-tests.jar -s com.tfs.test.MyTest2`
    
3. Running all tests together:
    `scala -J-Xmx2g -cp "scalatest_2.11-3.0.5.jar:scalactic_2.11-3.0.5.jar:try-scalatest-1.0-SNAPSHOT.jar" org.scalatest.tools.Runner -o -R try-scalatest-1.0-SNAPSHOT-tests.jar`
    
