# try-scalatest
Project to test out running automated tests with Scalatest.
Specifically, meant to show a problem I am facing.

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
    
    This is currently failing with the following message:
    <pre>
    Discovery starting.
    *** RUN ABORTED ***
      java.lang.IllegalAccessError: class org.xml.sax.helpers.SecuritySupport12 cannot access its superclass org.xml.sax.helpers.SecuritySupport
      at java.lang.ClassLoader.defineClass1(Native Method)
      at java.lang.ClassLoader.defineClass(ClassLoader.java:763)
      at java.security.SecureClassLoader.defineClass(SecureClassLoader.java:142)
      at java.net.URLClassLoader.defineClass(URLClassLoader.java:467)
      at java.net.URLClassLoader.access$100(URLClassLoader.java:73)
      at java.net.URLClassLoader$1.run(URLClassLoader.java:368)
      at java.net.URLClassLoader$1.run(URLClassLoader.java:362)
      at java.security.AccessController.doPrivileged(Native Method)
      at java.net.URLClassLoader.findClass(URLClassLoader.java:361)
      at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
    </pre>
  
