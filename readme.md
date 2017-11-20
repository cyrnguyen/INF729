# Clone repository
```
git clone git@github.com:cyrnguyen/INF729.git
cd INF729
```

# Configuration
## Preprocessing
The input directory and filename are defined in the file src/main/scala/com/sparkProject/Preprocessor.scala :
* directory : variable _filePath_
* name : variable _fileName_

The cleaned file is written in the same directory.

## Trainer
The input directory of the cleaned file is defined in the file src/main/scala/com/sparkProject/Trainer.scala :
* directory : variable _filePath_

The generated model is written in the same directory.

# Launch application 
```
./build_and_submit.sh Preprocessor _pathToSpark_ _masterIp_
./build_and_submit.sh Trainer _pathToSpark_ _masterIp_
```  
