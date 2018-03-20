# Speech Technologies. #
#### Master’s degree in Telecommunications Engineering (MET). Universitat Politècnica de Catalunya ####

Assignment: Pitch and voicing estimation  
Professor: José Adrián Rodríguez Fonollosa

Student: Alejandro Casar Berazaluce

The code provided extracts features to train a random forest for classification and has 2 ways of being used. But both require data to be in a hirerarchy as specified in the gui files. (I do not upload the data since you already have it and it just wastes space)
1. The first approach was intended to allow you to train the classifier with one database (say the ptdb_tug.gui) and then test it with another (say the fda_ue.gui). It can be done by following steps but my experiments were not using this approach since processing the ptdb_tug.gui was taking an extremely long amount of time.

a) To extract features from the ptdb and create the training database
> python3 pitch.py -p 0 -a 0 -f 10 ptdb_tug.gui

*-a is "action" and 0 is to perform feature extraction

b) Then, to train and create a random forest model
>run the "VoicePitchTrainer.java" located in .\VoicePitchEstimator\src\voicepitchestimator 
* I tried running it from command line with javac but it needed the weka package and I do not know how to include stuff from terminal, so I run it from a netbeans proyect

c) Then, to extract features from the fda_ue and create the testing database
> python3 pitch.py -a 0 fda_ue.gui

d) Then, to test the random forest with this database
>run the "VoicePitchTester.java" located in .\VoicePitchEstimator\src\voicepitchestimator 
* Same issue about weka package

e) Then, to generate the f0 files
> python3 pitch.py -a 1 fda_ue.gui

*-a is "action" and 1 is to create f0 files

f) Finally, to evaluate performance
> pitch_compare fda_ue.gui

**This approach can be used to train and test using the same database, but it results in a 100% accuracy due to overfitting and this is not representative of actual behavior. For this reason I created a 2nd approach.

2. This approach uses only 1 database for training and testing, but it performs multiple evaluation. First it performs a 10fold cross validation, and reports some basic performance metrics, then it splits half of the database to train the model and half to test it and report the metrics requested by the professor.

a) To extract features from the ptdb and create the training database
> python3 pitch.py -a 0 fda_ue.gui

b) Then, to train and test the random forest with this database
>run the "VoicePitchCrossValidated.java" located in .\VoicePitchEstimator\src\voicepitchestimator 
* Same issue about weka package
* The crossvalidated performs the evaluation so no need to run the python_compare afterwards

My experiments and report were done using the second approach using the fda_ue.gui database.
