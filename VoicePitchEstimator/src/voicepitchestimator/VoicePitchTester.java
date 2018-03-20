/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package voicepitchestimator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
/**
 *
 * @author Alex Casar
 */
public class VoicePitchTester {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        try{
            convertToArff("../data/classbase.csv","../data/classbase.arff");
            convertToArff("../data/pitchbase.csv","../data/pitchbase.arff");
            classifier("../data/classbase.arff","../data/pitchbase.arff");
        }catch (Exception e){
            System.out.println(e.toString());
        }
    }
    
    public static void convertToArff(String source, String dest) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(source));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(dest));  
        saver.writeBatch();
    }
    
    public static void classifier(String classbase, String pitchbase) throws Exception{
        String voice = "../data/voice.model";
        String pitch = "../data/pitch.model";
        String dest = "../data/full.f0";
        BufferedWriter writer;
        File file;
        file = new File(dest);
        writer = new BufferedWriter(new FileWriter(file));

        ConverterUtils.DataSource classsources = new ConverterUtils.DataSource(classbase);
        ConverterUtils.DataSource pitchsources = new ConverterUtils.DataSource(pitchbase);
        Instances classinstances = classsources.getDataSet();
        Instances pitchinstances = pitchsources.getDataSet();
     
        classinstances.setClassIndex(12);
        pitchinstances.setClassIndex(12);
        Classifier voiceDetection = (Classifier) weka.core.SerializationHelper.read(voice);
        Classifier pitchEstimator = (Classifier) weka.core.SerializationHelper.read(pitch);
        
        for(int i=0; i<classinstances.numInstances();i++){
            if(voiceDetection.classifyInstance(classinstances.get(i))==0){
                writer.write(String.valueOf(0));
            }else{
                writer.write(String.valueOf(pitchEstimator.classifyInstance(pitchinstances.get(i))));
            }
            writer.newLine();
        }
        writer.close();
    }
}
