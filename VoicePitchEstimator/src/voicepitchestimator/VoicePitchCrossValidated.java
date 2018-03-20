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
import java.lang.Math.*;
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
public class VoicePitchCrossValidated {

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
        int vr=0, ur=0, vu=0, uv=0, nfine=0;
        double vv=0, err=0, ge=0, fe=0;
        String result;

        ConverterUtils.DataSource classsources = new ConverterUtils.DataSource(classbase);
        ConverterUtils.DataSource pitchsources = new ConverterUtils.DataSource(pitchbase);
        Instances classinstances = classsources.getDataSet();
        Instances pitchinstances = pitchsources.getDataSet();
        Instances classtrain = classsources.getDataSet();
        Instances classtest = classsources.getDataSet();
        Instances pitchtrain = pitchsources.getDataSet();
        Instances pitchtest = pitchsources.getDataSet();
        classtrain.clear();
        classtest.clear();
        pitchtrain.clear();
        pitchtest.clear();
        classinstances.setClassIndex(12);
        pitchinstances.setClassIndex(12);
        classtrain.setClassIndex(12);
        classtest.setClassIndex(12);
        pitchtrain.setClassIndex(12);
        pitchtest.setClassIndex(12);
        
        Classifier voiceDetection;
        Classifier pitchEstimator;
        
        voiceDetection = new weka.classifiers.trees.RandomForest();
        pitchEstimator = new weka.classifiers.trees.RandomForest();
        voiceDetection.buildClassifier(classinstances);
        pitchEstimator.buildClassifier(pitchinstances);
        
        Evaluation classevaluation = new Evaluation(classinstances);
        Random x = new Random(1);
        
        classevaluation.crossValidateModel(voiceDetection,classinstances,10,x);
        result = classevaluation.toClassDetailsString();
        System.out.println("Quick preliminary results given by 10fold cross validation");
        System.out.println(result);
          
        voiceDetection = new weka.classifiers.trees.RandomForest();
        pitchEstimator = new weka.classifiers.trees.RandomForest();
        
        //Training voice classifier----------------------------------------------------
        for(int i=0; i<classinstances.numInstances()-2;i+=2){
            classtrain.add(classinstances.instance(i));
            classtest.add(classinstances.instance(i+1));
        }
        
        //Training pitch estimator --------------------------------------------------
        for(int i=0; i<pitchinstances.numInstances()-2;i+=2){
            pitchtrain.add(pitchinstances.instance(i));
            pitchtest.add(pitchinstances.instance(i+1));
        }
       
        //removing 0s from pitch estimator trainer to reduce misleading training
        for(int i=pitchtrain.numInstances()-1;i>=0;i--)
            if(pitchtrain.get(i).classValue()==0)
                pitchtrain.remove(i);
        
        voiceDetection.buildClassifier(classtrain);
        pitchEstimator.buildClassifier(pitchtrain);
        
        System.out.println("\nResults given by dividing half database for train and half for test");
        System.out.println("Total: " +classinstances.numInstances()+" "+pitchinstances.numInstances());
        System.out.println("Train: " +classtrain.numInstances()+" "+pitchtrain.numInstances());
        System.out.println("Test: "+ classtest.numInstances()+" "+pitchtest.numInstances());
        System.out.println("**The pitch training set is smaller because 0s were removed to avoid misleading training");
        //double[] pitch = pitchtest.attributeToDoubleArray(11);
        
        for(int i=0; i<classtest.numInstances();i++){
            if(voiceDetection.classifyInstance(classtest.get(i))==0){
                if(classtest.get(i).classValue()==1){
                    uv++;
                    vr++;
                }else
                    ur++;
            }else{
                if(classtest.get(i).classValue()==0){
                    vu++;
                    ur++;
                }else{
                    err = Math.abs(pitchEstimator.classifyInstance(pitchtest.get(i))-pitchtest.get(i).classValue())/pitchtest.get(i).classValue();
                    //err = Math.abs(pitch[i]-pitchtest.get(i).classValue())/pitchtest.get(i).classValue();
                    if(err>.2)
                        ge++;
                    else{
                        nfine++;
                        fe+=err*err;
                    }
                    vv++;
                    vr++;
                }
            }
        }
        if(nfine>0)
            fe=Math.sqrt(fe/((double)nfine));
        
        System.out.println("Num frames: "+(vr+ur)+" = "+vr+" voiced + "+ur+" unvoiced.");
        System.out.println("Unvoiced frames as voiced: "+uv+"/"+ur+" ("+100*(double)((double)uv/(double)ur)+"%).");
        System.out.println("Voiced frames as unvoiced: "+vu+"/"+vr+" ("+100*(double)((double)vu/(double)vr)+"%).");
        System.out.println("Gross voiced errors (+20%): "+ge+"/"+vv+" ("+100*(double)(ge/vv)+"%).");
        System.out.println("MSE of fine error: "+100*fe+"% .");
    }
}
