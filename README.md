# Confidence Measure Tool 

### Introduction  

Here we provide the confidence measure as a ready to use tool that can be wrapped around classification models.  

The confidence measure enables one to:  
- increases model robustness  
- quantify prediction trust  
- operate a classifier virtually up to any accuracy level  

The confidence measure can be used for one, or ################

When used for several rounds of predictions of decreasing thresholds, the idea is to feed into the next round all the examples that could not be predicted with confidence. The outputs of each round will fall into the prediction confidence interval limited by the threshold of the previous and current round. This allows great descrimination between the confidence of predictions of different examples.  

Nevertheless, a compromise between accuracy and access to answers is required for the achievement of useful results.

See the [reference](https://pubs.rsc.org/en/content/articlelanding/2020/ce/d0ce00111b#!divAbstract) for more details.

### Applications  

One could use this approach to:  
- Increase model robustness (by filtering marginal classification events due to noise in data).  
- Increase model performance and retrieve examples whose prediction confidence is at least x%.  
- Separate the examples by confidence of their prediction.  
- Determine in which confidence interval the prediction of an example falls.  

It can be further used for label propagation.

### Limitations  

The model to be used should be a classifier, able to output an array of class probabilities for each example. Currently, the confidence measure relies on the scikit-learn method ***clf.predict_proba()***.   

Thus we recomend you to only use scikit-learn classifiers that have the **predict_proba()** method. Some are:   

		sklearn.svm.SVC(probability=True)
		sklearn.naive_bayes.GaussianNB()
		sklearn.ensemble.ExtraTreesClassifier()
		sklearn.ensemble.GradientBoostingClassifier()
		sklearn.linear_model.SGDClassifier(loss='modified_huber')
		sklearn.neighbors.KNeighborsClassifier()
		sklearn.gaussian_process.GaussianProcessClassifier()
	
If your classifier is not from scikit-learn, ensure that it is able to output arrays of class probabilities for each example. In this case you will also have to change the load_models script in the source code, to accommodate your model.


## Installation   

1. Ensure that you are operating with ***Python 3*** or above.

2. Then install the **dependencies**:  

	Dependency installation via pip:  

	    $ pip install pandas numpy

	Dependency installation via conda:  

	    $ conda install pandas numpy 
	
3. Finally install the confidence threshold tool:  

    	$ pip install -i https://test.pypi.org/simple/ ConfidenceMeasure


### References

**Please cite:**  

*A. P. Frade, P. McCabe and R. I. Cooper. “Increasing the performance, trustworthiness and practical value of machine learning models: a case study predicting hydrogen bond network dimensionalities from molecular diagrams”. 2020. CrystEngComm. DOI: 10.1039/D0CE00111B* [Reference](https://pubs.rsc.org/en/content/articlelanding/2020/ce/d0ce00111b#!divAbstract) for more details.


## Understanding the algorithm   

### Input:  

This tool can be called in a single line of code and it has 4 inputs: the classifier, the confidence threshold, a list of  identifiers as well as the matrix of descriptors for the examples to be predicted. These should be provided as described below:

**- model & confidence threshold:**  
    
   **Model:** the absolute path of a pickled file of the model to be used. Eg. C:\Desktop\model_1  
   **Confidence threshold:** a float between 0 and 1. Eg. 0.5  
    
   You must provide them as a list of tuples.   
    			
			Eg. [(C:\Desktop\model_1, 0.5)] 
    
    
   You may be interested in running different scenarios:  
    
    1. One model with one confidence threshold:                   
    
			Eg. [(model_1_file_path, ct_1)] 
    
    2. One model over different round of confidence thresholds:   
    
			Eg. [(model_1_file_path, ct_1), (model_1_file_path, ct_2)] 
    
    3. Different models with different confidence thresholds: 
    
			Eg. [(model_1_file_path, ct_1), (model_2_file_path, ct_2)] 


**- ids:**    
    This should be a pandas Series containing the identifiers of the examples to be predicted. This is required, so the algorthim is able to handle the instances that can and cannot be predicted.  

**- descriptors:**    
    This sould be a pandas dataframe containing the descriptor values associated with the examples to be predicted.  
    The nth row of the descriptor table should correspond to the nth identfier in the list of ids.  

### Output:  

The tool outputs the confidence predictions and the unpredicted examples:

**- confident_predictions:**  
   This is a dictionary of the type
    
			dict[ct] = [(predicted label 1, example id 2), (predicted label 2, example id 2), ...] 

   *ct* is the key of each dictionary entry and corresponds to the confidence threshold used
   *[(predicted label, example id)]* is the list of results that the model confidently predicted for that ct. 
   *(predicted label, example id)* is a tuple of predicted label and corresponding example identifier

**- unpredicted_examples_ids:**    
    This is a list of example identifiers for which the model could not make any confident prediction.


## Basic tour

The [Basic Tour](https://github.com/apfrade/ConfidenceMeasure/blob/master/examples/basic_tour.ipynb) will walk you through the use and different application of the confidence tool.
