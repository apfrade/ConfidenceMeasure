﻿# Confidence Tool 

### Introduction  

Welcome to the confidence tool page.  

The confidence tool measures how confident a given classifier is about each prediction it makes. The tool can be implemented with a single line of code, and may be applied in a number of ways that will increase the capabilities of your original machine learning model.        
If you just like to see how it can be applied in practice, keep reading and check out our [Basic Tour](https://github.com/apfrade/ConfidenceMeasure/blob/master/examples/basic_tour.ipynb) in a jupyter notebook.  

To actually understand how it works, check the [reference](https://pubs.rsc.org/en/content/articlelanding/2020/ce/d0ce00111b#!divAbstract) for more details.  


### Applications  

One could use this tool to:  
- Quantifying the confidence of each prediction.  
- Increasing model robustness, by filtering marginal classification events due to noise in data.  
- Operating the classifier (virtually) up to any accuracy level.   
- Binning examples according to the confidence of their prediction.    

Check how to achieve each of these applications [here](https://github.com/apfrade/ConfidenceMeasure/blob/master/examples/basic_tour.ipynb).  

The tool is very versatile and may also be used as an ensemble framework, or even as a semi supervised learning strategy for automatic label assigment (self learning).  


### Limitations  

This tool is only compatible to **classifiers** that are able to output **class probabilities** for each prediction. Currently, the confidence tool relies on scikit-learn models that have a *predict_proba()* attribute. The majority is listed below:

		sklearn.svm.SVC(probability=True)
		sklearn.naive_bayes.GaussianNB()
		sklearn.ensemble.ExtraTreesClassifier()
		sklearn.ensemble.GradientBoostingClassifier()
		sklearn.linear_model.SGDClassifier(loss='modified_huber')
		sklearn.neighbors.KNeighborsClassifier()
		sklearn.gaussian_process.GaussianProcessClassifier()
	
If your classifier is not from scikit-learn, ensure that it is able to output arrays of class probabilities for each example. In this case you will also have to change the load_models script in the source code, to accommodate your model.


### Installation   

1. Ensure that you are operating with ***Python 3*** or above.

2. Install the **dependencies**:  

	Via pip:  
	
	    $ pip install pandas numpy matplotlib

	Via conda:  
	
	    $ conda install pandas numpy matplotlib
	
3. Install the **confidence tool**:  

    	$ pip install -i https://test.pypi.org/simple/ confidence-tool


### References

**Please cite us:**  

*A. P. Frade, P. McCabe and R. I. Cooper. “Increasing the performance, trustworthiness and practical value of machine learning models: a case study predicting hydrogen bond network dimensionalities from molecular diagrams”. 2020. CrystEngComm. DOI: 10.1039/D0CE00111B* 

Access the paper [here](https://pubs.rsc.org/en/content/articlelanding/2020/ce/d0ce00111b#!divAbstract).
