# ConfidenceMeasure

The confidence measure enables one to (1) increases model robustness, (2) quantify prediction trust, and (3) operate a classifier virtually up to any accuracy level. 

Nevertheless, a compromise between accuracy and access to answers is required for the achievement of useful results.

Here we provide the confidence measure as a ready to use tool that can be wrapped around a classification model.

The confidence measure can be used for one, or several rounds of predictions of decreasing thresholds.

The idea is to feed into the next round all the examples that could not be predicted with confidence.

The outputs of each round will fall into the prediction confidence interval limited by the threshold of the previous and current round.

This allows great descrimination between the confidence of predictions of different examples.

One could use this approach to:

- Increase model robustness (by filtering marginal classification events due to noise in data).
    
- Increase model performance and retrieve examples whose prediction confidence is is at least x%.
    
- Separate the examples by confidence of their prediction.
    
- Determine in which confidence interval the prediction of an example falls.


Input:
- model_confidence_threshold_list: 
    This is a list of tuples. 
    
    The first element of the tuple is the name file of the predictive model to be used.
    
    The second element is the confidence threshold value to be used (between 0 and 1).

- ids:
    This is a list of the ids of the examples to be predicted.

- descriptors:
    This is a matrix of descriptors associated with the examples to be predicted. 
    
    The n row of descriptors in this matrix should correspond to the n example_id in the ids list.
    

Outputs:
- confident_predictions:
    This is a dictionary of the type 
    
        dict[ct] = [(predicted label, example id), ...]
    
    The key of each dictionary entry corresponds to the confidence threshold used, 
    
        dict[ct]
    
    Each key is associated with a list of prediction outputs that the model confidently predicted for that confidence threshold,
    
        [(tuple1), ..., (tuple2)]
    
    Each element of that list if a tuple of predicted labels and example id, 
    
        (predicted label, example id)

- unpredicted_examples_ids:
    This is a list of example ids for which the model could not make any confident prediction.


# Disclamer:

The model to be used should be a classifier, able to output an array of class assigment probabilities for each example.

Currently, the confidence measure relies on the scikit-learn method clf.predict_proba(). Thus we recomend you to only use:

   - sklearn classifiers that have a probability option: eg. sklearn.svm.SVC(probability=True)
   
   - possess the atribute: predict_proba

If your classifier is not from scikit-learn, ensure that it produce arrays of class class assigment probabilities for each example. In this case you will also have to change the load_models script in the source code, to accommodate your model.

See the reference for more details.


# Dependencies

The code should be run using Python 3.

    Pandas
    NumPy

Dependency installation via conda:

    $ conda install pandas numpy
	
# Installation

Via pip: (to be filled in)


# References

Paper under revision (link)
