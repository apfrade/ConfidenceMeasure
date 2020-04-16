try:
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "In order to run the model you must have the libraries: " +
        "`numpy`, `pandas`, and 'pickle' installed.")

def prediction_confidence(model_file_path, X):

    # unwrap the model and predicts examples
    model = joblib.load(model_file_path)
    prob = [np.around(x,3) for x in model.predict_proba([X])]

    # build label_dic
    label_dic = {}
    for i in range(len(model.classes_)): label_dic[i]=model.classes_[i] 
    
    confidence = max(prob[0]) - max(sorted(prob[0],reverse=True)[1:])
    label = label_dic[np.argmax(prob[0])]
    
    return label, confidence

def load_models(model_file_path, X):
    
    ''' A function that predicts class probabilities.
    
    It loads a predictive model and predicts a set of examples, outputting the probabilities
    of them to belong to each class.
    
    Parameters
    ----------
    model_file_path: str
        The path to the pickled file of the model to use. This must be a classifier from sklearn, with the predict_proba() attribute.
    
    X: pandas dataframe
        The dataframe containing the scaled descriptors to use for label prediction. 
    
    Returns
    -------
    prob: ndarray
        The array containing the arrays of probabilities for each example.
    
    label_dic: dictionary
        The dictionary of labels.
    '''

    # unwrap the model and predicts examples
    model = joblib.load(model_file_path)
    prob = [np.around(x,3) for x in model.predict_proba(X)]

    # build label_dic
    label_dic = {}
    for i in range(len(model.classes_)): label_dic[i]=model.classes_[i]

    return prob, label_dic

def confident_guesses_calc(prob, ct, label_dic):
    
    ''' A function that identifies confident predictions.
    
    It analyses the class assigment probabilities of each example and determines whether the
    margin between the two largest probabilities in the array is of at least the size of ct.
    
    Parameters
    ----------
    prob: ndarray
        The array containing the arrays of probabilities for each example.
        
    ct: float
        The confident threshold to be used. This can vary between 0 and 1.
    
    label_dic: dictionary
        The dictionary of labels.
    
    Returns
    -------
    solution: 1d array
        The array containing predictions.    
    '''

    # define confident guesses
    solution = []
    for prediction in prob:
        if max(prediction) - ct >= max(sorted(prediction,reverse=True)[1:]): 
            solution.append(label_dic[np.argmax(prediction)])
        else: 
            solution.append('Not Confident')

    return solution

def X_update(X, ct, ids, solution, res_dic):
    
    ''' A function that identifies the examples that could not be predicted with confidence.
    
    It updates the pool of examples only to those that could not been predicted with confidence.
    
    Parameters
    ----------
    X: pandas dataframe
        The dataframe containing the scaled descriptors to use for label prediction. 
    
    ct: float
        The confident threshold to be used. This can vary between 0 and 1.
    
    ids: pandas series
        The series containing the examples IDs.
    
    solution: 1d array
        The array containing predictions.
    
    res_dic: dictionary
        The dictionary that stores confident predictions. It is the form of res_dic[ct]=[(prediction, example_id), ...]
    
    Returns
    -------
    X: pandas dataframe
        Updated version of dataframe containing the scaled descriptors of examples that could have not been predicted with confidence.
   
    ids: pandas series
        Updated version of the series containing the examples IDs that could have not been predicted with confidence.
    
    res_dic: dictionary
        Updated res_dic with confident predictions.'''
    
    indices = []
    confident_pred = []
    
    for i in range(len(X)):
        
        if solution[i] == 'Not Confident': indices.append(i)
        else: confident_pred.append((solution[i], ids.values[i]))
    
    # updates
    X = X.iloc[indices,:]    #### do I need to reset the indices of X and ids ???
    ids = ids.iloc[indices]
    res_dic[ct] = confident_pred

    return X, ids, res_dic

def confidence_measure(clf_ct_list, ids, X):
    
    ''' A wrapper to apply confidence thresholds.
    
    It applies confidence thresholds to any classification model (build from sklearn) that is able to
    produce class probabilities (compatible with the method <.predict_proba()>).
    
    Parameters
    ----------
    clf_ct_list: list of tuples
        The list of the type [(model_file_name, confidence_threshold)].
    
    X: pandas dataframe
        The dataframe containing the scaled descriptors to use for label prediction. 
    
    Returns
    -------
    res: dictionary
        Dictionary of confident predictions. This is of the type res[ct]=[(prediction, example_id), ...]
   
    ids: pandas series
        The series containing the examples IDs that could have not been predicted with confidence.
   '''
    
    # format data when predicting 1 example
    if len(X.values.shape) == 1: 

        ids = pd.Series(ids)
        X = X.values.reshape(1, -1)
        X = pd.DataFrame(X)
        
    
    res = {}
    total = len(ids)
    
    # run models with ct
    for clf_ct in clf_ct_list:
        
        clf_name = clf_ct[0]
        ct = clf_ct[1]

        # calculate class probabilities
        prob, label_dic = load_models(clf_name, X)

        # calculate confident guesses
        solution = confident_guesses_calc(prob, ct, label_dic)
        
        # update X
        X, ids, res = X_update(X, ct, ids, solution, res)
        
        # round report
        print('%s (ct %.2f) confident predictions: %i total' %(clf_name, ct, len(res[ct])))
        if len(X) == 0: 
            print('\nAll examples have been classified.')
            break    
            
    # overall report
    if len(X) != 0:  print('\nExamples that could not be determine: %i / %i' %(X.shape[0], total))
    
    return res, ids

def ct_studies(model, X_test, y_test):
    
    def load_models(model, X_test):
        model = joblib.load(model)
        prob = [np.around(x,3) for x in model.predict_proba(X_test)]
        label_dic = {}
        for i in range(len(model.classes_)): label_dic[i]=model.classes_[i]
        return prob, label_dic
    
    def confident_guesses_calc(prob, ct, label_dic):
        solution = []
        confident = 0
        not_confident = 0
        confidence_threshold = ct
        for prediction in prob:
            if max(prediction) - confidence_threshold >= max(sorted(prediction,reverse=True)[1:]): 
                solution.append(label_dic[np.argmax(prediction)])
                confident += 1
            else: 
                solution.append('Not Confident')
                not_confident += 1
        return solution, confident, not_confident
        
    def right_confident_guesses_calc(y_test, solution):
        indices = []
        pred_temp = [] 
        true_temp = []         
        right_guesses = 0
        test_examples = len(y_test)
        for i in range(test_examples):
            if solution[i] == 'Not Confident': indices.append(i)
            else: 
                pred_temp.append(solution[i])
                true_temp.append(y_test[i])
                if solution[i] == y_test[i]: right_guesses +=1
        return indices, pred_temp, true_temp, right_guesses
    
    def confidence_system(model, ct, X_test, y_test): 
        
        # initialize cumulative variables
        CG = 0
        RCG = 0
        overall_pred = []
        overall_true = []
        test_set_size = len(y_test)
        
        # calculate class probabilities
        prob, label_dic = load_models(model, X_test)

        # calculate confident guesses
        solution, confident, not_confident = confident_guesses_calc(prob, ct, label_dic)
        CG += confident

        # calculate right confident guesses
        indices, pred_temp, true_temp, right_guesses = right_confident_guesses_calc(y_test, solution)
        RCG += right_guesses
        overall_pred += pred_temp
        overall_true += true_temp 
            
        if CG != 0: res = [CG, RCG, round(CG/test_set_size, 2), round(RCG/CG,2)]
        else: res = [CG, RCG, round(CG/test_set_size, 2), 0]
            
        return res


    def ct_plots(cts, percen, accu, y_test):
        
        plt.plot(cts, accu, 'bo', label='Prediction accuracy')
        plt.plot(cts, percen, 'r*', label='Test set predicted')
        plt.ylabel('%')
        plt.xticks(cts)
        plt.xlabel('Confidence Threshold')
        plt.legend(loc='center right')

    
    # initialize variables
    accu = []
    percen = []
    cts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    
    # apply cts
    for ct in cts:
        
        res = confidence_system(model, ct, X_test, y_test)
        percen.append(res[2]*100)
        accu.append(res[3]*100)

    # store info
    df = pd.DataFrame({'Confident Threshold': cts, '% test set predicted': percen, 'Prediction accuracy': accu})
    print('\nSummary:\n',df,'\n')
    
    # plot
    ct_plots(cts, percen, accu, y_test)