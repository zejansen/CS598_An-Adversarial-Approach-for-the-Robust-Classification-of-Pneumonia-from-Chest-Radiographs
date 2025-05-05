#!/usr/bin/env python
# train.py
import argparse
import sklearn.metrics
import random

from models import CXRClassifier, CXRAdvClassifier
from cxrdataset import CheXpertDataset, MIMICDataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))

def _train_standard(datasetclass, checkpoint_path, logpath, ignoreAP_PA=False):
    print(checkpoint_path)
    print(logpath)
    trainds = datasetclass(fold='train',exclude_view = ignoreAP_PA)
    valds = datasetclass(fold='val',exclude_view = ignoreAP_PA)
    testds = datasetclass(fold='test',exclude_view = ignoreAP_PA)
  
    classifier = CXRClassifier()
    classifier.train(trainds,
                valds,
                max_epochs=100,
                lr=0.01, 
                weight_decay=1e-4,
                logpath=logpath,
                checkpoint_path=checkpoint_path,
                verbose=True)
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:,pneumonia_index]
    true_pneumonia = true[:,pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
            true_pneumonia,
            probs_pneumonia)
    print("area under ROC curve of pneumonia: {:.04f}".format(auroc))
    
def _train_adversarial(datasetclass, checkpoint_path, logpath, ignoreAP_PA=False):
    trainds = datasetclass(fold='train')
    valds = datasetclass(fold='val')
    testds = datasetclass(fold='test')
    print(checkpoint_path)
    
    
    classifier = CXRAdvClassifier()
    classifier.train(trainds,
                valds,
                lr=0.01, 
                max_epochs=30, # TO DO - maybe reduce this. 
                weight_decay=1e-4,
                logpath=logpath,
                checkpoint_path=checkpoint_path,
                verbose=True)
    probs = classifier.predict(testds)
    true = testds.get_all_labels()

    # find the label index corresponding to pneumonia
    pneumonia_index = _find_index(testds, 'pneumonia')
    probs_pneumonia = probs[:,pneumonia_index]
    true_pneumonia = true[:,pneumonia_index]
    auroc = sklearn.metrics.roc_auc_score(
            true_pneumonia,
            probs_pneumonia)
    print("area under ROC curve of pneumonia: {:.04f}".format(auroc))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', action="store", default='MIMIC')
    parser.add_argument('training', action="store", default='Standard')
    parser.add_argument('ignoreAP_PA', action="store",default = False)
    
    args = parser.parse_args()
    ignoreview = False

    print("in main of training")
    
    if args.dataset == 'MIMIC' and args.training =='Standard':
        _train_standard(MIMICDataset, 'mimic_standard_model.pkl', 'mimic_standard.log', args.ignoreAP_PA)
    elif args.dataset == 'CheXpert' and args.training =='Standard':
      print("Standard CheXpert")
      if args.ignoreAP_PA == 'False':
        "Considering VIEW"
        ignoreview = False
        modelP = 'chexpert_standard_model.pkl'
        logP = 'chexpert_standard.log'
      else:
        "IGNORING VIEW"
        ignoreview=True
        modelP = 'chexpert_standard_model_ignore_view.pkl'
        logP='chexpert_standard_ignore_view.log'
      _train_standard(CheXpertDataset, modelP, logP,ignoreview)
    elif args.dataset == 'CheXpert' and args.training =='Adversarial':
        _train_adversarial(CheXpertDataset, 'chexpert_adversarial_model.pkl', 'chexpert_adversarial.log')
    elif args.dataset == 'MIMIC' and args.training =='Adversarial':
        print('MIMIC Lacks AP/PA labels, can not do adversarial training.')
    else:
        print('arguments not understood')
        
if __name__ == '__main__':
    main()
        
        
        