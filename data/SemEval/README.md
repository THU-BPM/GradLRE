# SemEval2010-Task8

There are 8000 sentences in train set and 2717 sentences in test set.  

## train
- FULL_TRAIN.txt 
- train.txt
- train_result.txt
- train_result_full.txt

e.g.  
FULL_TRAIN.txt  
  &emsp;&emsp;73&emsp;	"The <e1>fire</e1> inside WTC was caused by exploding <e2>fuel</e2>."  
  &emsp;&emsp;Cause-Effect(e2,e1)  
  &emsp;&emsp;Comment:  
train.txt  
  &emsp;&emsp;73&emsp;	The <e1>fire</e1> inside WTC was caused by exploding <e2>fuel</e2>  
train_result.txt  
  &emsp;&emsp;73&emsp;  Cause-Effect  
train_result_full.txt  
  &emsp;&emsp;73&emsp;  Cause-Effect(e2,e1)  

## test
- FULL_TEST.txt
- test.txt
- test_result.txt
- test_result_full.txt

## scorer
two tools for SemEval-2010 Task #8
- official output file format checker : semeval2010_task8_format_checker.pl
- official scorer for SemEval-2010 Task #8 : semeval2010_task8_scorer-v1.2.pl

## original.zip  
the original data from the official website http://semeval2.fbk.eu/semeval2.php
