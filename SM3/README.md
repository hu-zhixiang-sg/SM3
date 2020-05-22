# OML Project Notes

Need to submit:

* Project report
* Code (link to github)
* In-class presentation


## Paper:

 Memory Efficient Adaptive Optimization by Anil .et al
	link: https://arxiv.org/pdf/1901.11150.pdf

## Possible Datasets:

1. https://snap.stanford.edu/data/web-Amazon.html  

2. https://opendatascience.com/25-excellent-machine-learning-open-datasets/ 

3. https://medium.com/@ODSC/20-open-datasets-for-natural-language-processing-538fbfaf8e38 

4. There's a clean one: https://nijianmo.github.io/amazon/index.html 

## Report requirements:

*Optimization method studied

	*Pseudocode of the optimization method

  	*Short description of motivation for the development

  		-Claimed benefits listed in the paper

	-Experimental setup

  		-Datasets used

  		-Baseline algorithm used for comparison
			*can we test SM3 agains different opt algos?

  		-Hyperparameter choices

  		-Evaluation metrics

	-Experimental results

  		-Describe findings concisely and clearly

		-Critical evaluation of claims

  		-Critically evaluate paper's claims based off results

	-Conclusions from study

  		-Describe conclusions and inferences from results

Code:

	-Language: Python 
	-Available Libs: cvxopt, matplotlib, numpy, pandas, scipy,
			sklearn, statsmodels, pytorch, tensorflow
	-Using: tensorflow
		see how others are done:
		https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/training 
		-> check gradient_descent.py

	*Need to provide script producing results
	*Need to keep documentation in source file
	*Need to provide linkss to datasets
	*Need to compress to 1MB ZIP file

## SM3 Algorithm

The algorithm maintains a single variable for each set S_r in the cover

"in lage scale applications, k will be chosen to be negligible in comparison to d, which would 
translate to substantial savings in memory"

For each set S_r in the cover, the algorithm maintains a running sum mu_t(r), of the maximal
variance over all gradient entries j in S_r. Next, for each parameter i, we take the minimum
over all variables mu_t(r) associated with sets which cover i, denoted i in S_r

Thereafter, the learning rate corresponding to the i'th gradient entry is determined by taking 
the square root of this minimum denoted by v_t(i).
