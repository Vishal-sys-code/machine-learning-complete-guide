In sklearn there are 15 hyperparameters in the Logistic Regression
* **penalty:** {l1, l2, elasticnet, none}, default = 'l2'
* **dual:** bool, default = False
    - Always prefer, dual = False [number of rows > number of columns] -> Primal Formulation
* **tol:** float, default = 1e-4. [This is used for the stopping condition, it is the tolerance].
* **C:** float, default = 1.0. [C = 1/(lambda), inverse of regularization strength, Smaller values specify stronger regularization.]
* **fit_intercept:** bool, default = True.
* **intercept_scaling:** float, default = 1
* **class_weight:** dict or balanced, default = 1. [When we have imbalanced dataset, then we use this.]
* **solver:** {newton_cg, lbfgs, liblinear, sag, saga}. [default: lbfgs]
* **random_state:** int, RandomState instance, default = None
* **max_iter:** int, default = 100. [Number of epochs taken by algorithm to converge].
* **multi_class:** {auto, ovr, multinomial}, default = 'auto'.
    * ovr -> binary {for each classes there will be that number of log regression}
    * multinomial -> {as we discussed in notes}
* **verbose:** int, default = 0 {Seeing the results in between the training}
* **warm_start:** bool, default = False
    * [When set to True, it will start from the place where you left otherwise it will erase the previous solution.]
* n_jobs: int, default = None. [To use the cores of the CPU].
* l1_ratio: float, default = None.
