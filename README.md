# Characterization of Movement-related neural States in NHP's

This repository contains the scripts used to aggregate, pre-process, and analyze the EEG/LFP datasets. The project was supervised by Ignasi Cos at the University of Barcelona.

#### Abstract

The overall goal of the project was the principled characterization of neural states during movement execution before and after traumatic intervention (Muscimol inactivation
or stroke). To this end we developed a machine learning pipeline to perform
neural state classification using high dimensional EEG and LFP data. These neural
data were collected while NHP’s performed reward retrieval tasks involving reaching
and grasping. An intervention was performed in the portion of the brain affecting
movement of the right hand and the experiment was repeated. Three classification
features were analyzed: spectral power, and functional connectivity which consisted
of inter-signal covariance and correlation. These features were computed for
nine frequency bands. Multinomial logistic regression (MLR) and 1-nearest neighbor
(1-NN) classifiers were independently fit to each of the three types of classification
features for all nine frequency bands (27 combinations). Classification performance
was then analyzed in tasks of varying difficulty. MLR outperformed 1-NN
and achieved AUC scores of above 0.8 for all tasks when fit to the highest frequency
band power data. Lower frequency bands yielded considerably worse accuracies.
Functional connectivity features yielded lower accuracies than spectral power features,
though they far exceeded random chance. Finally, discriminative support networks
were generated to further characterize the movement-related states.

<p align="center"><img src="https://github.com/madepass/FinalMasterProject-MAD/blob/master/Report/figures/pipeline.PNG" align=middle width=800pt height=200/>
</p>
<p align="center">
<em>Pipeline for EEG/LFP classification.</em>
</p>


## Topics

> - Neuroscience
> - Machine Learning (ML)
> - EEG/LFP data

## Contributions
Contributions are welcome! For bug reports or requests please submit an [submit an issue](https://github.com/madepass/FinalMasterProject-MAD/issues).

## Contact
Feel free to contact me to discuss any issues, questions or comments.
* GitHub: [madepass](https://github.com/madepass)
* Linkedin: [Michael DePass](https://www.linkedin.com/in/michael-depass/)


## License

The content developed by Michael DePass is distributed under the following license:

    Copyright 2020 Michael DePass - UB
